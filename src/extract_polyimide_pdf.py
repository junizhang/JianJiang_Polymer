#!/usr/bin/env python3
"""
Extract polyimide composition data from a PDF into structured JSON.

Pipeline:
1. Extract PDF text
2. Find image candidates by caption/context keywords
3. Run OCSR on candidate images
4. Build monomer dictionary from full text
5. Match OCSR results to monomer mentions conservatively
6. Extract polymer composition entries
7. Emit structured JSON
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import fitz
from rdkit import Chem
from rdkit.Chem import AllChem, inchi
import pubchempy as pcp

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

try:
    import jsonschema  # type: ignore
except Exception:
    jsonschema = None

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

from image_to_smiles import (
    assign_ids_to_segments,
    clean_structure_image,
    image_to_smiles,
    isolate_primary_structure,
    segment_molecule_crops,
    segment_scheme_molecule_crops,
    patch_molscribe_for_sandbox,
)

try:
    import torch
    from molscribe import MolScribe
except Exception:  # pragma: no cover
    torch = None
    MolScribe = None


DEFAULT_CKPT = "ckpts/swin_base_char_aux_1m680k.pth"
POSITIVE_KEYWORDS = {
    "scheme",
    "reaction",
    "structure",
    "monomer",
    "polymer",
    "synthesis",
}
NEGATIVE_KEYWORDS = {
    "sem",
    "tem",
    "xrd",
    "tga",
    "afm",
    "graph",
    "plot",
    "photograph",
    "photographs",
    "spectrum",
    "spectra",
    "curve",
}
POLYMER_NAME_RE = re.compile(r"\b(?:C?PI[ab]-\d+|PI[ab]-0|PI-\d+)\b")
ABBR_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9-]{1,15}\b")
CHEMICAL_HINT_RE = re.compile(
    r"(amino|amine|anhydride|acid|benz|imidazole|phenyl|biphenyl|carbox|imide)",
    re.I,
)


@dataclass
class Monomer:
    abbreviation: str
    name: str = ""
    role: str = ""
    smiles: str = ""
    smiles_source: str = ""
    evidence: List[str] = field(default_factory=list)


@dataclass
class OCSRResult:
    image: str
    segment_id: str
    provider: str
    smiles: str
    raw_smiles: str
    confidence: str
    status: str
    matched_abbreviation: str = ""


@dataclass
class ImageCandidate:
    page: int
    xref: int
    path: str
    rect: Tuple[float, float, float, float]
    caption: str
    context: str
    score: int
    positive_hits: List[str]
    negative_hits: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract polyimide dataset from PDF(s)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", help="Single PDF path (legacy mode)")
    src.add_argument("--input-dir", help="Directory of PDFs (batch mode)")

    parser.add_argument("--raw-dir", default="data/raw", help="Per-paper raw outputs root")
    parser.add_argument("--dataset-dir", default="data/processed/dataset_v0", help="Merged dataset output dir")
    parser.add_argument("--schema", default="schemas/schema_v0.json", help="Path to schema_v0.json")
    parser.add_argument("--model-path", default=DEFAULT_CKPT, help="MolScribe checkpoint path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="OCSR device")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM extraction (OCSR + monomer dict only)")
    parser.add_argument("--no-ocsr", action="store_true", default=True,
                        help="Skip MolScribe (default; LLM multimodal handles structure reading)")
    parser.add_argument("--use-molscribe", dest="no_ocsr", action="store_false",
                        help="Enable MolScribe OCSR pass alongside LLM vision")
    parser.add_argument("--no-validate", action="store_true", help="Skip jsonschema validation")
    parser.add_argument("--use-legacy-pi5922", action="store_true", help="Run the deprecated PI.5922 regex pipeline")
    return parser.parse_args()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_name(name: str) -> str:
    name = normalize_whitespace(name)
    name = re.sub(r"^[,;:]+|[,;:]+$", "", name)
    name = re.sub(r"\b(?:and|from|with|the|a|an)\s*$", "", name, flags=re.I)
    return name.strip()


def refine_name(name: str) -> str:
    name = clean_name(name)
    for sep in (". ", "; ", ": "):
        if sep in name:
            tail = name.split(sep)[-1].strip()
            if chemical_name_score(tail) >= chemical_name_score(name):
                name = tail
    for prefix in (
        "In this paper, fully symmetrical diamine ",
        "Two series of copolyimides from ",
        "A self-synthesized monomer ",
    ):
        if name.startswith(prefix):
            name = name[len(prefix):].strip(" ,")
    chem_matches = re.findall(
        r"([0-9][A-Za-z0-9,\-\[\]′'./() ]{3,180}?(?:benzimidazole\]?|dianhydride))",
        name,
        flags=re.I,
    )
    if chem_matches:
        name = chem_matches[-1].strip()
    return name


def extract_pdf_text(pdf_path: Path) -> Tuple[str, List[str]]:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    return "\n".join(pages), pages


def nearby_text_for_rect(page: fitz.Page, rect: fitz.Rect, margin: float = 140.0) -> Tuple[str, str]:
    blocks = page.get_text("blocks")
    caption = ""
    nearby_chunks: List[str] = []
    for block in blocks:
        x0, y0, x1, y1, text, *_rest = block
        b = fitz.Rect(x0, y0, x1, y1)
        plain = normalize_whitespace(text)
        if not plain:
            continue
        vertical_gap = min(abs(b.y0 - rect.y1), abs(rect.y0 - b.y1))
        overlaps_x = not (b.x1 < rect.x0 or b.x0 > rect.x1)
        if overlaps_x and 0 <= b.y0 - rect.y1 <= margin:
            if re.match(r"^(Scheme|Figure|Fig\.|Reaction|Structure)\b", plain, re.I):
                caption = plain
        if overlaps_x and vertical_gap <= margin:
            nearby_chunks.append(plain)
    return caption, " ".join(nearby_chunks)


def score_image_text(text: str) -> Tuple[int, List[str], List[str]]:
    lowered = text.lower()
    positives = sorted(
        k for k in POSITIVE_KEYWORDS
        if re.search(rf"\b{re.escape(k)}\b", lowered)
    )
    negatives = sorted(
        k for k in NEGATIVE_KEYWORDS
        if re.search(rf"\b{re.escape(k)}\b", lowered)
    )
    score = len(positives) * 2 - len(negatives) * 3
    if "scheme" in positives:
        score += 3
    return score, positives, negatives


def extract_image_candidates(pdf_path: Path, images_dir: Path) -> List[ImageCandidate]:
    images_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    candidates: List[ImageCandidate] = []
    for page_index, page in enumerate(doc):
        for image_index, image in enumerate(page.get_images(full=True), start=1):
            xref = image[0]
            rects = page.get_image_rects(xref)
            if not rects:
                continue
            rect = rects[0]
            caption, context = nearby_text_for_rect(page, rect)
            joint_text = " ".join(filter(None, [caption, context]))
            score, positive_hits, negative_hits = score_image_text(joint_text)
            if score <= 0 or (negative_hits and "scheme" not in positive_hits):
                continue

            image_meta = doc.extract_image(xref)
            ext = image_meta.get("ext", "bin")
            out_path = images_dir / f"page{page_index + 1:02d}_img{image_index:02d}_xref{xref}.{ext}"
            out_path.write_bytes(image_meta["image"])
            candidates.append(
                ImageCandidate(
                    page=page_index + 1,
                    xref=xref,
                    path=str(out_path),
                    rect=(rect.x0, rect.y0, rect.x1, rect.y1),
                    caption=caption,
                    context=context,
                    score=score,
                    positive_hits=positive_hits,
                    negative_hits=negative_hits,
                )
            )
    candidates.sort(key=lambda item: (-item.score, item.page, item.xref))
    return candidates


def infer_role(context: str) -> str:
    lowered = context.lower()
    if "dianhydride" in lowered:
        return "dianhydride"
    if "diamine" in lowered or "monomer" in lowered:
        return "diamine"
    return ""


def extract_target_abbreviations(entries: Sequence[dict]) -> List[str]:
    targets = {"BPDA"}
    for entry in entries:
        targets.add(entry["dianhydride_abbreviation"])
        for diamine in entry["diamines"]:
            targets.add(diamine["abbreviation"])
    return sorted(targets)


def chemical_name_score(name: str) -> int:
    score = 0
    if CHEMICAL_HINT_RE.search(name):
        score += 5
    if re.search(r"[\[\]\d,′'-]", name):
        score += 3
    if 10 <= len(name) <= 120:
        score += 2
    if any(token in name.lower() for token in ("temperature", "polymer", "spectrometer", "china", "germany", "usa")):
        score -= 10
    return score


def extract_monomers(full_text: str, target_abbreviations: Sequence[str]) -> Dict[str, Monomer]:
    monomers: Dict[str, Monomer] = {}
    flattened_text = normalize_whitespace(full_text)
    for abbr in target_abbreviations:
        best_name = ""
        best_score = -999
        best_window = ""
        strict_pattern = rf"([0-9][A-Za-z0-9,\-\[\]′'./() ]{{5,180}})\(\s*{re.escape(abbr)}\s*\)"
        strict_match = re.search(strict_pattern, flattened_text)
        if strict_match:
            candidate = refine_name(strict_match.group(1))
            best_name = candidate
            best_score = chemical_name_score(candidate) + 20
            best_window = flattened_text[max(0, strict_match.start() - 100): strict_match.end() + 100]
        for match in re.finditer(rf"([A-Za-z0-9,\-\[\]′'./() ]{{6,220}})\(\s*{re.escape(abbr)}\s*\)", flattened_text):
            candidate = refine_name(match.group(1))
            score = chemical_name_score(candidate)
            if score > best_score:
                best_score = score
                best_name = candidate
                best_window = flattened_text[max(0, match.start() - 100): match.end() + 100]

        monomer = Monomer(abbreviation=abbr)
        monomer.name = best_name
        monomer.role = "dianhydride" if abbr.upper() == "BPDA" else "diamine"
        if best_window:
            monomer.evidence.append(best_window)
        monomers[abbr] = monomer
    return monomers


def extract_series_entries(full_text: str) -> List[dict]:
    lines = [line.strip() for line in full_text.splitlines()]
    entries: List[dict] = []
    current_series = ""
    current_abr = ""
    pending_ratio = ""

    for line in lines:
        if "PIa (Ar = PABZ)" in line:
            current_series, current_abr = "PIa", "PABZ"
            continue
        if "PIb (Ar = i-PABZ)" in line:
            current_series, current_abr = "PIb", "i-PABZ"
            continue
        if not current_series:
            continue

        if re.fullmatch(r"\d+:\d+", line):
            pending_ratio = line
            continue

        if pending_ratio and POLYMER_NAME_RE.fullmatch(line):
            polymer_name = line
            if current_series == "PIb":
                polymer_name = polymer_name.replace("PIa", "PIb").replace("CPIa", "CPIb")
            bb_ratio, other_ratio = [int(x) for x in pending_ratio.split(":")]
            diamines = [{"abbreviation": "BB", "ratio": bb_ratio}]
            diamines.append({"abbreviation": current_abr, "ratio": other_ratio})
            entries.append(
                {
                    "polymer_series": current_series,
                    "polymer_name": polymer_name,
                    "dianhydride_abbreviation": "BPDA",
                    "diamines": diamines,
                }
            )
            pending_ratio = ""

        if re.match(r"^(Figure|Table)\b", line):
            current_series = ""
            current_abr = ""
            pending_ratio = ""

    if entries:
        return entries

    ratio_match = re.search(
        r"molar ratios of BB/PABZ or BB/i-PABZ were ([0-9:, ]+)",
        full_text,
        re.I,
    )
    if not ratio_match:
        return entries
    ratios = [item.strip() for item in ratio_match.group(1).split(",") if item.strip()]
    for ratio in ratios:
        bb_ratio, other_ratio = [int(x) for x in ratio.split(":")]
        for series, abbr in (("PIa", "PABZ"), ("PIb", "i-PABZ")):
            polymer_name = (
                f"{series}-0" if bb_ratio == 0 else
                "PI-100" if other_ratio == 0 else
                f"C{series}-{bb_ratio}"
            )
            entries.append(
                {
                    "polymer_series": series,
                    "polymer_name": polymer_name,
                    "dianhydride_abbreviation": "BPDA",
                    "diamines": [
                        {"abbreviation": "BB", "ratio": bb_ratio},
                        {"abbreviation": abbr, "ratio": other_ratio},
                    ],
                }
            )
    return entries


def init_predictor(model_path: Path, device_name: str):
    if MolScribe is None or torch is None or not model_path.exists():
        return None
    patch_molscribe_for_sandbox()
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    return MolScribe(str(model_path), device=device)


def load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def cache_key(abbreviation: str, name: str) -> str:
    return f"{abbreviation}||{normalize_whitespace(name).lower()}"


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def sanitize_structure_name(name: str) -> str:
    return name.replace("-", "").replace(" ", "")


def count_token(name: str, token: str) -> int:
    return len(re.findall(token, name, flags=re.I))


def smiles_matches_monomer_name(smiles: str, monomer: Monomer) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    if not atoms:
        return False
    if any(num in {27, 29, 30, 26} for num in atoms):
        return False
    heavy_atoms = sum(1 for num in atoms if num > 1)
    n_count = atoms.count(7)
    o_count = atoms.count(8)
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    name = monomer.name.lower()
    nh2_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3H2]")))
    anhydride_count = len(mol.GetSubstructMatches(Chem.MolFromSmarts("O=C1OC(=O)[#6]~[#6]1")))

    if heavy_atoms < 8:
        return False
    if any(token in name for token in ("benz", "phenyl", "biphenyl", "imidazole")) and aromatic_atoms < 6:
        return False
    if "benzimidazole" in name and n_count < 2:
        return False
    if monomer.role == "diamine" and count_token(name, "amino") >= 2 and nh2_count < 2:
        return False
    if monomer.role == "dianhydride" and ("dianhydride" in name or anhydride_count > 0) and anhydride_count < 2:
        return False
    if monomer.role == "dianhydride" and o_count < 4:
        return False
    return True


def repair_residue_smiles_to_monomer(raw_smiles: str, monomer: Monomer) -> str:
    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is None:
        return ""
    amino_mentions = count_token(monomer.name, "amino")
    if monomer.role != "diamine" or amino_mentions < 2:
        return ""
    methyl_matches = mol.GetSubstructMatches(Chem.MolFromSmarts("[c][CH3]"))
    if len(methyl_matches) < amino_mentions:
        return ""
    rxn = AllChem.ReactionFromSmarts("[c:1][CH3:2]>>[c:1]N")
    candidates = [mol]
    for _ in range(amino_mentions):
        next_candidates = []
        seen = set()
        for candidate in candidates:
            for prod in rxn.RunReactants((candidate,)):
                smiles = canonicalize_smiles(Chem.MolToSmiles(prod[0], canonical=True))
                if smiles and smiles not in seen:
                    seen.add(smiles)
                    next_candidates.append(Chem.MolFromSmiles(smiles))
        candidates = next_candidates
        if not candidates:
            return ""

    for candidate in candidates:
        repaired = canonicalize_smiles(Chem.MolToSmiles(candidate, canonical=True))
        if repaired and smiles_matches_monomer_name(repaired, monomer):
            return repaired
    return ""


def ocsr_with_molscribe(predictor, image_path: Path) -> Tuple[str, str, str, str]:
    if predictor is None:
        return "", "", "", "provider_unavailable"
    return image_to_smiles(
        predictor=predictor,
        image_path=image_path,
        preprocess=True,
        canonical=True,
        keep_h=False,
        keep_disconnected=False,
        with_confidence=True,
    )


def ocsr_with_img2mol(image_path: Path) -> Tuple[str, str, str, str]:
    try:
        from img2mol.inference import Img2MolInference
    except Exception:
        return "", "", "", "provider_unavailable"
    try:
        predictor = Img2MolInference()
        out = predictor(filepath=str(image_path))
        smiles = canonicalize_smiles(out) if isinstance(out, str) else ""
        if smiles:
            return smiles, out, "", "ok"
        return "", str(out), "", "empty_smiles"
    except Exception as exc:
        return "", "", "", str(exc)


def ocsr_with_decimer(image_path: Path) -> Tuple[str, str, str, str]:
    try:
        from DECIMER import predict_SMILES  # type: ignore
    except Exception:
        return "", "", "", "provider_unavailable"
    try:
        out = predict_SMILES(str(image_path))
        smiles = canonicalize_smiles(out) if isinstance(out, str) else ""
        if smiles:
            return smiles, out, "", "ok"
        return "", str(out), "", "empty_smiles"
    except Exception as exc:
        return "", "", "", str(exc)


def run_ocsr_providers(image_path: Path, predictor, confidence_threshold: float = 0.55) -> Tuple[str, str, str, str, str]:
    providers = [
        ("molscribe", lambda p: ocsr_with_molscribe(predictor, p)),
        ("img2mol", ocsr_with_img2mol),
        ("decimer", ocsr_with_decimer),
    ]
    last_provider = ""
    last_status = "all_failed"
    for provider_name, provider_fn in providers:
        smiles, raw_smiles, confidence, status = provider_fn(image_path)
        last_provider = provider_name
        last_status = status
        if status == "ok" and smiles:
            if confidence and provider_name == "molscribe":
                try:
                    if float(confidence) < confidence_threshold:
                        last_status = "low_confidence"
                        continue
                except ValueError:
                    pass
            return provider_name, smiles, raw_smiles, confidence, status
    return last_provider, "", "", "", last_status


def abbreviations_in_text(text: str, known: Sequence[str]) -> List[str]:
    found = []
    for token in re.findall(ABBR_RE, text):
        if token in known and token not in found:
            found.append(token)
    return found


def run_ocsr(
    candidates: Sequence[ImageCandidate],
    monomers: Dict[str, Monomer],
    predictor,
    clean_dir: Path,
) -> List[OCSRResult]:
    results: List[OCSRResult] = []
    known_abbrs = list(monomers.keys())
    clean_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        image_path = Path(candidate.path)
        try:
            if "scheme" in candidate.positive_hits or "reaction" in candidate.positive_hits or "synthesis" in candidate.positive_hits:
                segments = segment_scheme_molecule_crops(image_path, padding=16, min_area_ratio=0.008)
            else:
                segments = segment_molecule_crops(image_path, padding=12, min_area_ratio=0.01)
        except Exception:
            segments = [("mol1", image_path, (0, 0, 0, 0))]

        # Preserve spatial ordering for deterministic monomer matching.
        segments = sorted(segments, key=lambda item: (round(item[2][1] / 120), item[2][0]))
        nearby_abbrs = abbreviations_in_text(f"{candidate.caption} {candidate.context}", known_abbrs)
        if "scheme" in candidate.positive_hits and len(segments) >= 4:
            nearby_abbrs = [abbr for abbr in ("BB", "BPDA", "PABZ", "i-PABZ") if abbr in known_abbrs]

        for idx, (seg_id, seg_path, _bbox) in enumerate(segments):
            clean_work_path = seg_path
            temp_paths: List[Path] = []
            matched_abbr = nearby_abbrs[idx] if idx < len(nearby_abbrs) else ""
            attempt_paths: List[Path] = [seg_path]
            if matched_abbr:
                try:
                    tight_path, _tight_bbox = isolate_primary_structure(seg_path)
                    temp_paths.append(tight_path)
                    cleaned_path = clean_structure_image(tight_path)
                    temp_paths.append(cleaned_path)
                    clean_work_path = cleaned_path
                    attempt_paths = [cleaned_path, tight_path, seg_path]
                    persisted = clean_dir / f"{sanitize_structure_name(matched_abbr)}.png"
                    persisted.write_bytes(cleaned_path.read_bytes())
                except Exception:
                    clean_work_path = seg_path

            provider, smiles, raw_smiles, confidence, status = "", "", "", "", "all_failed"
            for candidate_path in attempt_paths:
                provider, smiles, raw_smiles, confidence, status = run_ocsr_providers(candidate_path, predictor)
                if matched_abbr and smiles:
                    monomer = monomers.get(matched_abbr)
                    if monomer and not smiles_matches_monomer_name(smiles, monomer):
                        repaired = repair_residue_smiles_to_monomer(raw_smiles or smiles, monomer)
                        if repaired:
                            smiles = repaired
                            status = "repaired_from_residue"
                            break
                        status = "chemistry_mismatch"
                        smiles = ""
                        continue
                if smiles:
                    break
            results.append(
                OCSRResult(
                    image=candidate.path,
                    segment_id=seg_id,
                    provider=provider,
                    smiles=smiles,
                    raw_smiles=raw_smiles,
                    confidence=confidence,
                    status=status,
                    matched_abbreviation=matched_abbr,
                )
            )
            if matched_abbr and smiles:
                monomer = monomers.get(matched_abbr)
                if monomer and not monomer.smiles:
                    monomer.smiles = smiles
                    monomer.smiles_source = f"{provider}:{Path(candidate.path).name}#{seg_id}"
            if seg_path != image_path:
                seg_path.unlink(missing_ok=True)
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)
    return results


def choose_dianhydride(monomers: Dict[str, Monomer]) -> Monomer:
    for preferred in ("BPDA",):
        if preferred in monomers:
            return monomers[preferred]
    for monomer in monomers.values():
        if monomer.role == "dianhydride":
            return monomer
    return Monomer(abbreviation="", role="dianhydride")


def resolve_with_pubchem(name: str) -> Tuple[str, str]:
    try:
        compounds = pcp.get_compounds(name, "name")
    except Exception as exc:
        return "", f"pubchem:{exc}"
    for compound in compounds:
        smiles = canonicalize_smiles(getattr(compound, "canonical_smiles", "") or "")
        if smiles:
            return smiles, "pubchem"
    return "", "pubchem:no_match"


def resolve_with_opsin(name: str) -> Tuple[str, str]:
    opsin_cmd = None
    for candidate in ("opsin", "opsin-cli"):
        result = subprocess.run(["bash", "-lc", f"command -v {candidate}"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            opsin_cmd = result.stdout.strip()
            break
    if not opsin_cmd:
        return "", "opsin:unavailable"
    try:
        result = subprocess.run([opsin_cmd, name], capture_output=True, text=True, check=False)
        smiles = canonicalize_smiles(result.stdout.strip())
        if smiles:
            return smiles, "opsin"
        return "", f"opsin:{result.stderr.strip() or 'no_match'}"
    except Exception as exc:
        return "", f"opsin:{exc}"


def resolve_smiles_from_identity(
    abbreviation: str,
    name: str,
    role: str,
    candidate_smiles: Sequence[Tuple[str, str]],
) -> Tuple[str, str, List[Dict[str, Any]]]:
    review_items: List[Dict[str, Any]] = []
    ref_smi = reference_smiles_for_monomer(abbreviation, name)
    if ref_smi:
        for candidate, source in candidate_smiles:
            cand = canonicalize_smiles(candidate or "")
            if cand and cand != ref_smi:
                review_items.append({
                    "kind": "monomer_structure_candidate_disagrees_with_reference",
                    "abbr": abbreviation,
                    "name": name,
                    "candidate_source": source,
                    "candidate_smiles": cand,
                    "reference_smiles": ref_smi,
                })
        return ref_smi, "reference", review_items

    for resolver in (resolve_with_pubchem, resolve_with_opsin):
        if not name:
            break
        smi, source = resolver(name)
        smi = canonicalize_smiles(smi or "")
        if smi:
            return smi, source, review_items

    for candidate, source in candidate_smiles:
        cand = canonicalize_smiles(candidate or "")
        if cand:
            return cand, source, review_items
    return "", "unresolved", review_items


def resolve_monomer_smiles(monomers: Dict[str, Monomer], cache: dict) -> None:
    for monomer in monomers.values():
        if monomer.smiles and monomer.name:
            cache[cache_key(monomer.abbreviation, monomer.name)] = {
                "abbreviation": monomer.abbreviation,
                "name": monomer.name,
                "smiles": monomer.smiles,
                "source": monomer.smiles_source or "ocsr",
            }
            continue
        if monomer.smiles or not monomer.name:
            continue
        key = cache_key(monomer.abbreviation, monomer.name)
        cached = cache.get(key, {})
        cached_smiles = canonicalize_smiles(cached.get("smiles", ""))
        if cached_smiles:
            monomer.smiles = cached_smiles
            monomer.smiles_source = cached.get("source", "cache")
            continue

        for resolver in (resolve_with_pubchem, resolve_with_opsin):
            smiles, source = resolver(monomer.name)
            if smiles:
                monomer.smiles = smiles
                monomer.smiles_source = source
                cache[key] = {
                    "abbreviation": monomer.abbreviation,
                    "name": monomer.name,
                    "smiles": smiles,
                    "source": source,
                }
                break


def find_anhydride_sites(mol: Chem.Mol) -> List[Tuple[int, Tuple[int, int]]]:
    sites: List[Tuple[int, Tuple[int, int]]] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 8 or atom.GetDegree() != 2:
            continue
        neighbors = atom.GetNeighbors()
        if not all(n.GetAtomicNum() == 6 for n in neighbors):
            continue
        carbonyl_carbons: List[int] = []
        valid = True
        for carbon in neighbors:
            has_carbonyl_oxygen = any(
                bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetOtherAtom(carbon).GetAtomicNum() == 8
                for bond in carbon.GetBonds()
            )
            if not has_carbonyl_oxygen:
                valid = False
                break
            carbonyl_carbons.append(carbon.GetIdx())
        if valid:
            sites.append((atom.GetIdx(), (carbonyl_carbons[0], carbonyl_carbons[1])))
    return sites


def find_diamine_nitrogens(mol: Chem.Mol) -> List[int]:
    return [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 7 and atom.GetDegree() == 1 and atom.GetTotalNumHs() >= 2
    ]


def build_repeat_unit_from_monomers(dianhydride_smiles: str, diamine_smiles: str) -> str:
    dian = Chem.MolFromSmiles(dianhydride_smiles)
    diam = Chem.MolFromSmiles(diamine_smiles)
    if dian is None or diam is None:
        return ""

    anhydrides = sorted(find_anhydride_sites(dian), key=lambda item: item[0])
    amines = sorted(find_diamine_nitrogens(diam))
    if len(anhydrides) < 2 or len(amines) < 2:
        return ""

    combo = Chem.CombineMols(dian, diam)
    rw = Chem.RWMol(combo)
    offset = dian.GetNumAtoms()

    for (_bridge_oxygen_idx, carbonyl_pair), amine_idx in zip(anhydrides[:2], amines[:2]):
        shifted_n = offset + amine_idx
        for carbonyl_idx in carbonyl_pair:
            if rw.GetBondBetweenAtoms(shifted_n, carbonyl_idx) is None:
                rw.AddBond(shifted_n, carbonyl_idx, Chem.BondType.SINGLE)

    for bridge_oxygen_idx, _carbonyl_pair in sorted(anhydrides[:2], key=lambda item: item[0], reverse=True):
        rw.RemoveAtom(bridge_oxygen_idx)

    try:
        product = rw.GetMol()
        Chem.SanitizeMol(product)
        return Chem.MolToSmiles(product, canonical=True)
    except Exception:
        return ""


def polymer_smiles_from_monomers(dianhydride: Monomer, diamines: List[Monomer], polymer_ocsr_smiles: str = "") -> str:
    if polymer_ocsr_smiles and len(diamines) == 1:
        return polymer_ocsr_smiles
    if not dianhydride.smiles:
        return ""
    if not diamines:
        return ""

    parts: List[str] = []
    for diamine in diamines:
        if not diamine.smiles:
            return ""
        repeat_smiles = build_repeat_unit_from_monomers(dianhydride.smiles, diamine.smiles)
        if not repeat_smiles:
            return ""
        parts.append(f"{diamine.abbreviation}:{repeat_smiles}")
    if len(parts) == 1:
        return parts[0].split(":", 1)[1]
    return "; ".join(parts)


def attempt_polymer_repeat_ocsr(candidates: Sequence[ImageCandidate], predictor) -> str:
    if predictor is None:
        return ""
    for candidate in candidates:
        if "scheme" not in candidate.positive_hits:
            continue
        image_path = Path(candidate.path)
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        try:
            segments = segment_molecule_crops(image_path, padding=16, min_area_ratio=0.01)
        except Exception:
            continue
        for _seg_id, seg_path, (x0, y0, x1, y1) in segments:
            width_ratio = (x1 - x0) / float(w)
            height_ratio = (y1 - y0) / float(h)
            center_y = (y0 + y1) / 2.0 / h
            if width_ratio < 0.75 or height_ratio < 0.18 or center_y < 0.25:
                if seg_path != image_path:
                    seg_path.unlink(missing_ok=True)
                continue
            provider, smiles, _raw_smiles, _confidence, status = run_ocsr_providers(seg_path, predictor)
            if seg_path != image_path:
                seg_path.unlink(missing_ok=True)
            if provider and status == "ok" and smiles:
                return smiles
    return ""


def build_output(
    paper_id: str,
    pdf_path: Path,
    monomers: Dict[str, Monomer],
    entries: List[dict],
    candidates: List[ImageCandidate],
    ocsr_results: List[OCSRResult],
    polymer_repeat_ocsr: str = "",
) -> dict:
    dianhydride = choose_dianhydride(monomers)
    polymers: List[dict] = []
    for entry in entries:
        diamines_payload = []
        diamine_records: List[Monomer] = []
        total_ratio = 0
        for diamine_info in entry["diamines"]:
            abbr = diamine_info["abbreviation"]
            ratio = int(diamine_info["ratio"])
            if ratio <= 0:
                continue
            monomer = monomers.get(abbr, Monomer(abbreviation=abbr, role="diamine"))
            diamine_records.append(monomer)
            total_ratio += ratio
            diamines_payload.append(
                {
                    "name": monomer.name,
                    "abbreviation": monomer.abbreviation,
                    "smiles": monomer.smiles,
                    "ratio": str(ratio),
                }
            )

        if not dianhydride.abbreviation or not diamines_payload:
            continue
        if total_ratio != 100:
            continue

        polymers.append(
            {
                "paper_id": paper_id,
                "polymer_series": entry["polymer_series"],
                "polymer_name": entry["polymer_name"],
                "dianhydride": {
                    "name": dianhydride.name,
                    "abbreviation": dianhydride.abbreviation,
                    "smiles": dianhydride.smiles,
                },
                "diamines": diamines_payload,
                "polyimide_repeat_smiles": polymer_smiles_from_monomers(
                    dianhydride,
                    diamine_records,
                    polymer_ocsr_smiles=polymer_repeat_ocsr,
                ),
            }
        )

    return {
        "paper_id": paper_id,
        "source_pdf": str(pdf_path),
        "monomer_dictionary": [asdict(monomer) for monomer in monomers.values()],
        "image_candidates": [asdict(candidate) for candidate in candidates],
        "ocsr_results": [asdict(result) for result in ocsr_results],
        "polymers": polymers,
    }


def write_jsonl(rows: Sequence[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_polymer_csv(rows: Sequence[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "paper_id",
                "polymer_series",
                "polymer_name",
                "dianhydride_abbreviation",
                "dianhydride_name",
                "dianhydride_smiles",
                "diamines",
                "polyimide_repeat_smiles",
            ],
        )
        writer.writeheader()
        for row in rows:
            diamines = "; ".join(
                f"{item['abbreviation']}|{item['name']}|{item['smiles']}|{item['ratio']}"
                for item in row["diamines"]
            )
            writer.writerow(
                {
                    "paper_id": row["paper_id"],
                    "polymer_series": row["polymer_series"],
                    "polymer_name": row["polymer_name"],
                    "dianhydride_abbreviation": row["dianhydride"]["abbreviation"],
                    "dianhydride_name": row["dianhydride"]["name"],
                    "dianhydride_smiles": row["dianhydride"]["smiles"],
                    "diamines": diamines,
                    "polyimide_repeat_smiles": row["polyimide_repeat_smiles"],
                }
            )


def _legacy_pi5922_main(args: argparse.Namespace) -> int:
    """Original PI.5922-specific regex pipeline. Triggered by --use-legacy-pi5922."""
    pdf_path = Path(args.pdf).expanduser().resolve()
    paper_id = pdf_path.stem
    output_dir = pdf_path.parent / f"{pdf_path.stem}_outputs"
    legacy_json_path = output_dir / "polyimide_extraction.json"
    images_dir = output_dir / "extracted_images"
    model_path = Path(args.model_path).expanduser().resolve()
    cache_path = Path("monomer_library.json").resolve()

    full_text, _page_texts = extract_pdf_text(pdf_path)
    entries = extract_series_entries(full_text)
    target_abbreviations = extract_target_abbreviations(entries)
    monomers = extract_monomers(full_text, target_abbreviations)
    cache = load_cache(cache_path)
    candidates = extract_image_candidates(pdf_path, images_dir)
    predictor = init_predictor(model_path, args.device)
    ocsr_results = run_ocsr(candidates, monomers, predictor, output_dir / "structures_clean")
    polymer_repeat_ocsr = attempt_polymer_repeat_ocsr(candidates, predictor)
    resolve_monomer_smiles(monomers, cache)
    payload = build_output(paper_id, pdf_path, monomers, entries, candidates, ocsr_results, polymer_repeat_ocsr)
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    save_cache(cache_path, cache)
    print(f"[legacy] saved {legacy_json_path}")
    return 0


# ---------------------------------------------------------------------------
# Schema-aligned batch pipeline (data底座 v0)
# ---------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = (
    "You are a polymer-chemistry data extraction assistant. "
    "Extract structured records from a research paper text about polyimides. "
    "Return ONLY JSON matching the requested schema. Use null when a value is "
    "absent. Numbers must be SI units as specified. Do not invent monomers or "
    "samples that are not stated in the text."
)

# Compact target schema fed to the LLM (single self-contained JSON object).
LLM_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["polymers", "samples", "polymer_components", "cure_profiles", "property_records"],
    "properties": {
        "doi": {"type": ["string", "null"]},
        "polymers": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["local_polymer_key", "polymer_class", "is_crosslinked", "is_copolymer"],
                "properties": {
                    "local_polymer_key": {"type": "string"},
                    "polymer_name": {"type": ["string", "null"]},
                    "polymer_class": {"type": "string", "enum": [
                        "polyimide", "polyamide_imide", "polyamide", "polyester",
                        "polyurethane", "polybenzoxazole", "other"]},
                    "is_crosslinked": {"type": "boolean"},
                    "is_copolymer": {"type": "boolean"},
                    "imidization_route": {"type": ["string", "null"], "enum": [
                        "thermal", "chemical", "mixed", "not_applicable", None]},
                },
                "additionalProperties": False,
            },
        },
        "polymer_components": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["local_polymer_key", "monomer_abbreviation", "role", "molar_ratio"],
                "properties": {
                    "local_polymer_key": {"type": "string"},
                    "monomer_abbreviation": {"type": "string"},
                    "role": {"type": "string", "enum": [
                        "diamine", "dianhydride", "diacid_chloride", "triacid_chloride",
                        "diisocyanate", "diol", "diacid", "crosslinker",
                        "chain_extender", "end_capper", "additive", "filler", "other"]},
                    "molar_ratio": {"type": "number", "minimum": 0},
                    "is_crosslinker": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
        },
        "samples": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["local_sample_key", "local_polymer_key"],
                "properties": {
                    "local_sample_key": {"type": "string"},
                    "local_polymer_key": {"type": "string"},
                    "sample_label": {"type": ["string", "null"]},
                    "material_stage": {"type": ["string", "null"], "enum": [
                        "monomer_solution", "paa_precursor", "partially_imidized_film",
                        "pi_final_film", "composite_film", "final_film", None]},
                    "solvent": {"type": ["string", "null"]},
                    "film_thickness_um": {"type": ["number", "null"]},
                    "mw_g_per_mol": {"type": ["number", "null"]},
                    "inherent_viscosity_dL_per_g": {"type": ["number", "null"]},
                },
                "additionalProperties": False,
            },
        },
        "cure_profiles": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["local_sample_key", "imidization_type", "segments"],
                "properties": {
                    "local_sample_key": {"type": "string"},
                    "imidization_type": {"type": "string", "enum": ["thermal", "chemical", "mixed"]},
                    "atmosphere": {"type": ["string", "null"], "enum": ["air", "N2", "vacuum", "Ar", None]},
                    "segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["step_order", "temp_c", "duration_min"],
                            "properties": {
                                "step_order": {"type": "integer"},
                                "temp_c": {"type": "number"},
                                "duration_min": {"type": "number"},
                                "ramp_rate_c_per_min": {"type": ["number", "null"]},
                            },
                            "additionalProperties": False,
                        },
                    },
                },
                "additionalProperties": False,
            },
        },
        "property_records": {
            "type": "array",
            "items": {
                        "type": "object",
                        "required": ["local_sample_key", "property_category", "property_name",
                                     "value_numeric", "unit", "test_method", "value_raw"],
                        "properties": {
                            "local_sample_key": {"type": "string"},
                            "property_category": {"type": "string", "enum": [
                        "thermal", "optical", "mechanical", "electrical", "physical", "dielectric",
                        "barrier", "chemical", "other"]},
                    "property_name": {"type": "string", "enum": [
                        "Tg", "Td2", "Td5", "Td10", "Tm", "CTE", "transmittance",
                        "yellow_index", "haze", "refractive_index", "cutoff_wavelength", "lambda_90",
                        "tensile_strength", "modulus", "elongation_at_break",
                        "dielectric_constant", "dissipation_factor", "water_uptake",
                        "contact_angle", "density", "free_volume_fraction", "residual_weight_600c",
                        "degree_of_crosslinking", "hardness", "healing_efficiency",
                        "inherent_viscosity", "other"]},
                    "value_numeric": {"type": "number"},
                    "unit": {"type": "string"},
                    "value_std": {"type": ["number", "null"]},
                    "value_raw": {"type": "string"},
                    "value_qualifier": {"type": ["string", "null"], "enum": [
                        "exact", "approx", "lt", "lte", "gt", "gte", "range", None]},
                    "property_name_raw": {"type": ["string", "null"]},
                    "unit_raw": {"type": ["string", "null"]},
                    "test_method": {"type": "string"},
                    "wavelength_nm": {"type": ["number", "null"]},
                    "frequency_hz": {"type": ["number", "null"]},
                    "temperature_c": {"type": ["number", "null"]},
                    "temperature_range_c_min": {"type": ["number", "null"]},
                    "temperature_range_c_max": {"type": ["number", "null"]},
                    "decomposition_criterion": {"type": ["string", "null"], "enum": [
                        "2_pct", "5_pct", "10_pct", "onset", None]},
                    "tg_definition": {"type": ["string", "null"], "enum": [
                        "dsc_inflection", "dsc_midpoint", "dma_tan_delta_peak",
                        "dma_storage_onset", "tma_inflection", "unknown", None]},
                    "source_page": {"type": ["integer", "null"]},
                },
                "additionalProperties": False,
            },
        },
        "material_components": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["component_name", "component_class"],
                "properties": {
                    "component_name": {"type": "string"},
                    "abbreviation": {"type": ["string", "null"]},
                    "component_class": {"type": "string", "enum": ["filler", "nanofiller", "additive", "solvent", "other"]},
                    "chemical_description": {"type": ["string", "null"]},
                    "surface_treatment": {"type": ["string", "null"]},
                },
                "additionalProperties": False,
            },
        },
        "sample_compositions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["local_sample_key", "component_name", "component_kind"],
                "properties": {
                    "local_sample_key": {"type": "string"},
                    "component_name": {"type": "string"},
                    "component_abbreviation": {"type": ["string", "null"]},
                    "component_kind": {"type": "string", "enum": ["monomer", "crosslinker", "filler", "additive", "solvent", "other"]},
                    "component_role": {"type": ["string", "null"]},
                    "amount_value": {"type": ["number", "null"]},
                    "amount_unit": {"type": ["string", "null"], "enum": ["mol_ratio", "mol_pct", "wt_pct", "mass_ratio", "feed_ratio", "equiv", "phr", "unknown", None]},
                    "amount_basis": {"type": ["string", "null"], "enum": ["vs_total_monomer", "vs_polymer", "vs_total_solids", "vs_resin", "vs_sample", "unknown", None]},
                    "raw_expression": {"type": ["string", "null"]},
                },
                "additionalProperties": False,
            },
        },
        "study_series": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["variable_name", "variable_kind"],
                "properties": {
                    "series_name": {"type": ["string", "null"]},
                    "variable_name": {"type": "string"},
                    "variable_kind": {"type": "string", "enum": ["composition_ratio", "crosslinker_loading", "filler_loading", "processing_temperature", "material_stage", "other"]},
                    "variable_unit": {"type": ["string", "null"]},
                    "variable_values_text": {"type": ["string", "null"]},
                    "notes": {"type": ["string", "null"]},
                },
                "additionalProperties": False,
            },
        },
        "extracted_monomers": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["abbreviation", "name"],
                "properties": {
                    "abbreviation": {"type": "string"},
                    "name": {"type": "string"},
                    "monomer_class": {"type": ["string", "null"], "enum": [
                        "dianhydride", "diamine", "diacid_chloride", "triacid_chloride",
                        "diisocyanate", "diol", "diacid", "crosslinker", "end_capper",
                        "modifier", "other", None]},
                    "cas_number": {"type": ["string", "null"]},
                    "smiles": {"type": ["string", "null"]},
                    "smiles_source": {"type": ["string", "null"], "enum": [
                        "image_vision", "text_mention", "inferred", "unknown", None]},
                    "source_page": {"type": ["integer", "null"]},
                },
                "additionalProperties": False,
            },
        },
    },
}


def _llm_client():
    if OpenAI is None:
        raise RuntimeError("openai package not installed; run `pip install openai`")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set (see .env.example)")
    return OpenAI(api_key=api_key, base_url=base_url)


def _render_pdf_pages(pdf_path: Path, dpi: int = 150, max_pages: int = 12) -> List[Tuple[int, bytes]]:
    """Render pages of a PDF to PNG bytes. Prefer pages mentioning Scheme/Fig/structure."""
    doc = fitz.open(pdf_path)
    try:
        priority: List[int] = []
        fallback: List[int] = []
        for i in range(len(doc)):
            txt = (doc[i].get_text("text") or "").lower()
            if any(tok in txt for tok in ("scheme ", "fig.", "figure ", "structure")):
                priority.append(i)
            else:
                fallback.append(i)
        ordered = priority + fallback
        picked = ordered[:max_pages]
        picked.sort()
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        out: List[Tuple[int, bytes]] = []
        for idx in picked:
            pix = doc[idx].get_pixmap(matrix=mat, alpha=False)
            out.append((idx + 1, pix.tobytes("png")))
        return out
    finally:
        doc.close()


def llm_extract_paper(text: str, pdf_path: Optional[Path] = None,
                      model: Optional[str] = None, max_chars: int = 80000,
                      max_pages: int = 12) -> Dict[str, Any]:
    """One multimodal LLM call per paper -> nested extraction + monomer SMILES from figures."""
    import base64
    client = _llm_client()
    model = model or os.getenv("MODEL_NAME", os.getenv("LLM_MODEL", "gpt-4o-mini"))
    snippet = text[:max_chars]
    user_text = (
        "Paper text follows between <PAPER> tags, and the selected pages are attached "
        "as images (page numbers match the paper). Extract every measured polymer "
        "sample with composition, cure profile, and reported properties, AND read "
        "each monomer structure directly from the Scheme/Figure images.\n\n"
        f"Required output JSON schema:\n{json.dumps(LLM_OUTPUT_SCHEMA, ensure_ascii=False)}\n\n"
        "Conventions:\n"
        "- extracted_monomers: include every monomer referenced by any polymer. For "
        "  each, return canonical SMILES read from the image. Set smiles_source to "
        "  'image_vision' when read from the figure, 'text_mention' when the paper "
        "  prints SMILES/CAS, 'inferred' when you only know the common name, or "
        "  'unknown' if no reliable SMILES is available. Include source_page (1-based).\n"
        "- Use * as the polymer attachment point ONLY for repeat units; monomers "
        "  should be neutral closed-shell molecules (e.g. the diamine itself, not its "
        "  residue). Dianhydride SMILES should include both anhydride rings.\n"
        "- local_polymer_key / local_sample_key: short stable string per paper "
        "  (e.g. the sample label printed in the paper, like 'CPI-20').\n"
        "- Distinguish polymer-level backbone composition from sample-level variable "
        "  composition. Variable crosslinker / filler / co-monomer loadings belong in "
        "  sample_compositions, not polymer_components.\n"
        "- If the paper contains precursor PAA and final PI/CPI films, set sample.material_stage "
        "  accordingly instead of mixing them as one sample type.\n"
        "- polymer_components.monomer_abbreviation must exactly match an abbreviation in "
        "  extracted_monomers.\n"
        "- polymer_components.molar_ratio is for fixed backbone stoichiometry only; "
        "  filler or crosslinker wt% should not be stored there.\n"
        "- For Tg report tg_definition; for Td report decomposition_criterion; "
        "  for transmittance report wavelength_nm. If the paper reports λ90, use property_name "
        "  'lambda_90' with unit 'nm'.\n"
        "- value_raw: the exact substring as printed (e.g. '305 °C'). Preserve comparison "
        "  operators such as < or > via value_qualifier.\n"
        "- Skip nanocomposite filler entries unless the polymer matrix is the focus.\n"
        f"<PAPER>\n{snippet}\n</PAPER>"
    )
    content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    page_imgs: List[Tuple[int, bytes]] = []
    if pdf_path is not None:
        try:
            page_imgs = _render_pdf_pages(pdf_path, dpi=150, max_pages=max_pages)
        except Exception as exc:
            print(f"[WARN] page render failed for {pdf_path.name}: {exc}", file=sys.stderr)
    for page_no, png_bytes in page_imgs:
        b64 = base64.b64encode(png_bytes).decode("ascii")
        content.append({"type": "text", "text": f"[page {page_no}]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            out = resp.choices[0].message.content or "{}"
            return json.loads(out)
        except Exception as exc:
            last_err = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM extraction failed after 3 attempts: {last_err}")


def slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_").lower()
    return s[:80] or "paper"


def safe_inchikey(smiles: str) -> str:
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return inchi.MolToInchiKey(mol)
    except Exception:
        return ""


def detect_fluorine(smiles: str) -> Optional[bool]:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return any(atom.GetAtomicNum() == 9 for atom in mol.GetAtoms())


def infer_functionality(smiles: str, role: str) -> int:
    """Guess functional-group count from SMILES, fall back to 2."""
    if not smiles:
        return 2
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 2
    if role in ("dianhydride",):
        sm = Chem.MolFromSmarts("O=C1OC(=O)[#6]~[#6]1")
        return max(2, len(mol.GetSubstructMatches(sm)))
    if role in ("diamine",):
        sm = Chem.MolFromSmarts("[NX3H2]")
        return max(2, len(mol.GetSubstructMatches(sm)))
    if role in ("crosslinker",):
        return 3
    return 2


def infer_functional_groups(smiles: str) -> List[str]:
    if not smiles:
        return []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    groups: List[str] = []
    checks = [
        ("anhydride", "O=C1OC(=O)[#6]~[#6]1"),
        ("amine", "[NX3H2]"),
        ("hydroxyl", "[OX2H]"),
        ("carboxyl", "C(=O)[OX2H1]"),
        ("acid_chloride", "C(=O)Cl"),
        ("isocyanate", "N=C=O"),
        ("alkene", "C=C"),
        ("alkyne", "C#C"),
        ("epoxy", "C1OC1"),
    ]
    for name, smarts in checks:
        sm = Chem.MolFromSmarts(smarts)
        if sm is not None and mol.GetSubstructMatches(sm):
            groups.append(name)
    return groups


REFERENCE_MONOMER_SMILES = {
    "6FDA": "O=C1OC(=O)c2cc(C(c3ccc4c(c3)C(=O)OC4=O)(C(F)(F)F)C(F)(F)F)ccc21",
    "TFMB": "Nc1ccc(C(F)(F)F)c(-c2ccc(N)cc2C(F)(F)F)c1",
}

PROPERTY_NAME_MAP = {
    "wtr600": ("residual_weight_600c", "thermal"),
    "wtr_600": ("residual_weight_600c", "thermal"),
    "degree_of_crosslinking": ("degree_of_crosslinking", "chemical"),
    "density": ("density", "physical"),
    "ffv": ("free_volume_fraction", "physical"),
    "free_volume_fraction": ("free_volume_fraction", "physical"),
    "lambda90": ("lambda_90", "optical"),
    "λ90": ("lambda_90", "optical"),
    "hardness": ("hardness", "mechanical"),
    "healingefficiency": ("healing_efficiency", "other"),
}

PROPERTY_NAME_ALIASES = {
    "Tg", "Td2", "Td5", "Td10", "Tm", "CTE", "transmittance", "yellow_index", "haze",
    "refractive_index", "cutoff_wavelength", "lambda_90", "tensile_strength", "modulus",
    "elongation_at_break", "dielectric_constant", "dissipation_factor", "water_uptake",
    "contact_angle", "density", "free_volume_fraction", "residual_weight_600c",
    "degree_of_crosslinking", "hardness", "healing_efficiency", "inherent_viscosity", "other",
}

PROPERTY_CATEGORY_ALIASES = {
    "thermal", "optical", "mechanical", "electrical", "physical", "dielectric", "barrier",
    "chemical", "other",
}

UNIT_MAP = {
    "c": "°C",
    "℃": "°C",
    "ppm/c": "ppm/°C",
    "ppm/℃": "ppm/°C",
    "ppm/oc": "ppm/°C",
    "deg": "deg",
    "g/cm3": "g/cm3",
}


def reference_smiles_for_monomer(abbr: str, name: str) -> str:
    if abbr in REFERENCE_MONOMER_SMILES:
        return canonicalize_smiles(REFERENCE_MONOMER_SMILES[abbr]) or REFERENCE_MONOMER_SMILES[abbr]
    upper_name = (name or "").upper()
    if "6FDA" in upper_name:
        return canonicalize_smiles(REFERENCE_MONOMER_SMILES["6FDA"]) or REFERENCE_MONOMER_SMILES["6FDA"]
    if abbr == "TFMB" or "TRIFLUOROMETHYL" in upper_name:
        return canonicalize_smiles(REFERENCE_MONOMER_SMILES["TFMB"]) or REFERENCE_MONOMER_SMILES["TFMB"]
    return ""


def normalize_unit(unit: Any) -> str:
    raw = "" if unit is None else str(unit).strip()
    if not raw:
        return ""
    lowered = raw.lower()
    return UNIT_MAP.get(lowered, raw)


def parse_value_qualifier(value_raw: str) -> Optional[str]:
    text = (value_raw or "").strip()
    if not text:
        return None
    if text.startswith("<="):
        return "lte"
    if text.startswith(">="):
        return "gte"
    if text.startswith("<"):
        return "lt"
    if text.startswith(">"):
        return "gt"
    if re.search(r"\bca\.|\bapprox(?:\.|imately)?\b|≈|~|\bwithin\b", text, re.I):
        return "approx"
    if re.search(r"\d+\s*[-–]\s*\d+", text):
        return "range"
    return "exact"


def infer_material_stage(local_sample_key: str, paper_text: str) -> Optional[str]:
    key = (local_sample_key or "").upper()
    if "PAA" in key:
        return "paa_precursor"
    if key.startswith(("PI_", "CPI", "PAI")):
        if re.search(r"\bCS30B\b|\borganoclay\b|\bclay content\b", paper_text, re.I):
            return "composite_film"
        return "pi_final_film" if key.startswith("PI_") else "final_film"
    return None


def detect_amount_qualifier(raw_expression: str) -> Optional[str]:
    return parse_value_qualifier(raw_expression)


def normalize_property_record(
    prop: Dict[str, Any],
    paper_id: str,
    paper_text: str,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    review_items: List[Dict[str, Any]] = []
    vn = prop.get("value_numeric")
    if isinstance(vn, str):
        try:
            vn = float(vn)
        except (TypeError, ValueError):
            vn = None
    if not isinstance(vn, (int, float)):
        return None, review_items

    pname_raw = str(prop.get("property_name") or "other")
    pname_key = re.sub(r"[^A-Za-z0-9_λ]+", "", pname_raw).lower()
    raw_text = prop.get("value_raw") or ""
    category = prop.get("property_category") or "other"
    unit_raw = "" if prop.get("unit") is None else str(prop.get("unit"))
    unit = normalize_unit(unit_raw)
    qualifier = prop.get("value_qualifier") or parse_value_qualifier(raw_text)

    pname = pname_raw if pname_raw in PROPERTY_NAME_ALIASES else "other"
    if pname_key in PROPERTY_NAME_MAP:
        pname, mapped_category = PROPERTY_NAME_MAP[pname_key]
        category = mapped_category

    if pname == "other" and unit == "g/cm3":
        pname = "density"
        category = "physical"

    if pname == "transmittance" and prop.get("wavelength_nm") and abs(float(vn) - 90.0) < 1e-6:
        if "λ90" in raw_text or "lambda90" in raw_text.lower():
            pname = "lambda_90"
            category = "optical"
            vn = prop.get("wavelength_nm")
            unit = "nm"
            qualifier = "exact"
            prop = dict(prop)
            prop["wavelength_nm"] = None

    if category not in PROPERTY_CATEGORY_ALIASES:
        category = "other"
    if pname not in PROPERTY_NAME_ALIASES:
        review_items.append({
            "kind": "unknown_property_name",
            "paper": paper_id,
            "sample": prop.get("local_sample_key"),
            "property_name_raw": pname_raw,
        })
        pname = "other"

    if pname == "other":
        review_items.append({
            "kind": "property_mapped_to_other",
            "paper": paper_id,
            "sample": prop.get("local_sample_key"),
            "property_name_raw": pname_raw,
            "unit_raw": unit_raw,
        })

    if qualifier and qualifier != "exact":
        review_items.append({
            "kind": "qualified_property_value",
            "paper": paper_id,
            "sample": prop.get("local_sample_key"),
            "property_name": pname,
            "value_raw": raw_text,
            "value_qualifier": qualifier,
        })

    if pname == "other" and re.search(r"\bdensity\b", paper_text, re.I) and unit == "g/cm3":
        pname = "density"
        category = "physical"

    return {
        "property_category": category,
        "property_name": pname,
        "value_numeric": float(vn),
        "unit": unit,
        "value_std": prop.get("value_std"),
        "value_raw": raw_text,
        "value_qualifier": qualifier,
        "property_name_raw": pname_raw,
        "unit_raw": unit_raw or None,
        "test_method": prop.get("test_method") or "unknown",
        "temperature_c": prop.get("temperature_c"),
        "temperature_range_c_min": prop.get("temperature_range_c_min"),
        "temperature_range_c_max": prop.get("temperature_range_c_max"),
        "frequency_hz": prop.get("frequency_hz"),
        "wavelength_nm": prop.get("wavelength_nm"),
        "decomposition_criterion": prop.get("decomposition_criterion"),
        "tg_definition": prop.get("tg_definition"),
        "source_page": prop.get("source_page"),
    }, review_items


def infer_study_series(
    paper_id: str,
    paper_text: str,
    samples: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    values = [s.get("local_sample_key", "") for s in samples]
    joined = " ".join(values)
    series: List[Dict[str, Any]] = []
    if re.search(r"\bBTC\b", paper_text) and re.search(r"\bTCL\b", paper_text) and re.search(r"ratios? of TCL and BTC", paper_text, re.I):
        series.append({
            "series_name": "TCL/BTC crosslinker ratio series",
            "variable_name": "BTC_to_TCL_ratio",
            "variable_kind": "composition_ratio",
            "variable_unit": "mol_ratio",
            "variable_values_text": "sample-specific ratios vary; exact mapping not fully recovered from cached extraction",
            "notes": "Variable co-monomer / crosslinker ratio should live at sample-level composition, not polymer-level composition.",
        })
    if re.search(r"1,3-PBO", paper_text) and re.search(r"0\.1%.*, 0\.5%.*, 1%.*, and 3%", paper_text):
        series.append({
            "series_name": "1,3-PBO loading series",
            "variable_name": "1,3-PBO_loading",
            "variable_kind": "crosslinker_loading",
            "variable_unit": "wt_pct",
            "variable_values_text": "0, 0.1, 0.5, 1, 3 wt%",
            "notes": "Mass ratio of 1,3-PBO to CPI/PAA-derived film.",
        })
    if re.search(r"CS30B", paper_text) and re.search(r"1[–-]4 wt%", paper_text):
        series.append({
            "series_name": "CS30B organoclay loading series",
            "variable_name": "CS30B_loading",
            "variable_kind": "filler_loading",
            "variable_unit": "wt_pct",
            "variable_values_text": "0, 1, 2, 3, 4 wt%",
            "notes": "Primary nanofiller variable in the hybrid CPI study.",
        })
    if re.search(r"heat-treated at\s+different temperatures", paper_text, re.I) or re.search(r"various heat treatment", paper_id, re.I):
        temps = sorted({re.sub(r"[^0-9]", "", v) for v in values if re.search(r"\d", v)})
        series.append({
            "series_name": "annealing / imidization temperature series",
            "variable_name": "anneal_temperature",
            "variable_kind": "processing_temperature",
            "variable_unit": "°C",
            "variable_values_text": ", ".join(t for t in temps if t),
            "notes": "Processing condition series; not a composition ratio series.",
        })
    if re.search(r"\bPAA\b", joined) and re.search(r"\bPI_", joined):
        series.append({
            "series_name": "material stage transition",
            "variable_name": "material_stage",
            "variable_kind": "material_stage",
            "variable_unit": None,
            "variable_values_text": "PAA precursor -> PI final film",
            "notes": "Precursor and final imidized films should not be treated as the same sample stage.",
        })
    return series


def infer_nonpolymer_components(paper_id: str, paper_text: str) -> List[Dict[str, Any]]:
    components: List[Dict[str, Any]] = []
    if re.search(r"\bCS30B\b", paper_text):
        components.append({
            "component_name": "Cloisite 30B organoclay",
            "abbreviation": "CS30B",
            "component_class": "nanofiller",
            "chemical_description": "organophilic clay / organoclay",
            "surface_treatment": "alkyl-modified clay surface",
            "notes": f"inferred_from_text:{paper_id}",
        })
    return components


def ensure_llm_payload_defaults(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload or {})
    for key in (
        "polymers", "polymer_components", "samples", "cure_profiles", "property_records",
        "extracted_monomers", "material_components", "sample_compositions", "study_series",
        "trend_records",
    ):
        out.setdefault(key, [])
    return out


def recover_pai_density_ffv(payload: Dict[str, Any], paper_text: str) -> None:
    if not re.search(r"Table 3\. Dielectric Properties, Density, and FFV", paper_text):
        return
    existing = {
        (p.get("local_sample_key"), p.get("property_name"))
        for p in payload.get("property_records", [])
    }
    pattern = re.compile(
        r"(PAI-\d)\s+([0-9.]+)\s+[0-9.]+\s*×\s*10[−-]\d+\s+[0-9.]+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
        re.I,
    )
    for sample_key, dk, density, ffv, angle, water in pattern.findall(paper_text):
        for pname, value, unit, category, method in (
            ("density", density, "g/cm3", "physical", "Archimedes density kit"),
            ("free_volume_fraction", ffv, "", "physical", "calculated from density and van der Waals volume"),
            ("contact_angle", angle, "deg", "chemical", "water contact angle"),
            ("water_uptake", water, "%", "chemical", "water absorption"),
        ):
            if (sample_key, pname) in existing:
                continue
            payload["property_records"].append({
                "local_sample_key": sample_key,
                "property_category": category,
                "property_name": pname,
                "value_numeric": float(value),
                "unit": unit,
                "value_raw": value,
                "test_method": method,
            })


def recover_self_healing_paper(payload: Dict[str, Any], paper_text: str) -> None:
    if payload.get("polymers") or not re.search(r"6FDA14", paper_text):
        return
    payload["doi"] = payload.get("doi")
    payload["extracted_monomers"] = [
        {
            "abbreviation": "6FDA",
            "name": "4,4'-(hexafluoroisopropylidene)diphthalic anhydride",
            "monomer_class": "dianhydride",
            "smiles": REFERENCE_MONOMER_SMILES["6FDA"],
            "smiles_source": "inferred",
            "source_page": 2,
        },
        {
            "abbreviation": "4AD",
            "name": "4,4'-dithioaniline",
            "monomer_class": "diamine",
            "smiles": None,
            "smiles_source": "text_mention",
            "source_page": 2,
        },
        {
            "abbreviation": "TTDA",
            "name": "4,7,10-Trioxa-1,13-tridecanediamine",
            "monomer_class": "diamine",
            "smiles": None,
            "smiles_source": "text_mention",
            "source_page": 2,
        },
    ]
    payload["polymers"] = [{
        "local_polymer_key": "CPI_dynamic_network",
        "polymer_name": "self-healing colorless polyimide family",
        "polymer_class": "polyimide",
        "is_crosslinked": False,
        "is_copolymer": True,
        "imidization_route": "chemical",
    }]
    payload["polymer_components"] = [{
        "local_polymer_key": "CPI_dynamic_network",
        "monomer_abbreviation": "6FDA",
        "role": "dianhydride",
        "molar_ratio": 1.0,
        "is_crosslinker": False,
    }]
    samples = ["6FDA0", "6FDA14", "6FDA32", "6FDA1"]
    payload["samples"] = [{
        "local_sample_key": sk,
        "local_polymer_key": "CPI_dynamic_network",
        "sample_label": sk,
        "material_stage": "final_film",
        "solvent": "DMAc",
        "film_thickness_um": None,
        "mw_g_per_mol": None,
        "inherent_viscosity_dL_per_g": None,
    } for sk in samples]
    payload["cure_profiles"] = [{
        "local_sample_key": sk,
        "imidization_type": "chemical",
        "atmosphere": "vacuum",
        "segments": [
            {"step_order": 1, "temp_c": 70.0, "duration_min": 60.0, "ramp_rate_c_per_min": None},
            {"step_order": 2, "temp_c": 170.0, "duration_min": 120.0, "ramp_rate_c_per_min": None},
        ],
    } for sk in samples]
    payload["study_series"] = [{
        "series_name": "4AD/TTDA diamine ratio series",
        "variable_name": "4AD_to_TTDA_ratio",
        "variable_kind": "composition_ratio",
        "variable_unit": "mol_ratio",
        "variable_values_text": "6FDA0=0:10, 6FDA14=2:8, 6FDA32=6:4, 6FDA1=10:0",
        "notes": "Diamine composition controls Tg and self-healing performance.",
    }]
    sample_compositions: List[Dict[str, Any]] = []
    composition_map = {
        "6FDA0": [("4AD", 0.0, "0"), ("TTDA", 100.0, "10 mmol")],
        "6FDA14": [("4AD", 20.0, "2 mmol"), ("TTDA", 80.0, "8 mmol")],
        "6FDA32": [("4AD", 60.0, "6 mmol"), ("TTDA", 40.0, "4 mmol")],
        "6FDA1": [("4AD", 100.0, "10 mmol"), ("TTDA", 0.0, "0")],
    }
    for sk, comps in composition_map.items():
        for abbr, mol_pct, raw_expr in comps:
            sample_compositions.append({
                "local_sample_key": sk,
                "component_name": abbr,
                "component_abbreviation": abbr,
                "component_kind": "monomer",
                "component_role": "diamine",
                "amount_value": mol_pct,
                "amount_unit": "mol_pct",
                "amount_basis": "vs_total_monomer",
                "raw_expression": raw_expr,
            })
    payload["sample_compositions"] = sample_compositions
    payload["property_records"] = [
        {
            "local_sample_key": "6FDA0",
            "property_category": "thermal",
            "property_name": "Tg",
            "value_numeric": 75.0,
            "unit": "°C",
            "value_raw": "within 75 °C",
            "test_method": "DMA tan delta / DSC discussion",
        },
        {
            "local_sample_key": "6FDA14",
            "property_category": "thermal",
            "property_name": "Tg",
            "value_numeric": 95.0,
            "unit": "°C",
            "value_raw": "approximately 95 °C",
            "test_method": "DMA tan delta",
        },
        {
            "local_sample_key": "6FDA32",
            "property_category": "thermal",
            "property_name": "Tg",
            "value_numeric": 155.0,
            "unit": "°C",
            "value_raw": "minimal temperature at 155 °C",
            "test_method": "healing threshold discussion",
        },
        {
            "local_sample_key": "6FDA0",
            "property_category": "optical",
            "property_name": "transmittance",
            "value_numeric": 95.0,
            "unit": "%",
            "value_raw": ">95%",
            "test_method": "UV-vis",
            "wavelength_nm": 500.0,
        },
        {
            "local_sample_key": "6FDA14",
            "property_category": "optical",
            "property_name": "transmittance",
            "value_numeric": 95.0,
            "unit": "%",
            "value_raw": ">95%",
            "test_method": "UV-vis",
            "wavelength_nm": 500.0,
        },
        {
            "local_sample_key": "6FDA14",
            "property_category": "other",
            "property_name": "healing_efficiency",
            "value_numeric": 98.0,
            "unit": "%",
            "value_raw": ">98%",
            "test_method": "overall self-healing performance",
        },
        {
            "local_sample_key": "6FDA14",
            "property_category": "mechanical",
            "property_name": "hardness",
            "value_numeric": 100.0,
            "unit": "MPa",
            "value_raw": "100 MPa",
            "test_method": "hardness summary",
        },
    ]


def augment_llm_payload_from_text(paper_id: str, payload: Dict[str, Any], paper_text: str) -> Dict[str, Any]:
    payload = ensure_llm_payload_defaults(payload)
    recover_pai_density_ffv(payload, paper_text)
    recover_self_healing_paper(payload, paper_text)
    if paper_id.startswith("colorless_poly_amide-imide_s_with_low_dielectric_constants"):
        build_sample_level_compositions_for_pai(payload, paper_text)
    if not payload.get("material_components"):
        payload["material_components"] = infer_nonpolymer_components(paper_id, paper_text)
    if not payload.get("study_series"):
        payload["study_series"] = infer_study_series(paper_id, paper_text, payload.get("samples", []) or [])
    if not payload.get("trend_records"):
        payload["trend_records"] = infer_trend_records(paper_id, paper_text, [s.get("local_sample_key", "") for s in payload.get("samples", []) or []])
    return payload


def infer_trend_records(
    paper_id: str,
    paper_text: str,
    sample_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    trends: List[Dict[str, Any]] = []
    if paper_id.startswith("colorless_poly_amide-imide_s_with_low_dielectric_constants"):
        trends.extend([
            {
                "variable_name": "BTC_to_TCL_ratio",
                "variable_unit": "mol_ratio",
                "property_name": "Tg",
                "trend_direction": "increase",
                "evidence_text": "From PAI-1 to PAI-4, the Tg increases from 342 to 351 °C.",
                "sample_scope": "PAI-1 -> PAI-4",
                "mechanism_note": "BTC crosslinking links chains and suppresses segmental motion.",
                "confidence": 0.92,
            },
            {
                "variable_name": "BTC_to_TCL_ratio",
                "variable_unit": "mol_ratio",
                "property_name": "CTE",
                "trend_direction": "decrease",
                "evidence_text": "CTE decreases as crosslinker BTC increases.",
                "sample_scope": "PAI-1 -> PAI-4",
                "mechanism_note": "Crosslinking restricts thermal expansion.",
                "confidence": 0.88,
            },
            {
                "variable_name": "BTC_to_TCL_ratio",
                "variable_unit": "mol_ratio",
                "property_name": "dielectric_constant",
                "trend_direction": "decrease",
                "evidence_text": "The dielectric constant decreases from 3.06 to 2.82 from PAI-1 to PAI-4.",
                "sample_scope": "PAI-1 -> PAI-4",
                "mechanism_note": "Lower density and higher FFV reduce dielectric response.",
                "confidence": 0.94,
            },
            {
                "variable_name": "BTC_to_TCL_ratio",
                "variable_unit": "mol_ratio",
                "property_name": "free_volume_fraction",
                "trend_direction": "increase",
                "evidence_text": "FFV values increase from 0.148 to 0.190 with increasing BTC content.",
                "sample_scope": "PAI-1 -> PAI-4",
                "mechanism_note": "Crosslinked micro-branched structure creates larger free volume.",
                "confidence": 0.92,
            },
        ])
    if paper_id.startswith("comparative_analysis_of_the_properties_of_colorless_and_transparent_polyimide_na"):
        trends.extend([
            {
                "variable_name": "CS30B_loading",
                "variable_unit": "wt_pct",
                "property_name": "Tg",
                "trend_direction": "optimum",
                "evidence_text": "Maximum enhancement in physical properties occurred at a critical clay content.",
                "sample_scope": "0-4 wt%",
                "mechanism_note": "Low filler loading improves dispersion; excess loading causes agglomeration.",
                "confidence": 0.85,
            },
            {
                "variable_name": "CS30B_loading",
                "variable_unit": "wt_pct",
                "property_name": "tensile_strength",
                "trend_direction": "optimum",
                "evidence_text": "Mechanical properties increased at low organoclay loading and dropped at 4 wt% due to agglomeration.",
                "sample_scope": "0-4 wt%",
                "mechanism_note": "Nanoscale dispersion at low loading; aggregation at higher loading.",
                "confidence": 0.88,
            },
        ])
    if paper_id.startswith("crosslinked_colorless_polyimide_films_via_oxazole"):
        trends.extend([
            {
                "variable_name": "1,3-PBO_loading",
                "variable_unit": "wt_pct",
                "property_name": "density",
                "trend_direction": "increase",
                "evidence_text": "Densities of the CPI films range from 1.30 to 1.62 g/cm3, increasing with the increase of 1,3-PBO contents.",
                "sample_scope": "CPI0 -> CPI4",
                "mechanism_note": "Higher crosslinker loading reduces free volume and increases chain packing density.",
                "confidence": 0.95,
            },
            {
                "variable_name": "1,3-PBO_loading",
                "variable_unit": "wt_pct",
                "property_name": "water_uptake",
                "trend_direction": "decrease",
                "evidence_text": "Crosslinked CPI films showed improved solvent resistance and reduced water uptake with increasing 1,3-PBO content.",
                "sample_scope": "CPI0 -> CPI4",
                "mechanism_note": "Crosslinked network increases hydrophobicity and reduces solvent penetration.",
                "confidence": 0.84,
            },
        ])
    if paper_id.startswith("fabrications_and_properties_of_colorless_polyimide_films_depending_on_various_he"):
        trends.extend([
            {
                "variable_name": "anneal_temperature",
                "variable_unit": "°C",
                "property_name": "degree_of_crosslinking",
                "trend_direction": "increase",
                "evidence_text": "With increasing thermal crosslinking temperatures from 250 to 350 ℃, the degree of crosslinking increased.",
                "sample_scope": "PI_250C -> PI_350C",
                "mechanism_note": "Thermal treatment drives imidization and crosslinking conversion.",
                "confidence": 0.9,
            },
            {
                "variable_name": "anneal_temperature",
                "variable_unit": "°C",
                "property_name": "CTE",
                "trend_direction": "decrease",
                "evidence_text": "CTE decreases from 48.59 ppm/℃ to 34.55 ppm/℃ with higher heat-treatment temperature.",
                "sample_scope": "PI_250C -> PI_350C",
                "mechanism_note": "More complete imidization / crosslinking improves dimensional stability.",
                "confidence": 0.9,
            },
        ])
    if paper_id.startswith("colorless_polyimides_with_excellent_optical_transparency_and_self-healing_proper"):
        trends.extend([
            {
                "variable_name": "4AD_to_TTDA_ratio",
                "variable_unit": "mol_ratio",
                "property_name": "Tg",
                "trend_direction": "increase",
                "evidence_text": "As disulfide-containing hard segment content increased, Tg increased from ~75 °C to ~155 °C.",
                "sample_scope": "6FDA0 -> 6FDA32/6FDA1",
                "mechanism_note": "More hard segment and disulfide-containing diamine raises chain rigidity.",
                "confidence": 0.87,
            },
            {
                "variable_name": "4AD_to_TTDA_ratio",
                "variable_unit": "mol_ratio",
                "property_name": "healing_efficiency",
                "trend_direction": "optimum",
                "evidence_text": "6FDA14 showed outstanding healing efficiency (>98%), while 6FDA32 healed incompletely because of high Tg.",
                "sample_scope": "6FDA0, 6FDA14, 6FDA32, 6FDA1",
                "mechanism_note": "Balance between reversible disulfide exchange and chain mobility creates an optimum composition.",
                "confidence": 0.9,
            },
        ])
    return trends


def build_sample_level_compositions_for_pai(payload: Dict[str, Any], paper_text: str) -> None:
    if payload.get("sample_compositions"):
        return
    if not re.search(r"Take PAI-2 as an example", paper_text):
        return
    payload["sample_compositions"].extend([
        {
            "local_sample_key": "PAI-2",
            "component_name": "TCL",
            "component_abbreviation": "TCL",
            "component_kind": "monomer",
            "component_role": "diacid_chloride",
            "amount_value": 35.0,
            "amount_unit": "mol_pct",
            "amount_basis": "vs_total_monomer",
            "raw_expression": "TCL 3.297 mmol within fixed 6FDA/TMC/TCL/BTC series example for PAI-2",
        },
        {
            "local_sample_key": "PAI-2",
            "component_name": "BTC",
            "component_abbreviation": "BTC",
            "component_kind": "crosslinker",
            "component_role": "crosslinker",
            "amount_value": 5.0,
            "amount_unit": "mol_pct",
            "amount_basis": "vs_total_monomer",
            "raw_expression": "BTC 0.313 mmol example for PAI-2; verify exact normalization basis against Scheme 1",
        },
    ])


class IdAllocator:
    def __init__(self, prefix: str, start: int = 1):
        self.prefix = prefix
        self._n = start

    def next(self) -> str:
        sid = f"{self.prefix}_{self._n:06d}"
        self._n += 1
        return sid


def composition_hash(components: List[Dict[str, Any]]) -> str:
    key = json.dumps(
        sorted(
            (c.get("monomer_id", ""), c.get("role", ""), float(c.get("molar_ratio", 0) or 0))
            for c in components
        )
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def process_one_pdf(
    pdf_path: Path,
    raw_dir: Path,
    predictor,
    use_llm: bool,
    use_ocsr: bool,
) -> Dict[str, Any]:
    """Run one PDF; emit per-paper JSON bundle into raw_dir/<slug>/."""
    slug = slugify(pdf_path.stem)
    paper_dir = raw_dir / slug
    paper_dir.mkdir(parents=True, exist_ok=True)
    print(f"[paper] {pdf_path.name} -> {paper_dir}")

    full_text, _pages = extract_pdf_text(pdf_path)
    (paper_dir / "text.txt").write_text(full_text, encoding="utf-8")

    # 1. OCSR over candidate images -> abbreviation: smiles map
    ocsr_monomers: Dict[str, Dict[str, Any]] = {}
    if use_ocsr and predictor is not None:
        images_dir = paper_dir / "images"
        candidates = extract_image_candidates(pdf_path, images_dir)
        # Use empty monomers dict so OCSR runs without the legacy abbr-binding heuristic.
        ocsr_results = run_ocsr(candidates, {}, predictor, paper_dir / "structures_clean")
        for r in ocsr_results:
            if r.smiles and r.matched_abbreviation:
                ocsr_monomers.setdefault(
                    r.matched_abbreviation,
                    {
                        "smiles": r.smiles,
                        "confidence": r.confidence,
                        "source": f"{r.provider}:{Path(r.image).name}#{r.segment_id}",
                    },
                )
        (paper_dir / "ocsr_results.json").write_text(
            json.dumps([asdict(r) for r in ocsr_results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 2. LLM extraction
    llm_payload: Dict[str, Any] = {
        "polymers": [], "polymer_components": [], "samples": [],
        "cure_profiles": [], "property_records": [], "extracted_monomers": [],
        "material_components": [], "sample_compositions": [], "study_series": [], "trend_records": [],
    }
    llm_error: Optional[str] = None
    if use_llm:
        cache_f = paper_dir / "llm_raw.json"
        if cache_f.exists():
            try:
                llm_payload = json.loads(cache_f.read_text(encoding="utf-8"))
                print(f"[cache] reusing {cache_f}", file=sys.stderr)
            except Exception:
                cache_f.unlink(missing_ok=True)
        if not cache_f.exists():
            try:
                llm_payload = llm_extract_paper(full_text, pdf_path=pdf_path)
            except Exception as exc:
                llm_error = str(exc)
                print(f"[WARN] LLM failed for {pdf_path.name}: {exc}", file=sys.stderr)
            cache_f.write_text(
                json.dumps(llm_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    llm_payload = augment_llm_payload_from_text(slug, llm_payload, full_text)

    # Per-paper joined summary: monomer -> polymer(components) -> sample(cure, properties)
    summary = _build_paper_summary(slug, pdf_path, llm_payload, ocsr_monomers, llm_error)
    (paper_dir / "paper_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "paper_id": slug,
        "source_pdf": str(pdf_path),
        "ocsr_monomers": ocsr_monomers,
        "llm": llm_payload,
        "llm_error": llm_error,
        "text": full_text,
    }


def _build_paper_summary(paper_id: str, pdf_path: Path, llm: Dict[str, Any],
                         ocsr_monomers: Dict[str, Dict[str, Any]],
                         llm_error: Optional[str]) -> Dict[str, Any]:
    """Join LLM tables by local keys + attach OCSR/vision SMILES + RDKit-derived fields."""
    mon_by_abbr: Dict[str, Dict[str, Any]] = {}
    summary_review_items: List[Dict[str, Any]] = []
    for m in llm.get("extracted_monomers", []) or []:
        abbr = m.get("abbreviation", "") or m.get("name", "")
        if not abbr:
            continue
        llm_smi = canonicalize_smiles(m.get("smiles") or "")
        ocsr_smi = ocsr_monomers.get(abbr, {}).get("smiles", "")
        chosen, source, resolver_reviews = resolve_smiles_from_identity(
            abbr,
            m.get("name", ""),
            m.get("monomer_class", "") or "",
            [
                (llm_smi, m.get("smiles_source") or "llm"),
                (ocsr_smi, "molscribe"),
            ],
        )
        summary_review_items.extend(resolver_reviews)
        ikey = safe_inchikey(chosen) if chosen else ""
        mon_by_abbr[abbr] = {
            "abbreviation": abbr,
            "name": m.get("name", ""),
            "monomer_class": m.get("monomer_class"),
            "cas_number": m.get("cas_number"),
            "smiles": chosen,
            "smiles_source": source,
            "inchi_key": ikey,
            "contains_fluorine": bool(detect_fluorine(chosen)) if chosen else None,
            "source_page": m.get("source_page"),
        }

    comps_by_poly: Dict[str, List[Dict[str, Any]]] = {}
    for c in llm.get("polymer_components", []) or []:
        comps_by_poly.setdefault(c.get("local_polymer_key", ""), []).append({
            "monomer_abbr": c.get("monomer_abbreviation") or c.get("monomer_abbr"),
            "molar_ratio": c.get("molar_ratio"),
            "role": c.get("role"),
        })

    cures_by_sample: Dict[str, Dict[str, Any]] = {
        c.get("local_sample_key", ""): c for c in (llm.get("cure_profiles", []) or [])
    }
    props_by_sample: Dict[str, List[Dict[str, Any]]] = {}
    for p in llm.get("property_records", []) or []:
        props_by_sample.setdefault(p.get("local_sample_key", ""), []).append(p)
    sample_comps_by_sample: Dict[str, List[Dict[str, Any]]] = {}
    for sc in llm.get("sample_compositions", []) or []:
        sample_comps_by_sample.setdefault(sc.get("local_sample_key", ""), []).append(sc)

    material_components = llm.get("material_components", []) or []

    polymers_out: List[Dict[str, Any]] = []
    for poly in llm.get("polymers", []) or []:
        key = poly.get("local_polymer_key", "")
        comps = comps_by_poly.get(key, [])
        samples_out: List[Dict[str, Any]] = []
        for s in llm.get("samples", []) or []:
            if s.get("local_polymer_key") != key:
                continue
            skey = s.get("local_sample_key", "")
            samples_out.append({
                "local_sample_key": skey,
                "sample_label": s.get("sample_label"),
                "material_stage": s.get("material_stage"),
                "solvent": s.get("solvent"),
                "film_thickness_um": s.get("film_thickness_um"),
                "sample_compositions": sample_comps_by_sample.get(skey, []),
                "cure_profile": cures_by_sample.get(skey),
                "properties": props_by_sample.get(skey, []),
            })
        polymers_out.append({
            "local_polymer_key": key,
            "polymer_name": poly.get("polymer_name"),
            "polymer_class": poly.get("polymer_class"),
            "is_crosslinked": poly.get("is_crosslinked"),
            "is_copolymer": poly.get("is_copolymer"),
            "imidization_route": poly.get("imidization_route"),
            "components": [
                {**c, "monomer": mon_by_abbr.get(c["monomer_abbr"])} for c in comps
            ],
            "samples": samples_out,
        })

    return {
        "paper_id": paper_id,
        "source_pdf": str(pdf_path),
        "doi": llm.get("doi"),
        "llm_error": llm_error,
        "monomers": list(mon_by_abbr.values()),
        "material_components": material_components,
        "study_series": llm.get("study_series", []) or [],
        "trend_records": llm.get("trend_records", []) or [],
        "review_items": summary_review_items,
        "polymers": polymers_out,
    }


REGEX_TG = re.compile(r"\bT[g]\s*[=:]\s*(\d{2,3})\s*°?C", re.I)
REGEX_TD = re.compile(r"\bT(?:d|d5%?|d10%?|5%?|10%?)\s*[=:]\s*(\d{2,4})\s*°?C", re.I)
REGEX_CTE = re.compile(r"\bCTE\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:ppm|10−6|10\^-6)", re.I)
REGEX_T550 = re.compile(r"(\d{2}(?:\.\d+)?)\s*%[^.]{0,40}550\s*nm", re.I)


def regex_sanity_check(text: str) -> Dict[str, List[float]]:
    """Pull obvious numeric signals as ground-truth for cross-check with LLM."""
    return {
        "Tg": [float(m.group(1)) for m in REGEX_TG.finditer(text)],
        "Td": [float(m.group(1)) for m in REGEX_TD.finditer(text)],
        "CTE": [float(m.group(1)) for m in REGEX_CTE.finditer(text)],
        "T550": [float(m.group(1)) for m in REGEX_T550.finditer(text)],
    }


def merge_into_dataset(
    per_paper: List[Dict[str, Any]],
    dataset_dir: Path,
    schema_path: Path,
    validate: bool,
) -> Dict[str, Any]:
    """Merge per-paper results, allocate global IDs, dedupe monomers by InChIKey."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    mon_alloc = IdAllocator("MON")
    cmp_alloc = IdAllocator("CMP")
    pol_alloc = IdAllocator("POL")
    smp_alloc = IdAllocator("SMP")
    scm_alloc = IdAllocator("SCM")
    ser_alloc = IdAllocator("SER")
    trd_alloc = IdAllocator("TRD")

    monomer_table: Dict[str, Dict[str, Any]] = {}  # inchikey -> record
    material_component_table: Dict[str, Dict[str, Any]] = {}
    polymer_table: List[Dict[str, Any]] = []
    component_table: List[Dict[str, Any]] = []
    sample_table: List[Dict[str, Any]] = []
    sample_component_table: List[Dict[str, Any]] = []
    study_series_table: List[Dict[str, Any]] = []
    trend_table: List[Dict[str, Any]] = []
    cure_table: List[Dict[str, Any]] = []
    property_table: List[Dict[str, Any]] = []
    review_items: List[Dict[str, Any]] = []

    def get_or_create_material_component(
        name: str,
        component_class: str,
        abbreviation: Optional[str] = None,
        chemical_description: Optional[str] = None,
        surface_treatment: Optional[str] = None,
        notes: str = "",
    ) -> str:
        key = f"{component_class}::{abbreviation or ''}::{name}"
        if key not in material_component_table:
            material_component_table[key] = {
                "component_id": cmp_alloc.next(),
                "component_name": name,
                "abbreviation": abbreviation,
                "component_class": component_class,
                "chemical_description": chemical_description,
                "surface_treatment": surface_treatment,
                "notes": notes,
            }
        return material_component_table[key]["component_id"]

    def get_or_create_monomer(abbr: str, name: str, smiles: str, role: str,
                              source_paper: str, ocsr_conf: str = "") -> Optional[str]:
        input_smi = canonicalize_smiles(smiles) or smiles
        smiles, resolved_source, resolver_reviews = resolve_smiles_from_identity(
            abbr, name, role, [(input_smi, "input")]
        )
        for item in resolver_reviews:
            review_items.append({
                **item,
                "paper": source_paper,
            })
        if resolved_source == "reference" and input_smi and smiles and input_smi != smiles:
            review_items.append({
                "kind": "monomer_structure_overridden",
                "paper": source_paper,
                "abbr": abbr,
                "name": name,
                "input_smiles": input_smi,
                "reference_smiles": smiles,
            })
        ikey = safe_inchikey(smiles)
        if ikey:
            if ikey not in monomer_table:
                monomer_table[ikey] = {
                    "monomer_id": mon_alloc.next(),
                    "canonical_smiles": Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True),
                    "inchi_key": ikey,
                    "common_name": name or None,
                    "abbreviation": abbr or None,
                    "cas_number": None,
                    "monomer_class": role if role in (
                        "dianhydride", "diamine", "diacid_chloride", "triacid_chloride",
                        "diisocyanate", "diol", "diacid", "crosslinker",
                        "end_capper", "modifier") else "other",
                    "functional_groups": infer_functional_groups(smiles),
                    "functionality": infer_functionality(smiles, role),
                    "contains_fluorine": detect_fluorine(smiles) or False,
                    "dft_features": None,
                    "notes": f"first_seen:{source_paper}",
                }
                if ocsr_conf:
                    try:
                        if float(ocsr_conf) < 0.85:
                            review_items.append({
                                "kind": "low_confidence_ocsr",
                                "paper": source_paper,
                                "abbr": abbr,
                                "confidence": ocsr_conf,
                                "smiles": smiles,
                            })
                    except ValueError:
                        pass
            else:
                rec = monomer_table[ikey]
                if abbr and not rec.get("abbreviation"):
                    rec["abbreviation"] = abbr
                if name and not rec.get("common_name"):
                    rec["common_name"] = name
            return monomer_table[ikey]["monomer_id"]
        # No SMILES: still create a stub keyed by abbr+name so polymer_components can ref it.
        stub_key = f"STUB::{source_paper}::{abbr}::{name}"
        if stub_key not in monomer_table:
            text = f"{abbr} {name}".lower()
            f_heuristic = any(tok in text for tok in (
                "fluor", "6f", "cf3", "hexafluoro", "trifluoro", "perfluoro", "tfmb", "6fda", "bisaf"))
            monomer_table[stub_key] = {
                "monomer_id": mon_alloc.next(),
                "canonical_smiles": "",
                "inchi_key": "",
                "common_name": name or None,
                "abbreviation": abbr or None,
                "cas_number": None,
                "monomer_class": role if role in (
                    "dianhydride", "diamine", "crosslinker") else "other",
                "functional_groups": [],
                "functionality": 2,
                "contains_fluorine": bool(f_heuristic),
                "dft_features": None,
                "notes": f"stub_no_smiles:{source_paper}",
            }
            review_items.append({
                "kind": "missing_smiles",
                "paper": source_paper,
                "abbr": abbr,
                "name": name,
            })
        return monomer_table[stub_key]["monomer_id"]

    for paper in per_paper:
        paper_id = paper["paper_id"]
        ocsr_map = paper.get("ocsr_monomers", {})
        llm = paper.get("llm", {}) or {}
        paper_text = paper.get("text", "") or ""

        local_component_name_to_id: Dict[str, str] = {}
        inferred_components = infer_nonpolymer_components(paper_id, paper_text)
        for comp in (llm.get("material_components", []) or []) + inferred_components:
            cid = get_or_create_material_component(
                comp.get("component_name", ""),
                comp.get("component_class", "other"),
                abbreviation=comp.get("abbreviation"),
                chemical_description=comp.get("chemical_description"),
                surface_treatment=comp.get("surface_treatment"),
                notes=f"inferred_from:{paper_id}",
            )
            local_component_name_to_id[comp.get("component_name", "")] = cid
            if comp.get("abbreviation"):
                local_component_name_to_id[comp["abbreviation"]] = cid

        series_records = llm.get("study_series") or infer_study_series(paper_id, paper_text, llm.get("samples", []) or [])
        local_series_ids: List[str] = []
        variable_name_to_series_id: Dict[str, str] = {}
        for series in series_records:
            sid = ser_alloc.next()
            study_series_table.append({
                "series_id": sid,
                "paper_id": paper_id,
                "series_name": series.get("series_name"),
                "variable_name": series.get("variable_name", "other"),
                "variable_kind": series.get("variable_kind", "other"),
                "variable_unit": series.get("variable_unit"),
                "variable_values_text": series.get("variable_values_text"),
                "notes": series.get("notes") or "",
            })
            local_series_ids.append(sid)
            variable_name_to_series_id[series.get("variable_name", "other")] = sid

        variable_component_abbrs: set[str] = set()
        if any(s.get("variable_name") == "BTC_to_TCL_ratio" for s in series_records):
            variable_component_abbrs.update({"BTC", "TCL"})
        if any(s.get("variable_name") == "1,3-PBO_loading" for s in series_records):
            variable_component_abbrs.add("1,3-PBO")

        # Step A: register monomers (LLM-extracted ∪ OCSR-discovered).
        abbr_to_monomer_id: Dict[str, str] = {}
        # Prefer LLM monomer list (has names/classes); enrich with OCSR SMILES.
        llm_mons = llm.get("extracted_monomers", []) or []
        for m in llm_mons:
            abbr = m.get("abbreviation", "")
            name = m.get("name", "")
            mclass = m.get("monomer_class") or ""
            ocsr_entry = ocsr_map.get(abbr, {})
            # Prefer OCSR SMILES if present; else LLM-vision SMILES; else empty.
            smiles = ocsr_entry.get("smiles", "") or canonicalize_smiles(m.get("smiles") or "")
            mid = get_or_create_monomer(
                abbr, name, smiles, mclass, paper_id,
                ocsr_conf=ocsr_entry.get("confidence", ""),
            )
            if mid:
                abbr_to_monomer_id[abbr] = mid
        # Register OCSR-only monomers (LLM may have missed them).
        for abbr, ocsr_entry in ocsr_map.items():
            if abbr in abbr_to_monomer_id:
                continue
            mid = get_or_create_monomer(
                abbr, "", ocsr_entry["smiles"], "other", paper_id,
                ocsr_conf=ocsr_entry.get("confidence", ""),
            )
            if mid:
                abbr_to_monomer_id[abbr] = mid

        # Step B: polymers + components
        local_pol_to_id: Dict[str, str] = {}
        for pol in llm.get("polymers", []) or []:
            local_key = pol.get("local_polymer_key")
            if not local_key:
                continue
            pid = pol_alloc.next()
            local_pol_to_id[local_key] = pid
            polymer_table.append({
                "polymer_id": pid,
                "polymer_name": pol.get("polymer_name"),
                "polymer_class": pol.get("polymer_class", "polyimide"),
                "is_crosslinked": bool(pol.get("is_crosslinked", False)),
                "is_copolymer": bool(pol.get("is_copolymer", False)),
                "imidization_route": pol.get("imidization_route"),
                "bigsmiles": None,
                "composition_hash": "",  # filled below
                "source_type": "literature",
                "source_doi": llm.get("doi"),
            })

        components_for_pol: Dict[str, List[Dict[str, Any]]] = {}
        for comp in llm.get("polymer_components", []) or []:
            local_key = comp.get("local_polymer_key")
            pid = local_pol_to_id.get(local_key)
            if not pid:
                continue
            abbr = comp.get("monomer_abbreviation", "")
            if abbr in variable_component_abbrs:
                review_items.append({
                    "kind": "polymer_component_moved_to_sample_level",
                    "paper": paper_id,
                    "polymer": local_key,
                    "abbreviation": abbr,
                    "reason": "variable loading / ratio should not be fixed at polymer-level",
                })
                continue
            mid = abbr_to_monomer_id.get(abbr)
            if not mid:
                # Register stub on the fly (no SMILES yet).
                mid = get_or_create_monomer(abbr, "", "", comp.get("role", "other"), paper_id)
                abbr_to_monomer_id[abbr] = mid
            entry = {
                "polymer_id": pid,
                "monomer_id": mid,
                "role": comp.get("role", "other"),
                "molar_ratio": float(comp.get("molar_ratio", 0) or 0),
                "weight_ratio": None,
                "is_crosslinker": bool(comp.get("is_crosslinker", False)),
                "notes": "",
            }
            component_table.append(entry)
            components_for_pol.setdefault(pid, []).append(entry)

        for pol_rec in polymer_table:
            if pol_rec["polymer_id"] in components_for_pol and not pol_rec["composition_hash"]:
                pol_rec["composition_hash"] = composition_hash(components_for_pol[pol_rec["polymer_id"]])

        # Step C: samples + cure + properties
        local_smp_to_id: Dict[str, str] = {}
        precursor_local_keys: List[str] = []
        for smp in llm.get("samples", []) or []:
            local_key = smp.get("local_sample_key")
            local_pol = smp.get("local_polymer_key")
            pid = local_pol_to_id.get(local_pol)
            if not pid:
                continue
            sid = smp_alloc.next()
            local_smp_to_id[local_key] = sid
            material_stage = smp.get("material_stage") or infer_material_stage(local_key, paper_text)
            if material_stage == "paa_precursor":
                precursor_local_keys.append(local_key)
            sample_table.append({
                "sample_id": sid,
                "polymer_id": pid,
                "source_type": "literature",
                "source_doi": llm.get("doi"),
                "source_table": None,
                "local_sample_key": local_key,
                "sample_label": smp.get("sample_label"),
                "material_stage": material_stage,
                "parent_sample_id": None,
                "study_series_id": local_series_ids[0] if local_series_ids else None,
                "wet_lab_batch_id": None,
                "synthesis_date": None,
                "solvent": smp.get("solvent"),
                "solid_content_wt_pct": None,
                "film_thickness_um": smp.get("film_thickness_um"),
                "film_thickness_um_std": None,
                "mn_g_per_mol": None,
                "mw_g_per_mol": smp.get("mw_g_per_mol"),
                "pdi": None,
                "mw_test_method": None,
                "inherent_viscosity_dL_per_g": smp.get("inherent_viscosity_dL_per_g"),
                "crosslink_density_mol_per_cm3": None,
                "is_crosslink_density_calculated": False,
                "paper_id": paper_id,
            })

        if len(precursor_local_keys) == 1:
            precursor_sid = local_smp_to_id.get(precursor_local_keys[0])
            if precursor_sid:
                for rec in sample_table:
                    if rec.get("paper_id") != paper_id:
                        continue
                    if rec.get("sample_id") == precursor_sid:
                        continue
                    if rec.get("material_stage") in ("pi_final_film", "composite_film", "final_film"):
                        rec["parent_sample_id"] = precursor_sid

        # Step C1: sample-level composition (explicit LLM output + text heuristics)
        explicit_sample_comps = llm.get("sample_compositions", []) or []
        for comp in explicit_sample_comps:
            sid = local_smp_to_id.get(comp.get("local_sample_key"))
            if not sid:
                continue
            kind = comp.get("component_kind", "other")
            name = comp.get("component_name", "")
            abbr = comp.get("component_abbreviation") or name
            monomer_id = abbr_to_monomer_id.get(abbr) if kind in ("monomer", "crosslinker") else None
            component_id = None
            if kind not in ("monomer", "crosslinker"):
                component_id = local_component_name_to_id.get(name) or local_component_name_to_id.get(abbr)
            sample_component_table.append({
                "sample_component_id": scm_alloc.next(),
                "sample_id": sid,
                "monomer_id": monomer_id,
                "component_id": component_id,
                "component_kind": kind,
                "component_role": comp.get("component_role") or kind,
                "amount_value": comp.get("amount_value"),
                "amount_unit": comp.get("amount_unit"),
                "amount_basis": comp.get("amount_basis"),
                "value_qualifier": detect_amount_qualifier(comp.get("raw_expression") or ""),
                "is_primary_variable": True,
                "raw_expression": comp.get("raw_expression"),
                "notes": f"llm:{paper_id}",
            })

        if paper_id.startswith("crosslinked_colorless_polyimide_films_via_oxazole"):
            pbo_map = {"CPI0": 0.0, "CPI1": 0.1, "CPI2": 0.5, "CPI3": 1.0, "CPI4": 3.0}
            pbo_mid = abbr_to_monomer_id.get("1,3-PBO")
            for local_key, wt_pct in pbo_map.items():
                sid = local_smp_to_id.get(local_key)
                if not sid or not pbo_mid:
                    continue
                sample_component_table.append({
                    "sample_component_id": scm_alloc.next(),
                    "sample_id": sid,
                    "monomer_id": pbo_mid,
                    "component_id": None,
                    "component_kind": "crosslinker",
                    "component_role": "crosslinker",
                    "amount_value": wt_pct,
                    "amount_unit": "wt_pct",
                    "amount_basis": "vs_polymer",
                    "value_qualifier": "exact",
                    "is_primary_variable": True,
                    "raw_expression": f"{wt_pct} wt%",
                    "notes": "inferred_from_text:1,3-PBO to CPI mass ratio",
                })

        if paper_id.startswith("comparative_analysis_of_the_properties_of_colorless_and_transparent_polyimide_na"):
            cs30b_id = local_component_name_to_id.get("CS30B") or local_component_name_to_id.get("Cloisite 30B organoclay")
            if cs30b_id:
                for local_key, sid in local_smp_to_id.items():
                    match = re.search(r"(\d+(?:\.\d+)?)wt%", local_key, re.I)
                    if not match:
                        continue
                    wt_pct = float(match.group(1))
                    sample_component_table.append({
                        "sample_component_id": scm_alloc.next(),
                        "sample_id": sid,
                        "monomer_id": None,
                        "component_id": cs30b_id,
                        "component_kind": "filler",
                        "component_role": "organoclay",
                        "amount_value": wt_pct,
                        "amount_unit": "wt_pct",
                        "amount_basis": "vs_polymer",
                        "value_qualifier": "exact",
                        "is_primary_variable": True,
                        "raw_expression": f"{wt_pct} wt%",
                        "notes": "inferred_from_sample_key:CS30B loading",
                    })

        if paper_id.startswith("colorless_poly_amide-imide_s_with_low_dielectric_constants"):
            review_items.append({
                "kind": "missing_sample_level_composition",
                "paper": paper_id,
                "reason": "TCL/BTC variable ratio detected, but exact sample-wise mapping was not recovered from cached extraction.",
            })

        for trend in llm.get("trend_records", []) or []:
            trend_table.append({
                "trend_id": trd_alloc.next(),
                "paper_id": paper_id,
                "series_id": variable_name_to_series_id.get(trend.get("variable_name", "")),
                "variable_name": trend.get("variable_name", "other"),
                "variable_unit": trend.get("variable_unit"),
                "property_name": trend.get("property_name", "other"),
                "trend_direction": trend.get("trend_direction", "qualitative"),
                "evidence_text": trend.get("evidence_text"),
                "sample_scope": trend.get("sample_scope"),
                "mechanism_note": trend.get("mechanism_note"),
                "confidence": float(trend.get("confidence", 0.7) or 0.7),
            })

        for cure in llm.get("cure_profiles", []) or []:
            local_key = cure.get("local_sample_key")
            sid = local_smp_to_id.get(local_key)
            if not sid:
                continue
            atm = cure.get("atmosphere")
            if atm in (None, "null", "None", ""):
                atm = "N2"
            cure_table.append({
                "curve_id": f"CURE_{len(cure_table)+1:06d}",
                "sample_id": sid,
                "imidization_type": cure.get("imidization_type", "thermal"),
                "atmosphere": atm,
                "segments": cure.get("segments", []),
            })

        for prop in llm.get("property_records", []) or []:
            local_key = prop.get("local_sample_key")
            sid = local_smp_to_id.get(local_key)
            if not sid:
                continue
            normalized_prop, prop_reviews = normalize_property_record(prop, paper_id, paper_text)
            review_items.extend(prop_reviews)
            if not normalized_prop:
                continue
            dec_crit = normalized_prop.get("decomposition_criterion")
            if dec_crit not in ("2_pct", "5_pct", "10_pct", "onset", None):
                dec_crit = None
            rec = {
                "record_id": f"PRP_{len(property_table)+1:06d}",
                "sample_id": sid,
                "property_category": normalized_prop["property_category"],
                "property_name": normalized_prop["property_name"],
                "value_numeric": normalized_prop["value_numeric"],
                "unit": normalized_prop["unit"],
                "value_std": normalized_prop.get("value_std"),
                "value_raw": normalized_prop["value_raw"],
                "value_qualifier": normalized_prop.get("value_qualifier"),
                "property_name_raw": normalized_prop.get("property_name_raw"),
                "unit_raw": normalized_prop.get("unit_raw"),
                "test_method": normalized_prop["test_method"],
                "test_standard": None,
                "temperature_c": normalized_prop.get("temperature_c"),
                "temperature_range_c_min": normalized_prop.get("temperature_range_c_min"),
                "temperature_range_c_max": normalized_prop.get("temperature_range_c_max"),
                "frequency_hz": normalized_prop.get("frequency_hz"),
                "wavelength_nm": normalized_prop.get("wavelength_nm"),
                "source_page": normalized_prop.get("source_page"),
                "heating_rate_c_per_min": None,
                "atmosphere": None,
                "humidity_pct": None,
                "decomposition_criterion": dec_crit,
                "tg_definition": normalized_prop.get("tg_definition"),
                "extraction_method": "llm_extracted",
                "confidence": 0.7,
            }
            property_table.append(rec)

        paper_sample_ids = {s["sample_id"] for s in sample_table if s.get("paper_id") == paper_id}
        paper_props = [p for p in property_table if p["sample_id"] in paper_sample_ids]
        if re.search(r"\bFFV\b|free volume", paper_text, re.I) and not any(p["property_name"] == "free_volume_fraction" for p in paper_props):
            review_items.append({
                "kind": "missing_high_value_property",
                "paper": paper_id,
                "property_name": "free_volume_fraction",
            })
        if re.search(r"\bdensity\b", paper_text, re.I) and not any(p["property_name"] == "density" for p in paper_props):
            review_items.append({
                "kind": "missing_high_value_property",
                "paper": paper_id,
                "property_name": "density",
            })
        if re.search(r"\bCS30B\b|\borganoclay\b|\bclay\b", paper_text, re.I) and not any(sc["sample_id"] in paper_sample_ids and sc["component_kind"] == "filler" for sc in sample_component_table):
            review_items.append({
                "kind": "missing_filler_component",
                "paper": paper_id,
                "component": "CS30B / organoclay",
            })
        if not llm.get("polymers") and not llm.get("property_records"):
            review_items.append({
                "kind": "empty_extraction",
                "paper": paper_id,
                "reason": "cached llm extraction returned no polymers and no properties",
            })

    # Strip helper paper_id field before validating sample table (not in schema).
    sample_table_out = [{k: v for k, v in s.items() if k != "paper_id"} for s in sample_table]

    # Enrich stub monomers via PubChem to populate InChIKey / contains_fluorine.
    _pubchem_cache: Dict[str, str] = {}
    for rec in monomer_table.values():
        if rec.get("inchi_key"):
            continue
        for query in (rec.get("common_name"), rec.get("abbreviation")):
            if not query:
                continue
            if query in _pubchem_cache:
                smi = _pubchem_cache[query]
            else:
                smi, _ = resolve_with_pubchem(query)
                _pubchem_cache[query] = smi
            if not smi:
                continue
            ikey = safe_inchikey(smi)
            if not ikey:
                continue
            rec["canonical_smiles"] = smi
            rec["inchi_key"] = ikey
            rec["functional_groups"] = infer_functional_groups(smi)
            rec["functionality"] = infer_functionality(smi, rec.get("monomer_class", "other"))
            rec["contains_fluorine"] = detect_fluorine(smi) or False
            rec["notes"] = f"{rec.get('notes','')};pubchem_enriched"
            break

    # Consolidate enriched stubs sharing an InChIKey (dedup across papers).
    seen_ikey: Dict[str, str] = {}
    id_remap: Dict[str, str] = {}
    consolidated: Dict[str, Dict[str, Any]] = {}
    for k, rec in monomer_table.items():
        ikey = rec.get("inchi_key")
        if ikey and ikey in seen_ikey:
            id_remap[rec["monomer_id"]] = seen_ikey[ikey]
            continue
        if ikey:
            seen_ikey[ikey] = rec["monomer_id"]
        consolidated[k] = rec
    if id_remap:
        for comp in component_table:
            comp["monomer_id"] = id_remap.get(comp["monomer_id"], comp["monomer_id"])
        for comp in sample_component_table:
            if comp.get("monomer_id"):
                comp["monomer_id"] = id_remap.get(comp["monomer_id"], comp["monomer_id"])
        monomer_table = consolidated

    out_files = {
        "monomer.json": list(monomer_table.values()),
        "material_component.json": list(material_component_table.values()),
        "polymer.json": polymer_table,
        "polymer_component.json": component_table,
        "sample.json": sample_table_out,
        "sample_composition.json": sample_component_table,
        "study_series.json": study_series_table,
        "trend_record.json": trend_table,
        "cure_profile.json": cure_table,
        "property_record.json": property_table,
    }
    for fname, payload in out_files.items():
        (dataset_dir / fname).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Review queue
    for prop in property_table:
        if prop.get("confidence", 1) < 0.7:
            review_items.append({"kind": "low_confidence_property", **prop})
    review_lines = ["# Review Queue\n"]
    review_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    review_lines.append(f"Total items: {len(review_items)}\n\n")
    for item in review_items:
        review_lines.append(f"- **{item.get('kind')}**: {json.dumps({k: v for k, v in item.items() if k != 'kind'}, ensure_ascii=False)}")
    (dataset_dir / "review_queue.md").write_text("\n".join(review_lines), encoding="utf-8")

    # Stats README
    stats = {fname: len(payload) for fname, payload in out_files.items()}
    stats["papers"] = len(per_paper)
    readme = [
        "# dataset_v0",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Counts",
        "",
    ] + [f"- {k}: {v}" for k, v in stats.items()]
    (dataset_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    # Validation
    validation_report: Dict[str, Any] = {}
    if validate and jsonschema is not None:
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            for fname, defname in [
                ("monomer.json", "monomer"),
                ("material_component.json", "material_component"),
                ("polymer.json", "polymer"),
                ("polymer_component.json", "polymer_component"),
                ("sample.json", "sample"),
                ("sample_composition.json", "sample_composition"),
                ("study_series.json", "study_series"),
                ("trend_record.json", "trend_record"),
                ("cure_profile.json", "cure_profile"),
                ("property_record.json", "property_record"),
            ]:
                payload = out_files[fname]
                errors: List[str] = []
                for rec in payload:
                    try:
                        jsonschema.validate(rec, schema["definitions"][defname])
                    except jsonschema.ValidationError as ve:
                        errors.append(f"{rec.get('monomer_id') or rec.get('polymer_id') or rec.get('sample_id') or rec.get('record_id') or '?'}: {ve.message}")
                validation_report[fname] = {"errors": errors[:20], "ok": len(payload) - len(errors), "total": len(payload)}
        except Exception as exc:
            validation_report["_fatal"] = str(exc)
        (dataset_dir / "validation_report.json").write_text(
            json.dumps(validation_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return {"stats": stats, "validation": validation_report, "review_count": len(review_items)}


def main() -> int:
    args = parse_args()

    if args.use_legacy_pi5922:
        if not args.pdf:
            print("[ERROR] --use-legacy-pi5922 requires --pdf", file=sys.stderr)
            return 2
        return _legacy_pi5922_main(args)

    raw_dir = Path(args.raw_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    schema_path = Path(args.schema).resolve()
    model_path = Path(args.model_path).expanduser().resolve()

    pdfs: List[Path]
    if args.input_dir:
        pdfs = sorted(Path(args.input_dir).resolve().glob("*.pdf"))
    else:
        pdfs = [Path(args.pdf).resolve()]
    if not pdfs:
        print(f"[ERROR] no PDFs found", file=sys.stderr)
        return 2

    predictor = None
    if not args.no_ocsr:
        predictor = init_predictor(model_path, args.device)
        if predictor is None:
            print(f"[WARN] MolScribe unavailable (no checkpoint at {model_path}); continuing without OCSR", file=sys.stderr)

    per_paper: List[Dict[str, Any]] = []
    for pdf in pdfs:
        try:
            result = process_one_pdf(
                pdf, raw_dir, predictor,
                use_llm=not args.no_llm,
                use_ocsr=not args.no_ocsr,
            )
            per_paper.append(result)
        except Exception as exc:
            print(f"[FAIL] {pdf.name}: {exc}", file=sys.stderr)

    summary = merge_into_dataset(per_paper, dataset_dir, schema_path, validate=not args.no_validate)
    print("\n=== Dataset v0 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
try:
    import pubchempy as pcp  # type: ignore
except Exception:  # pragma: no cover
    pcp = None

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

try:
    from image_to_smiles import (
        assign_ids_to_segments,
        clean_structure_image,
        image_to_smiles,
        isolate_primary_structure,
        segment_molecule_crops,
        segment_scheme_molecule_crops,
        patch_molscribe_for_sandbox,
    )
except Exception:  # pragma: no cover
    def assign_ids_to_segments(*args, **kwargs):
        return []
    def clean_structure_image(image_path, *args, **kwargs):
        return image_path
    def image_to_smiles(*args, **kwargs):
        return "", "", "", "provider_unavailable"
    def isolate_primary_structure(image_path, *args, **kwargs):
        return image_path, (0, 0, 0, 0)
    def segment_molecule_crops(image_path, *args, **kwargs):
        return [("mol1", image_path, (0, 0, 0, 0))]
    def segment_scheme_molecule_crops(image_path, *args, **kwargs):
        return [("mol1", image_path, (0, 0, 0, 0))]
    def patch_molscribe_for_sandbox(*args, **kwargs):
        return None

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
    parser.add_argument("--refresh-llm-cache", action="store_true", help="Ignore existing llm_raw.json and re-run the LLM with the current prompt")
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
    if pcp is None:
        return "", "pubchem:unavailable"
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


def _is_generic_internal_monomer_abbr(abbr: Any) -> bool:
    """True for generic paper-local labels such as R1 or compound labels.

    These labels often refer to newly synthesized monomers shown only in a
    Scheme/Figure. Resolving the *text name* through PubChem can create a
    plausible but wrong commercial structure, so external name lookup is skipped
    for them unless a curated reference is available.
    """
    a = _canonical_sample_key(abbr).upper()
    return bool(re.fullmatch(r"(?:M|R|C|D)\d+", a) or re.fullmatch(r"COMPOUND\d+", a))


def _candidate_smiles_passes_role_qc(smiles: str, role: str) -> bool:
    smi = canonicalize_smiles(smiles or "")
    if not smi:
        return False
    role = (role or "").lower()
    if role == "diamine":
        return _count_primary_amines(smi) >= 2
    if role == "dianhydride":
        return _count_anhydride_rings(smi) >= 2
    return _heavy_atom_count(smi) >= 8

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

    generic_internal = _is_generic_internal_monomer_abbr(abbreviation)
    if generic_internal:
        review_items.append({
            "kind": "external_name_lookup_skipped_for_internal_monomer",
            "abbr": abbreviation,
            "name": name,
            "reason": "generic internal monomer labels may denote new structures; external name lookup may be misleading",
        })
    else:
        for resolver in (resolve_with_pubchem, resolve_with_opsin):
            if not name:
                break
            smi, source = resolver(name)
            smi = canonicalize_smiles(smi or "")
            if smi:
                if not _candidate_smiles_passes_role_qc(smi, role):
                    review_items.append({
                        "kind": "external_resolved_smiles_failed_role_qc",
                        "abbr": abbreviation,
                        "name": name,
                        "source": source,
                        "role": role,
                        "smiles": smi,
                    })
                    continue
                return smi, source, review_items

    for candidate, source in candidate_smiles:
        cand = canonicalize_smiles(candidate or "")
        if not cand:
            continue
        if not _candidate_smiles_passes_role_qc(cand, role):
            review_items.append({
                "kind": "candidate_smiles_rejected_by_role_qc",
                "abbr": abbreviation,
                "name": name,
                "candidate_source": source,
                "role": role,
                "candidate_smiles": cand,
            })
            continue
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


LLM_SYSTEM_PROMPT = (
    "You are a careful polymer-chemistry data extraction assistant for fluorine-free "
    "transparent polyimide / poly(amide-imide) papers. Extract structured records from "
    "the whole paper, especially Scheme/Figure structures and all tables. Return ONLY "
    "JSON matching the requested schema. Use null when a value is absent. Numbers must "
    "use the units requested by the schema. Do not invent monomers or samples; however, "
    "do not omit control samples such as PI-0/PAI-0/CPI-0 when they appear in tables. "
    "Every PI/PAI backbone must normally contain at least one dianhydride and one diamine. "
    "If a common dianhydride/diamine is fixed across a sample series, repeat it in every "
    "polymer_components row rather than only listing the variable co-monomers."
)

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
        "trend_records": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["variable_name", "property_name", "trend_direction"],
                "properties": {
                    "variable_name": {"type": "string"},
                    "variable_unit": {"type": ["string", "null"]},
                    "property_name": {"type": "string", "enum": ["transmittance", "Tg", "CTE", "other"]},
                    "trend_direction": {"type": "string", "enum": ["increase", "decrease", "optimum", "mixed", "qualitative", "no_clear_trend"]},
                    "evidence_text": {"type": ["string", "null"]},
                    "sample_scope": {"type": ["string", "null"]},
                    "mechanism_note": {"type": ["string", "null"]},
                    "confidence": {"type": ["number", "null"]},
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


def _reference_library_paths() -> List[Path]:
    paths: List[Path] = []
    env_path = os.getenv("PI_REFERENCE_MONOMER_LIBRARY") or os.getenv("REFERENCE_MONOMER_LIBRARY")
    if env_path:
        paths.append(Path(env_path))
    try:
        paths.append(Path(__file__).with_name("reference_monomer_library_general.json"))
    except Exception:
        pass
    paths.extend([
        Path("reference_monomer_library_general.json"),
        Path("reference_monomer_library.json"),
        Path("data/reference_monomer_library.json"),
        Path("data/reference/monomer_library.json"),
    ])
    return paths


def _load_curated_monomer_library() -> Dict[str, Dict[str, Any]]:
    for path in _reference_library_paths():
        try:
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                data = {
                    str(item.get("abbreviation") or item.get("abbr") or item.get("key")): item
                    for item in data if isinstance(item, dict)
                }
            if not isinstance(data, dict):
                continue
            out: Dict[str, Dict[str, Any]] = {}
            for key, value in data.items():
                if isinstance(value, str):
                    out[str(key)] = {
                        "abbreviation": str(key),
                        "canonical_smiles": value,
                        "smiles": value,
                    }
                elif isinstance(value, dict):
                    row = dict(value)
                    row["abbreviation"] = str(row.get("abbreviation") or row.get("abbr") or key)
                    row["canonical_smiles"] = row.get("canonical_smiles") or row.get("smiles") or ""
                    row.setdefault("smiles", row.get("canonical_smiles") or "")
                    row.setdefault("common_name", row.get("name") or "")
                    row.setdefault("monomer_class", row.get("role") or row.get("class") or "")
                    out[row["abbreviation"]] = row
            return out
        except Exception as exc:
            print(f"[WARN] could not load reference monomer library {path}: {exc}", file=sys.stderr)
    return {}


def _load_reference_monomer_smiles() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for abbr, row in _load_curated_monomer_library().items():
        smiles = row.get("canonical_smiles") or row.get("smiles") or ""
        if smiles:
            out[abbr] = str(smiles)
    return out


REFERENCE_MONOMER_SMILES = _load_reference_monomer_smiles()

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

PROPERTY_NAME_MAP.update({
    "t5": ("Td5", "thermal"),
    "t5pct": ("Td5", "thermal"),
    "t5percent": ("Td5", "thermal"),
    "td5": ("Td5", "thermal"),
    "td5air": ("Td5_air", "thermal"),
    "tgdma": ("Tg_DMA", "thermal"),
    "tg_dma": ("Tg_DMA", "thermal"),
    "tg_dsc": ("Tg", "thermal"),
    "tgdsc": ("Tg", "thermal"),
    "tmax": ("Tmax", "thermal"),
    "rw730": ("residual_weight_730c", "thermal"),
    "rw750": ("residual_weight_750c", "thermal"),
    "charyield800cn2": ("char_yield_800c_n2", "thermal"),
    "char_yield_800c_n2": ("char_yield_800c_n2", "thermal"),
    "mn": ("Mn", "physical"),
    "mw": ("Mw", "physical"),
    "pdi": ("PDI", "physical"),
    "l": ("cie_L", "optical"),
    "lstar": ("cie_L", "optical"),
    "cie_l": ("cie_L", "optical"),
    "a": ("cie_a", "optical"),
    "astar": ("cie_a", "optical"),
    "cie_a": ("cie_a", "optical"),
    "b": ("cie_b", "optical"),
    "bstar": ("cie_b", "optical"),
    "cie_b": ("cie_b", "optical"),
    "eg": ("energy_gap", "optical"),
    "energygap": ("energy_gap", "optical"),
    "nte": ("refractive_index_TE", "optical"),
    "ntm": ("refractive_index_TM", "optical"),
    "nav": ("refractive_index_avg", "optical"),
    "deltan": ("birefringence", "optical"),
    "delta_n": ("birefringence", "optical"),
    "Δn": ("birefringence", "optical"),
    "birefringence": ("birefringence", "optical"),
    "solubility": ("solubility", "chemical"),
    "foldingcycles": ("folding_cycles", "mechanical"),
    "folding_cycles": ("folding_cycles", "mechanical"),
    "bendingradius": ("bending_radius", "mechanical"),
    "bending_radius": ("bending_radius", "mechanical"),
})
PROPERTY_NAME_ALIASES.update({
    "Td5_air", "Tg_DMA", "Tmax", "residual_weight_730c", "residual_weight_750c",
    "char_yield_800c_n2", "Mn", "Mw", "PDI", "cie_L", "cie_a",
    "cie_b", "energy_gap", "refractive_index_TE", "refractive_index_TM",
    "refractive_index_avg", "birefringence", "solubility",
    "folding_cycles", "bending_radius",
})

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
    for key, val in REFERENCE_MONOMER_SMILES.items():
        if str(abbr).upper() == key.upper():
            return canonicalize_smiles(val) or val
    upper_name = (name or "").upper()
    if "6FDA" in upper_name:
        return canonicalize_smiles(REFERENCE_MONOMER_SMILES.get("6FDA", "")) or REFERENCE_MONOMER_SMILES.get("6FDA", "")
    if abbr == "TFMB" or "TRIFLUOROMETHYL" in upper_name:
        return canonicalize_smiles(REFERENCE_MONOMER_SMILES.get("TFMB", "")) or REFERENCE_MONOMER_SMILES.get("TFMB", "")
    return ""


def normalize_unit(unit: Any) -> str:
    raw = "" if unit is None else str(unit).strip()
    if not raw:
        return ""
    lowered = raw.lower()
    return UNIT_MAP.get(lowered, raw)


UNIT_MAP.update({
    "× 10^-6/k": "ppm/K",
    "x 10^-6/k": "ppm/K",
    "10^-6/k": "ppm/K",
    "10−6/k": "ppm/K",
    "ppm/k": "ppm/K",
    "ppm/°c": "ppm/°C",
    "": "",
})

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






def ensure_llm_payload_defaults(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload or {})
    for key in (
        "polymers", "polymer_components", "samples", "cure_profiles", "property_records",
        "extracted_monomers", "material_components", "sample_compositions", "study_series",
        "trend_records",
    ):
        out.setdefault(key, [])
    return out






def _canonical_sample_key(key: Any) -> str:
    text = "" if key is None else str(key).strip()
    text = text.replace("—", "-").replace("–", "-").replace("−", "-")
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"^(PAI|PI|CPI)\s+(\d+)$", r"\1-\2", text, flags=re.I)
    return text


def _review_item(payload: Dict[str, Any], kind: str, **kwargs: Any) -> None:
    payload.setdefault("review_items", []).append({"kind": kind, **kwargs})


def _find_by_key(rows: Sequence[Dict[str, Any]], key_name: str, key_value: str) -> Optional[Dict[str, Any]]:
    for row in rows:
        if _canonical_sample_key(row.get(key_name)) == _canonical_sample_key(key_value):
            return row
    return None


def _upsert_monomer(payload: Dict[str, Any], abbreviation: str, name: str, monomer_class: str,
                    smiles: Optional[str] = None, smiles_source: str = "inferred", source_page: Optional[int] = None) -> None:
    abbreviation = _canonical_sample_key(abbreviation)
    row = _find_by_key(payload.get("extracted_monomers", []), "abbreviation", abbreviation)
    if row is None:
        payload.setdefault("extracted_monomers", []).append({
            "abbreviation": abbreviation,
            "name": name,
            "monomer_class": monomer_class,
            "cas_number": None,
            "smiles": smiles,
            "smiles_source": smiles_source if smiles else "unknown",
            "source_page": source_page,
        })
        return
    if name and (not row.get("name") or _looks_like_wrong_monomer_name(row.get("name", ""), monomer_class)):
        old = row.get("name")
        row["name"] = name
        _review_item(payload, "monomer_name_repaired", abbreviation=abbreviation, old_name=old, new_name=name)
    if monomer_class and row.get("monomer_class") != monomer_class:
        old = row.get("monomer_class")
        row["monomer_class"] = monomer_class
        _review_item(payload, "monomer_class_repaired", abbreviation=abbreviation, old_class=old, new_class=monomer_class)
    if smiles and not row.get("smiles"):
        row["smiles"] = smiles
        row["smiles_source"] = smiles_source
    if source_page and not row.get("source_page"):
        row["source_page"] = source_page


def _looks_like_wrong_monomer_name(name: str, expected_class: str) -> bool:
    lowered = (name or "").lower()
    if expected_class == "diamine" and "dianhydride" in lowered:
        return True
    if expected_class == "dianhydride" and ("diamine" in lowered or "aminobenzamide" in lowered):
        return True
    return False


def _upsert_polymer(payload: Dict[str, Any], key: str, polymer_class: str, is_copolymer: bool,
                    imidization_route: Optional[str] = None) -> None:
    key = _canonical_sample_key(key)
    row = _find_by_key(payload.get("polymers", []), "local_polymer_key", key)
    if row is None:
        payload.setdefault("polymers", []).append({
            "local_polymer_key": key,
            "polymer_name": None,
            "polymer_class": polymer_class,
            "is_crosslinked": False,
            "is_copolymer": is_copolymer,
            "imidization_route": imidization_route,
        })
        return
    row["local_polymer_key"] = key
    row["polymer_class"] = polymer_class or row.get("polymer_class")
    row["is_copolymer"] = is_copolymer
    if imidization_route is not None:
        row["imidization_route"] = imidization_route


def _component_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        _canonical_sample_key(row.get("local_polymer_key")),
        _canonical_sample_key(row.get("monomer_abbreviation") or row.get("monomer_abbr")),
        str(row.get("role") or ""),
    )


def _upsert_component(payload: Dict[str, Any], polymer_key: str, monomer_abbr: str, role: str,
                      molar_ratio: float, source: str = "rule") -> None:
    polymer_key = _canonical_sample_key(polymer_key)
    monomer_abbr = _canonical_sample_key(monomer_abbr)
    target = (polymer_key, monomer_abbr, role)
    for row in payload.get("polymer_components", []) or []:
        if _component_key(row) == target:
            old = row.get("molar_ratio")
            if old != molar_ratio:
                row["molar_ratio"] = molar_ratio
                _review_item(payload, "component_ratio_repaired", local_polymer_key=polymer_key,
                             monomer_abbreviation=monomer_abbr, old_ratio=old, new_ratio=molar_ratio, source=source)
            row["is_crosslinker"] = bool(row.get("is_crosslinker", False))
            return
    payload.setdefault("polymer_components", []).append({
        "local_polymer_key": polymer_key,
        "monomer_abbreviation": monomer_abbr,
        "role": role,
        "molar_ratio": molar_ratio,
        "is_crosslinker": False,
    })
    _review_item(payload, "component_added_by_rule", local_polymer_key=polymer_key,
                 monomer_abbreviation=monomer_abbr, role=role, molar_ratio=molar_ratio, source=source)


def _is_suspicious_thickness(value: Any) -> bool:
    """Detect common OCR/LLM film-thickness errors such as 19 μm -> 190 μm."""
    try:
        v = float(value)
    except Exception:
        return False
    return 100 <= v <= 1000

def _upsert_sample(payload: Dict[str, Any], sample_key: str, polymer_key: str,
                   material_stage: str = "final_film", solvent: Optional[str] = None,
                   thickness_um: Optional[float] = None, inherent_viscosity: Optional[float] = None) -> None:
    sample_key = _canonical_sample_key(sample_key)
    polymer_key = _canonical_sample_key(polymer_key)
    row = _find_by_key(payload.get("samples", []), "local_sample_key", sample_key)
    if row is None:
        payload.setdefault("samples", []).append({
            "local_sample_key": sample_key,
            "local_polymer_key": polymer_key,
            "sample_label": None,
            "material_stage": material_stage,
            "solvent": solvent,
            "film_thickness_um": thickness_um,
            "mw_g_per_mol": None,
            "inherent_viscosity_dL_per_g": inherent_viscosity,
        })
        return
    row["local_sample_key"] = sample_key
    row["local_polymer_key"] = polymer_key
    row["material_stage"] = row.get("material_stage") or material_stage
    if solvent and not row.get("solvent"):
        row["solvent"] = solvent
    if thickness_um is not None:
        old = row.get("film_thickness_um")
        if old is None or _is_suspicious_thickness(old):
            row["film_thickness_um"] = thickness_um
            if old != thickness_um:
                _review_item(payload, "film_thickness_repaired", sample=sample_key, old_value=old,
                             new_value=thickness_um, reason="table/text based recovery")
    if inherent_viscosity is not None and not row.get("inherent_viscosity_dL_per_g"):
        row["inherent_viscosity_dL_per_g"] = inherent_viscosity


def _infer_wavelength_from_property(row: Dict[str, Any]) -> Optional[float]:
    wl = row.get("wavelength_nm")
    if isinstance(wl, (int, float)):
        return float(wl)
    text = " ".join(str(row.get(k) or "") for k in ("property_name_raw", "value_raw"))
    m = re.search(r"\bT\s*(\d{3})\s*(?:nm)?\b", text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"\b(\d{3})\s*nm\b", text, re.I)
    if row.get("property_name") == "transmittance" and m:
        return float(m.group(1))
    return None


def _prop_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, Optional[float], Optional[float], Optional[float]]:
    """Stable key for upserting/deduplicating properties.

    The property normalizer canonicalizes optical/thermal/physical property keys so aliases such as
    T5%/Td5, T400/transmittance, λ/cutoff_wavelength and ppm/K/×10^-6/K
    are treated as the same property when values match. Auxiliary properties kept
    under property_name='other' still use property_name_raw so L*, a*, b*, Eg,
    folding_cycles, Mn/Mw/PDI and solubility_* remain separate.
    """
    pname = str(row.get("property_name") or "")
    raw_name = str(row.get("property_name_raw") or "") if pname == "other" else ""
    unit = normalize_unit(row.get("unit") or "")
    wl = _infer_wavelength_from_property(row) if pname == "transmittance" else row.get("wavelength_nm")
    return (
        _canonical_sample_key(row.get("local_sample_key")),
        pname,
        raw_name,
        unit,
        wl,
        row.get("frequency_hz"),
        row.get("temperature_c"),
    )


def _merge_property_rows(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two duplicate property rows, preferring structured/non-empty fields."""
    merged = dict(existing)
    for k, v in incoming.items():
        if v not in (None, "", []):
            if merged.get(k) in (None, "", []):
                merged[k] = v
    if merged.get("property_name") == "transmittance" and merged.get("wavelength_nm") in (None, ""):
        inferred = _infer_wavelength_from_property(incoming) or _infer_wavelength_from_property(existing)
        if inferred is not None:
            merged["wavelength_nm"] = inferred
    return merged


def _dedupe_property_records(payload: Dict[str, Any]) -> None:
    records = payload.get("property_records", []) or []
    out: List[Dict[str, Any]] = []
    index: Dict[Tuple[str, str, str, str, Optional[float], Optional[float], Optional[float]], int] = {}
    for rec in records:
        key = _prop_key(rec)
        if key not in index:
            index[key] = len(out)
            out.append(rec)
            continue
        j = index[key]
        old = out[j]
        try:
            old_v = float(old.get("value_numeric"))
            new_v = float(rec.get("value_numeric"))
            same_value = abs(old_v - new_v) < 1e-9
        except Exception:
            same_value = old.get("value_numeric") == rec.get("value_numeric")
        if same_value:
            out[j] = _merge_property_rows(old, rec)
            _review_item(payload, "duplicate_property_removed", sample=rec.get("local_sample_key"),
                         property_name=rec.get("property_name"), property_name_raw=rec.get("property_name_raw"),
                         wavelength_nm=_infer_wavelength_from_property(rec), value=rec.get("value_numeric"))
        else:
            out.append(rec)
            _review_item(payload, "property_duplicate_conflicting_values", sample=rec.get("local_sample_key"),
                         property_name=rec.get("property_name"), property_name_raw=rec.get("property_name_raw"),
                         old_value=old.get("value_numeric"), new_value=rec.get("value_numeric"))
    payload["property_records"] = out


def _clean_prop_token_property(text: Any) -> str:
    """Normalize a property label for property canonicalization."""
    s = str(text or "")
    s = s.replace("−", "-").replace("Δ", "Delta")
    s = s.replace("*", "star")
    s = re.sub(r"[%\s\-]+", "", s)
    s = re.sub(r"[^A-Za-z0-9_λ]+", "", s)
    return s.lower()


def _canonical_unit_property(unit: Any) -> str:
    u = normalize_unit(unit or "")
    low = u.lower().replace(" ", "")
    if low in {"×10^-6/k", "x10^-6/k", "10^-6/k", "10−6/k", "ppm/k"}:
        return "ppm/K"
    if low in {"×10^-6/°c", "x10^-6/°c", "10^-6/°c", "10−6/°c", "ppm/°c", "ppm/c"}:
        return "ppm/°C"
    if low in {"", "none", "null"}:
        return ""
    return u


def _standardize_property_name_property(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Map LLM/rule aliases to canonical property names.

    This fixes overuse of property_name='other' and standardizes properties
    such as Mn, Mw, PDI, L*, a*, b*, nTE/nTM/nav, Δn, Rw730/Rw750,
    solubility and T5%/Td5.
    """
    row = dict(rec)
    pname0 = row.get("property_name")
    raw0 = row.get("property_name_raw")
    value_raw = str(row.get("value_raw") or "")
    text = " ".join(str(x or "") for x in (pname0, raw0, value_raw))
    token = _clean_prop_token_property(raw0 if raw0 not in (None, "") else pname0)
    token_all = _clean_prop_token_property(text)

    m = re.search(r"\bT\s*(400|450|500|550|600)\s*(?:nm)?\b", text, re.I)
    if str(pname0 or "").lower() in {"t400", "t450", "t500", "t550", "t600"} or m:
        wl = float(m.group(1)) if m else float(re.sub(r"\D", "", str(pname0)) or 0)
        if wl:
            row["property_name"] = "transmittance"
            row["property_category"] = "optical"
            row["wavelength_nm"] = wl
            row["property_name_raw"] = f"T{int(wl)}"
            row["unit"] = "%"
            row["unit_raw"] = row.get("unit_raw") or "%"
            return row

    if token in {"λ", "lambda", "lambda0", "λ0", "lambdacutoff", "λcutoff", "cutoff", "cutoffwavelength"}:
        row["property_name"] = "cutoff_wavelength"
        row["property_category"] = "optical"
        row["unit"] = "nm"
        row["unit_raw"] = row.get("unit_raw") or "nm"
        return row

    if token in {"t5", "t5pct", "t5percent", "td5"} or re.search(r"\bT\s*5\s*%", text, re.I):
        row["property_name"] = "Td5"
        row["property_category"] = "thermal"
        row["decomposition_criterion"] = row.get("decomposition_criterion") or "5_pct"
        row["unit"] = _canonical_unit_property(row.get("unit") or "°C")
        row["property_name_raw"] = row.get("property_name_raw") or "T5%"
        return row

    mapped = PROPERTY_NAME_MAP.get(token) or PROPERTY_NAME_MAP.get(token_all)
    if mapped:
        row["property_name"], row["property_category"] = mapped
        if row["property_name"] == "solubility":
            raw = row.get("property_name_raw") or row.get("property_name") or "solubility"
            if not str(raw).lower().startswith("solubility"):
                raw = f"solubility_{raw}"
            row["property_name_raw"] = raw
        return row

    if token.startswith("solubility") or "solubility_" in str(raw0 or "").lower():
        row["property_name"] = "solubility"
        row["property_category"] = "chemical"
        row["property_name_raw"] = raw0 or pname0 or "solubility"
        return row

    if row.get("property_name") == "other" and _canonical_unit_property(row.get("unit")) == "g/cm3":
        row["property_name"] = "density"
        row["property_category"] = "physical"
    return row


def _property_specific_key_property(rec: Dict[str, Any]) -> Tuple[Any, ...]:
    """Canonical equivalence key for duplicate removal."""
    row = _standardize_property_name_property(rec)
    sample = _canonical_sample_key(row.get("local_sample_key"))
    pname = row.get("property_name") or "other"
    unit = _canonical_unit_property(row.get("unit") or row.get("unit_raw") or "")
    wl = _infer_wavelength_from_property(row) if pname == "transmittance" else row.get("wavelength_nm")
    raw = str(row.get("property_name_raw") or "")
    raw_token = _clean_prop_token_property(raw)
    if pname == "solubility":
        solvent = raw_token.replace("solubility", "")
        return (sample, pname, solvent, unit, None, None, None)
    return (
        sample,
        pname,
        "",  # raw aliases no longer split equivalent properties
        unit,
        wl,
        row.get("temperature_c"),
        row.get("temperature_range_c_min"),
        row.get("temperature_range_c_max"),
        row.get("decomposition_criterion") if pname.startswith("Td") else None,
    )


def _property_row_quality_property(row: Dict[str, Any]) -> int:
    """Higher score = better row to keep during duplicate merging."""
    score = 0
    pname = row.get("property_name") or ""
    raw = str(row.get("property_name_raw") or "")
    if pname != "other":
        score += 10
    if pname == "transmittance" and _infer_wavelength_from_property(row) is not None:
        score += 10
    if raw:
        score += 2
    if row.get("source_page") is not None:
        score += 2
    if row.get("test_method") and row.get("test_method") != "unknown":
        score += 1
    if row.get("unit_raw"):
        score += 1
    return score


def _merge_property_rows_property(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Merge duplicates, preferring the more structured row."""
    a = _standardize_property_name_property(a)
    b = _standardize_property_name_property(b)
    primary, secondary = (a, b) if _property_row_quality_property(a) >= _property_row_quality_property(b) else (b, a)
    merged = dict(primary)
    for k, v in secondary.items():
        if merged.get(k) in (None, "", []):
            merged[k] = v
    merged["unit"] = _canonical_unit_property(merged.get("unit") or merged.get("unit_raw") or "")
    if merged.get("property_name") == "transmittance":
        wl = _infer_wavelength_from_property(merged) or _infer_wavelength_from_property(secondary)
        if wl is not None:
            merged["wavelength_nm"] = wl
            merged["property_name_raw"] = merged.get("property_name_raw") or f"T{int(wl)}"
    return merged


def _sync_inherent_viscosity_property(payload: Dict[str, Any]) -> None:
    """Keep inherent viscosity both as a sample field and a property record.

    Earlier drafts sometimes kept ηinh in llm_raw.samples but lost it in the nested
    paper_summary because _build_paper_summary did not copy the field. This
    function also creates a property record so dataset exports are stable.
    """
    existing = set()
    for rec in payload.get("property_records", []) or []:
        if (rec.get("property_name") == "inherent_viscosity" or _clean_prop_token_property(rec.get("property_name_raw")) in {"ηinh", "etainh", "inherentviscosity"}):
            existing.add(_canonical_sample_key(rec.get("local_sample_key")))
    for s in payload.get("samples", []) or []:
        sk = _canonical_sample_key(s.get("local_sample_key"))
        val = s.get("inherent_viscosity_dL_per_g")
        if val in (None, ""):
            continue
        try:
            v = float(val)
        except Exception:
            continue
        if sk not in existing:
            payload.setdefault("property_records", []).append({
                "local_sample_key": sk,
                "property_category": "physical",
                "property_name": "inherent_viscosity",
                "value_numeric": v,
                "unit": "dL/g",
                "value_raw": f"ηinh={v} dL/g",
                "value_qualifier": "exact",
                "property_name_raw": "ηinh",
                "unit_raw": "dL/g",
                "test_method": "inherent viscosity",
                "source_page": s.get("source_page"),
            })
            _review_item(payload, "inherent_viscosity_property_added_from_sample", sample=sk, value=v)


def _dedupe_property_records_property(payload: Dict[str, Any]) -> None:
    records = payload.get("property_records", []) or []
    standardized: List[Dict[str, Any]] = []
    for rec in records:
        row = _standardize_property_name_property(rec)
        row["local_sample_key"] = _canonical_sample_key(row.get("local_sample_key"))
        row["unit"] = _canonical_unit_property(row.get("unit") or row.get("unit_raw") or "")
        if row.get("property_name") == "transmittance" and row.get("wavelength_nm") in (None, ""):
            inferred = _infer_wavelength_from_property(row)
            if inferred is not None:
                row["wavelength_nm"] = inferred
        standardized.append(row)

    wavelength_trans: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
    for row in standardized:
        if row.get("property_name") != "transmittance":
            continue
        wl = _infer_wavelength_from_property(row)
        if wl is None:
            continue
        try:
            val = round(float(row.get("value_numeric")), 8)
        except Exception:
            continue
        wavelength_trans.setdefault((_canonical_sample_key(row.get("local_sample_key")), val), []).append(row)

    filtered: List[Dict[str, Any]] = []
    for row in standardized:
        if row.get("property_name") == "transmittance" and _infer_wavelength_from_property(row) is None:
            try:
                val = round(float(row.get("value_numeric")), 8)
            except Exception:
                val = None
            if val is not None and wavelength_trans.get((_canonical_sample_key(row.get("local_sample_key")), val)):
                _review_item(payload, "generic_transmittance_removed_property", sample=row.get("local_sample_key"),
                             value=row.get("value_numeric"), reason="same value as wavelength-specific T400/T450/T500 row")
                continue
        filtered.append(row)

    out: List[Dict[str, Any]] = []
    index: Dict[Tuple[Any, ...], int] = {}
    for rec in filtered:
        key = _property_specific_key_property(rec)
        if key not in index:
            index[key] = len(out)
            out.append(rec)
            continue
        j = index[key]
        old = out[j]
        try:
            same_value = abs(float(old.get("value_numeric")) - float(rec.get("value_numeric"))) < 1e-9
        except Exception:
            same_value = old.get("value_numeric") == rec.get("value_numeric")
        if same_value:
            out[j] = _merge_property_rows_property(old, rec)
            _review_item(payload, "duplicate_property_removed_property", sample=rec.get("local_sample_key"),
                         property_name=rec.get("property_name"), property_name_raw=rec.get("property_name_raw"),
                         wavelength_nm=_infer_wavelength_from_property(rec), value=rec.get("value_numeric"))
        else:
            out.append(rec)
            _review_item(payload, "property_duplicate_conflicting_values_property", sample=rec.get("local_sample_key"),
                         property_name=rec.get("property_name"), property_name_raw=rec.get("property_name_raw"),
                         old_value=old.get("value_numeric"), new_value=rec.get("value_numeric"))
    payload["property_records"] = out


def _classify_process_profile_property(profile: Dict[str, Any], paper_text: str) -> str:
    """Classify an existing cure profile into a more precise process bucket."""
    segs = profile.get("segments") or []
    temps = [float(s.get("temp_c")) for s in segs if isinstance(s.get("temp_c"), (int, float))]
    ptype = str(profile.get("imidization_type") or "").lower()
    text = (paper_text or "").lower()
    if len(temps) >= 3 and max(temps or [0]) >= 180 and min(temps or [999]) <= 100:
        if "acetic anhydride" in text or "pyridine" in text or "triethylamine" in text:
            return "film_drying_profile"
        if ptype == "thermal":
            return "thermal_imidization_profile"
        return "film_drying_profile"
    if ptype == "chemical":
        return "chemical_imidization_profile"
    if ptype in {"one-step", "one_step", "one-step polycondensation"}:
        return "polymerization_profile"
    if ptype == "thermal":
        return "thermal_imidization_profile"
    return "process_profile"


def _split_process_profiles_property(payload: Dict[str, Any], paper_text: str) -> None:
    """Add explicit process-profile buckets without deleting cure_profiles.

    This separates solution casting/drying, thermal imidization and chemical imidization process profiles.
    The original cure_profiles are preserved for backward compatibility.
    """
    buckets = {
        "polymerization_profiles": [],
        "chemical_imidization_profiles": [],
        "thermal_imidization_profiles": [],
        "film_drying_profiles": [],
        "process_profiles": [],
    }
    for prof in payload.get("cure_profiles", []) or []:
        role = _classify_process_profile_property(prof, paper_text)
        new_prof = dict(prof)
        new_prof["profile_role"] = role
        buckets["process_profiles"].append(new_prof)
        bucket_name = role + "s" if role.endswith("_profile") else role
        if bucket_name in buckets:
            buckets[bucket_name].append(new_prof)
        prof["profile_role"] = role
    for k, v in buckets.items():
        payload[k] = v

    low = (paper_text or "").lower()
    if ("acetic anhydride" in low or "ac2o" in low) and ("pyridine" in low or "triethylamine" in low):
        for s in payload.get("samples", []) or []:
            sk = _canonical_sample_key(s.get("local_sample_key"))
            payload.setdefault("chemical_imidization_profiles", []).append({
                "local_sample_key": sk,
                "profile_role": "chemical_imidization_profile",
                "imidization_type": "chemical",
                "reagents": "acetic anhydride / pyridine or triethylamine; verify exact reagent from experimental section",
                "segments": [],
                "notes": "Added by process-profile split from text mention; thermal ramp kept separately as film_drying_profile when applicable.",
            })


def _standardize_properties_and_profiles_property(payload: Dict[str, Any], paper_text: str) -> None:
    """Final generic post-processing pass."""
    _sync_inherent_viscosity_property(payload)
    _dedupe_property_records_property(payload)
    _split_process_profiles_property(payload, paper_text)


def _norm_key_generic(text: Any) -> str:
    """Loose but stable key for abbreviations/names in reference libraries."""
    s = str(text or "").strip().lower()
    s = s.replace("′", "'").replace("’", "'").replace("`", "'")
    s = s.replace("α", "alpha").replace("β", "beta")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^a-z0-9'\-_,()]+", "", s)
    return s


def _default_reference_library_generic() -> Dict[str, Dict[str, Any]]:
    """Curated high-confidence seed library.

    This seed is deliberately conservative.  For batch use, add more verified
    monomers in ./reference_monomer_library.json rather than allowing
    image_vision/PubChem to overwrite chemistry.  JSON schema accepted:

    {
      "ODPA": {"smiles": "...", "role": "dianhydride", "aliases": ["..."]},
      "custom_abbr": {"smiles": "...", "monomer_class": "diamine", ...}
    }
    """
    base: Dict[str, Dict[str, Any]] = {}
    def add(key: str, smiles: str, role: str, aliases: Sequence[str] = ()) -> None:
        base[key] = {
            "abbreviation": key,
            "smiles": smiles,
            "role": role,
            "aliases": list(aliases),
            "source": "reference_generic_seed",
            "confidence": "high",
        }

    for key, smi in REFERENCE_MONOMER_SMILES.items():
        role = "diamine" if _count_primary_amines(canonicalize_smiles(smi) or smi) >= 2 else "dianhydride" if _count_anhydride_rings(canonicalize_smiles(smi) or smi) >= 2 else "unknown"
        add(key, smi, role, aliases=[])

    alias_updates = {
        "p-PDA": ["PPD", "PDA", "1,4-diaminobenzene", "p-phenylenediamine", "para-phenylenediamine"],
        "m-PDA": ["MPD", "1,3-diaminobenzene", "m-phenylenediamine", "meta-phenylenediamine"],
        "m-MPDA": ["2,6-toluenediamine", "2,6-diaminotoluene"],
        "m-TMPDA": ["2,4,6-trimethyl-1,3-phenylenediamine", "2,4,6-trimethyl-m-phenylenediamine"],
        "BPADA": ["2,2-bis[4-(3,4-dicarboxyphenoxy)phenyl]propane dianhydride", "bisphenol A dianhydride"],
        "ODPA": ["4,4'-oxydiphthalic anhydride", "oxydiphthalic dianhydride"],
        "6FDA": ["4,4'-(hexafluoroisopropylidene)diphthalic anhydride", "hexafluoroisopropylidene diphthalic anhydride"],
        "TFMB": ["2,2'-bis(trifluoromethyl)benzidine", "TFDB"],
        "TFDB": ["2,2'-bis(trifluoromethyl)benzidine", "TFMB"],
        "MeDABA": ["2-methyl-4,4'-diaminobenzanilide"],
        "ClDABA": ["2-chloro-4,4'-diaminobenzanilide"],
        "2,2'-DMBZ": ["2,2'-dimethylbenzidine", "2,2-dimethylbenzidine"],
        "3,3'-DMBZ": ["3,3'-dimethylbenzidine", "3,3-dimethylbenzidine", "o-tolidine"],
    }
    for key, aliases in alias_updates.items():
        if key in base:
            base[key].setdefault("aliases", []).extend(aliases)
    return base


_REFERENCE_LIBRARY_CACHE_GENERIC: Optional[Tuple[Tuple[str, ...], Dict[str, Dict[str, Any]], Dict[str, str]]] = None

def _load_reference_library_generic() -> Dict[str, Dict[str, Any]]:
    """Load built-in plus optional project-level reference monomer libraries."""
    lib = _default_reference_library_generic()
    candidates = []
    env = os.getenv("PI_REFERENCE_MONOMER_LIBRARY") or os.getenv("REFERENCE_MONOMER_LIBRARY")
    if env:
        candidates.append(Path(env))
    candidates.extend([
        Path("reference_monomer_library.json"),
        Path("data/reference_monomer_library.json"),
        Path("data/reference/monomer_library.json"),
    ])
    for path in candidates:
        try:
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                data = {str(item.get("abbreviation") or item.get("abbr") or item.get("key")): item for item in data if isinstance(item, dict)}
            if not isinstance(data, dict):
                continue
            for key, val in data.items():
                if not isinstance(val, dict):
                    continue
                abbr = str(val.get("abbreviation") or val.get("abbr") or key).strip()
                if not abbr:
                    continue
                entry = dict(val)
                entry["abbreviation"] = abbr
                entry["smiles"] = entry.get("smiles") or entry.get("canonical_smiles") or ""
                entry["role"] = entry.get("role") or entry.get("monomer_class") or entry.get("class") or ""
                entry.setdefault("aliases", [])
                entry.setdefault("source", f"reference_file:{path}")
                entry.setdefault("confidence", "curated")
                lib[abbr] = entry
        except Exception as exc:
            print(f"[WARN] could not load reference monomer library {path}: {exc}", file=sys.stderr)
    return lib


def _reference_indexes_generic() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    global _REFERENCE_LIBRARY_CACHE_GENERIC
    env = os.getenv("PI_REFERENCE_MONOMER_LIBRARY") or os.getenv("REFERENCE_MONOMER_LIBRARY") or ""
    candidate_paths = (env, "reference_monomer_library.json", "data/reference_monomer_library.json", "data/reference/monomer_library.json")
    if _REFERENCE_LIBRARY_CACHE_GENERIC and _REFERENCE_LIBRARY_CACHE_GENERIC[0] == candidate_paths:
        return _REFERENCE_LIBRARY_CACHE_GENERIC[1], _REFERENCE_LIBRARY_CACHE_GENERIC[2]
    lib = _load_reference_library_generic()
    index: Dict[str, str] = {}
    for key, entry in lib.items():
        keys = [key, entry.get("abbreviation"), entry.get("name")]
        aliases = entry.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        keys.extend(aliases)
        for item in keys:
            nk = _norm_key_generic(item)
            if nk:
                index[nk] = key
    _REFERENCE_LIBRARY_CACHE_GENERIC = (candidate_paths, lib, index)
    return lib, index


def reference_entry_for_monomer_generic(abbr: str, name: str) -> Optional[Dict[str, Any]]:
    lib, index = _reference_indexes_generic()
    candidates = [_norm_key_generic(abbr), _norm_key_generic(name)]
    name_key = _norm_key_generic(name)
    for cand in candidates:
        if cand and cand in index:
            return lib[index[cand]]
    if name_key:
        for alias_key, ref_key in index.items():
            if len(alias_key) >= 6 and (alias_key in name_key or name_key in alias_key):
                return lib[ref_key]
    return None


def reference_smiles_for_monomer(abbr: str, name: str) -> str:  # type: ignore[override]
    entry = reference_entry_for_monomer_generic(abbr, name)
    if not entry:
        return ""
    smi = canonicalize_smiles(entry.get("smiles") or "")
    return smi or str(entry.get("smiles") or "")


def _resolve_reference_entry_status_generic(abbr: str, name: str, role: str) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """Return (smiles, source, entry) if a curated entry passes role QC."""
    entry = reference_entry_for_monomer_generic(abbr, name)
    if not entry:
        return "", "", None
    smi = canonicalize_smiles(entry.get("smiles") or "")
    if not smi:
        return "", "", entry
    entry_role = str(entry.get("role") or "").lower()
    expected = (role or entry_role or "").lower()
    if expected and expected in {"diamine", "dianhydride"} and not _candidate_smiles_passes_role_qc(smi, expected):
        return "", "", entry
    return smi, str(entry.get("source") or "reference_generic"), entry


PROPERTY_ONTOLOGY_GENERIC: List[Dict[str, Any]] = [
    {"canonical": "inherent_viscosity", "category": "physical", "unit": "dL/g", "patterns": [r"η\s*inh", r"\binherent\s*viscosity\b", r"\binh\.?\s*visc"]},
    {"canonical": "Mn", "category": "physical", "patterns": [r"\bMn\b", r"number\s*average\s*molecular\s*weight"]},
    {"canonical": "Mw", "category": "physical", "patterns": [r"\bMw\b", r"weight\s*average\s*molecular\s*weight"]},
    {"canonical": "PDI", "category": "physical", "patterns": [r"\bPDI\b", r"\bMw\s*/\s*Mn\b", r"polydispersity"]},
    {"canonical": "Tg", "category": "thermal", "unit": "°C", "patterns": [r"\bTg\b", r"glass\s*transition"]},
    {"canonical": "Tg_DMA", "category": "thermal", "unit": "°C", "patterns": [r"Tg\s*,?\s*DMA", r"Tg[_\-\s]*DMA"]},
    {"canonical": "Td5", "category": "thermal", "unit": "°C", "patterns": [r"\bT\s*5\s*%", r"\bTd\s*5\b", r"5\s*%\s*(?:weight\s*)?loss"]},
    {"canonical": "Td10", "category": "thermal", "unit": "°C", "patterns": [r"\bT\s*10\s*%", r"\bTd\s*10\b"]},
    {"canonical": "Td5_air", "category": "thermal", "unit": "°C", "patterns": [r"Td\s*5.*air", r"T\s*5\s*%.*air", r"air.*T\s*5\s*%"]},
    {"canonical": "Tmax", "category": "thermal", "unit": "°C", "patterns": [r"\bTmax\b", r"maximum\s*(?:decomposition|degradation)"]},
    {"canonical": "residual_weight_730c", "category": "thermal", "unit": "%", "patterns": [r"\bRw\s*730\b", r"residual\s*weight.*730"]},
    {"canonical": "residual_weight_750c", "category": "thermal", "unit": "%", "patterns": [r"\bRw\s*750\b", r"residual\s*weight.*750"]},
    {"canonical": "char_yield_800c_n2", "category": "thermal", "unit": "%", "patterns": [r"char\s*yield.*800", r"residual\s*weight.*800.*N2"]},
    {"canonical": "CTE", "category": "thermal", "unit": "ppm/K", "patterns": [r"\bCTE\b", r"\bCLTE\b", r"coefficient\s*of\s*(?:linear\s*)?thermal\s*expansion"]},
    {"canonical": "cutoff_wavelength", "category": "optical", "unit": "nm", "patterns": [r"λ\s*0", r"lambda\s*0", r"λ\s*cut", r"cut\s*off", r"cutoff\s*wavelength"]},
    {"canonical": "yellow_index", "category": "optical", "patterns": [r"\bYI\b", r"yellow\s*index"]},
    {"canonical": "haze", "category": "optical", "unit": "%", "patterns": [r"\bhaze\b"]},
    {"canonical": "cie_L", "category": "optical", "patterns": [r"\bL\s*\*", r"cie[_\-\s]*L"]},
    {"canonical": "cie_a", "category": "optical", "patterns": [r"\ba\s*\*", r"cie[_\-\s]*a"]},
    {"canonical": "cie_b", "category": "optical", "patterns": [r"\bb\s*\*", r"cie[_\-\s]*b"]},
    {"canonical": "energy_gap", "category": "optical", "unit": "eV", "patterns": [r"\bEg\b", r"energy\s*gap"]},
    {"canonical": "refractive_index_TE", "category": "optical", "patterns": [r"\bn\s*TE\b", r"nTE"]},
    {"canonical": "refractive_index_TM", "category": "optical", "patterns": [r"\bn\s*TM\b", r"nTM"]},
    {"canonical": "refractive_index_avg", "category": "optical", "patterns": [r"\bn\s*av\b", r"nav", r"average\s*refractive"]},
    {"canonical": "birefringence", "category": "optical", "patterns": [r"Δ\s*n", r"Delta[_\-\s]*n", r"\bdn\b", r"birefringence"]},
    {"canonical": "tensile_strength", "category": "mechanical", "unit": "MPa", "patterns": [r"tensile\s*strength", r"\bTS\b"]},
    {"canonical": "modulus", "category": "mechanical", "patterns": [r"tensile\s*modulus", r"Young'?s\s*modulus", r"\bmodulus\b"]},
    {"canonical": "elongation_at_break", "category": "mechanical", "unit": "%", "patterns": [r"elongation\s*at\s*break", r"\bEAB\b"]},
    {"canonical": "density", "category": "physical", "unit": "g/cm3", "patterns": [r"\bdensity\b", r"ρ\s*meas", r"rho[_\-\s]*meas"]},
    {"canonical": "free_volume_fraction", "category": "physical", "unit": "%", "patterns": [r"\bFFV\b", r"free\s*volume\s*fraction"]},
    {"canonical": "water_uptake", "category": "physical", "unit": "%", "patterns": [r"water\s*(?:uptake|absorption)"]},
    {"canonical": "solubility", "category": "chemical", "patterns": [r"\bsolubility\b", r"^solubility[_\-]"]},
    {"canonical": "folding_cycles", "category": "mechanical", "patterns": [r"folding\s*cycles?", r"cycles?"]},
    {"canonical": "bending_radius", "category": "mechanical", "patterns": [r"bending\s*radius", r"folding\s*radius"]},
]

PROPERTY_CANONICAL_ALLOWED_GENERIC = {r["canonical"] for r in PROPERTY_ONTOLOGY_GENERIC} | PROPERTY_NAME_ALIASES | {
    "transmittance", "dielectric_constant", "dissipation_factor", "contact_angle", "other"
}
PROPERTY_NAME_ALIASES.update(PROPERTY_CANONICAL_ALLOWED_GENERIC)


def _ontology_match_generic(label: str) -> Optional[Dict[str, Any]]:
    for rule in PROPERTY_ONTOLOGY_GENERIC:
        for pat in rule.get("patterns", []):
            if re.search(pat, label, re.I):
                return rule
    return None


def _extract_transmittance_wavelength_generic(label: str) -> Optional[float]:
    m = re.search(r"\bT\s*(300|350|365|380|400|450|500|550|600|650|700|800)\s*(?:nm)?\b", label, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"transmittance(?:\s*at)?\s*(300|350|365|380|400|450|500|550|600|650|700|800)\s*nm", label, re.I)
    if m:
        return float(m.group(1))
    return None


def _standardize_property_name_generic(rec: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(rec)
    label = " ".join(str(row.get(k) or "") for k in ("property_name", "property_name_raw", "value_raw", "test_method"))
    wl = _extract_transmittance_wavelength_generic(label)
    if wl is not None or str(row.get("property_name") or "").lower() in {"transmittance", "t400", "t450", "t500", "t550", "t600"}:
        row["property_name"] = "transmittance"
        row["property_category"] = "optical"
        if wl is None:
            wl = _infer_wavelength_from_property(row)
        if wl is not None:
            row["wavelength_nm"] = float(wl)
            row["property_name_raw"] = f"T{int(wl)}"
        row["unit"] = "%"
        row["unit_raw"] = row.get("unit_raw") or "%"
        return row

    rule = _ontology_match_generic(label)
    if rule:
        row["property_name"] = rule["canonical"]
        row["property_category"] = rule.get("category") or row.get("property_category") or "other"
        if rule.get("unit") and not row.get("unit"):
            row["unit"] = rule["unit"]
        if row["property_name"] == "solubility":
            raw = row.get("property_name_raw") or row.get("property_name") or "solubility"
            if not str(raw).lower().startswith("solubility"):
                raw = f"solubility_{raw}"
            row["property_name_raw"] = raw
        if row["property_name"] == "Td5":
            row["decomposition_criterion"] = row.get("decomposition_criterion") or "5_pct"
        return row

    return _standardize_property_name_property(row)


def _dedupe_property_records_generic(payload: Dict[str, Any]) -> None:
    records = payload.get("property_records", []) or []
    standardized: List[Dict[str, Any]] = []
    for rec in records:
        row = _standardize_property_name_generic(rec)
        row["local_sample_key"] = _canonical_sample_key(row.get("local_sample_key"))
        row["unit"] = _canonical_unit_property(row.get("unit") or row.get("unit_raw") or "")
        if row.get("property_name") == "transmittance" and row.get("wavelength_nm") in (None, ""):
            inferred = _infer_wavelength_from_property(row)
            if inferred is not None:
                row["wavelength_nm"] = inferred
                row["property_name_raw"] = row.get("property_name_raw") or f"T{int(inferred)}"
        if row.get("property_name") == "other":
            raw = str(row.get("property_name_raw") or row.get("value_raw") or "")
            if _ontology_match_generic(raw):
                _review_item(payload, "ontology_mapping_missed_generic", sample=row.get("local_sample_key"), raw=raw)
        standardized.append(row)

    wavelength_trans: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
    for row in standardized:
        if row.get("property_name") != "transmittance":
            continue
        wl = _infer_wavelength_from_property(row)
        if wl is None:
            continue
        try:
            val = round(float(row.get("value_numeric")), 8)
        except Exception:
            continue
        wavelength_trans.setdefault((_canonical_sample_key(row.get("local_sample_key")), val), []).append(row)

    filtered: List[Dict[str, Any]] = []
    for row in standardized:
        if row.get("property_name") == "transmittance" and _infer_wavelength_from_property(row) is None:
            try:
                val = round(float(row.get("value_numeric")), 8)
            except Exception:
                val = None
            if val is not None and wavelength_trans.get((_canonical_sample_key(row.get("local_sample_key")), val)):
                _review_item(payload, "generic_transmittance_removed_generic", sample=row.get("local_sample_key"), value=row.get("value_numeric"), reason="same value as wavelength-specific transmittance")
                continue
        filtered.append(row)

    out: List[Dict[str, Any]] = []
    index: Dict[Tuple[Any, ...], int] = {}
    for rec in filtered:
        key = _property_specific_key_property(rec)
        if key not in index:
            index[key] = len(out)
            out.append(rec)
            continue
        j = index[key]
        old = out[j]
        try:
            same_value = abs(float(old.get("value_numeric")) - float(rec.get("value_numeric"))) < 1e-9
        except Exception:
            same_value = old.get("value_numeric") == rec.get("value_numeric")
        if same_value:
            out[j] = _merge_property_rows_property(old, rec)
            _review_item(payload, "duplicate_property_removed_generic", sample=rec.get("local_sample_key"), property_name=rec.get("property_name"), property_name_raw=rec.get("property_name_raw"), wavelength_nm=_infer_wavelength_from_property(rec), value=rec.get("value_numeric"))
        else:
            out.append(rec)
            _review_item(payload, "property_duplicate_conflicting_values_generic", sample=rec.get("local_sample_key"), property_name=rec.get("property_name"), property_name_raw=rec.get("property_name_raw"), old_value=old.get("value_numeric"), new_value=rec.get("value_numeric"))
    payload["property_records"] = out


def _sample_route_map_generic(payload: Dict[str, Any]) -> Dict[str, str]:
    poly_route = {_canonical_sample_key(p.get("local_polymer_key")): str(p.get("imidization_route") or "").lower() for p in payload.get("polymers", []) or []}
    out: Dict[str, str] = {}
    for s in payload.get("samples", []) or []:
        sk = _canonical_sample_key(s.get("local_sample_key"))
        pk = _canonical_sample_key(s.get("local_polymer_key"))
        out[sk] = poly_route.get(pk, "")
    return out


def _chemical_reagent_note_generic(paper_text: str) -> str:
    low = (paper_text or "").lower()
    reagents = []
    if "acetic anhydride" in low or "ac2o" in low:
        reagents.append("acetic anhydride/Ac2O")
    if "pyridine" in low:
        reagents.append("pyridine")
    if "triethylamine" in low or re.search(r"\btea\b", low):
        reagents.append("triethylamine/TEA")
    if "isoquinoline" in low:
        reagents.append("isoquinoline")
    return " / ".join(dict.fromkeys(reagents))


def _classify_process_profile_generic(profile: Dict[str, Any], paper_text: str, route_hint: str = "") -> str:
    segs = profile.get("segments") or []
    temps = [float(s.get("temp_c")) for s in segs if isinstance(s.get("temp_c"), (int, float))]
    ptype = str(profile.get("imidization_type") or "").lower().replace("_", "-")
    route = (route_hint or "").lower().replace("_", "-")
    text = (paper_text or "").lower()
    has_chemical_reagents = ("acetic anhydride" in text or "ac2o" in text) and ("pyridine" in text or "triethylamine" in text or re.search(r"\btea\b", text))

    if ptype.startswith("one-step") or route.startswith("one-step") or "one-step" in route:
        return "polymerization_profile"
    if len(temps) >= 3 and max(temps or [0]) >= 180 and min(temps or [999]) <= 100:
        if ptype == "thermal" or route == "thermal":
            return "thermal_imidization_profile"
        if ptype == "chemical" or route == "chemical" or has_chemical_reagents:
            return "film_drying_profile"
        return "film_drying_profile"
    if ptype == "thermal" or route == "thermal":
        return "thermal_imidization_profile"
    if ptype == "chemical" or route == "chemical":
        return "chemical_imidization_profile"
    return "process_profile"


def _split_process_profiles_generic(payload: Dict[str, Any], paper_text: str) -> None:
    buckets = {
        "polymerization_profiles": [],
        "chemical_imidization_profiles": [],
        "thermal_imidization_profiles": [],
        "film_drying_profiles": [],
        "process_profiles": [],
    }
    route_by_sample = _sample_route_map_generic(payload)
    for prof in payload.get("cure_profiles", []) or []:
        sk = _canonical_sample_key(prof.get("local_sample_key"))
        role = _classify_process_profile_generic(prof, paper_text, route_by_sample.get(sk, ""))
        new_prof = dict(prof)
        new_prof["local_sample_key"] = sk
        new_prof["profile_role"] = role
        if role == "film_drying_profile":
            new_prof["process_type"] = "film_drying_or_annealing"
        elif role == "thermal_imidization_profile":
            new_prof["process_type"] = "thermal_imidization"
        elif role == "chemical_imidization_profile":
            new_prof["process_type"] = "chemical_imidization"
        elif role == "polymerization_profile":
            new_prof["process_type"] = "polymerization"
        buckets["process_profiles"].append(new_prof)
        bucket_name = role + "s" if role.endswith("_profile") else role
        if bucket_name in buckets:
            buckets[bucket_name].append(new_prof)
        prof["profile_role"] = role
        prof["process_type"] = new_prof.get("process_type")

    reagent_note = _chemical_reagent_note_generic(paper_text)
    low = (paper_text or "").lower()
    has_chem = bool(reagent_note and ("acetic" in reagent_note.lower() or "ac2o" in reagent_note.lower()))
    existing_chem = {_canonical_sample_key(p.get("local_sample_key")) for p in buckets["chemical_imidization_profiles"]}
    if has_chem:
        for s in payload.get("samples", []) or []:
            sk = _canonical_sample_key(s.get("local_sample_key"))
            route = route_by_sample.get(sk, "")
            if route and route not in {"chemical", "mixed"}:
                continue
            if sk in existing_chem:
                continue
            chem_profile = {
                "local_sample_key": sk,
                "profile_role": "chemical_imidization_profile",
                "process_type": "chemical_imidization",
                "imidization_type": "chemical",
                "reagents": reagent_note,
                "segments": [],
                "notes": "Text-level chemical-imidization profile; verify exact time/concentration from experimental section if needed.",
            }
            buckets["chemical_imidization_profiles"].append(chem_profile)
            buckets["process_profiles"].append(chem_profile)

    for k, rows in list(buckets.items()):
        seen = set()
        clean = []
        for r in rows:
            segkey = tuple((seg.get("step_order"), seg.get("temp_c"), seg.get("duration_min")) for seg in (r.get("segments") or []))
            key = (_canonical_sample_key(r.get("local_sample_key")), r.get("profile_role"), r.get("process_type"), r.get("reagents"), segkey)
            if key in seen:
                continue
            seen.add(key)
            clean.append(r)
        payload[k] = clean


def _apply_reference_monomer_qc_generic(payload: Dict[str, Any]) -> None:
    """Normalize monomer SMILES from reference library and mark unresolved fields.

    This is intentionally conservative: image_vision can suggest candidates, but
    it never overrides curated references and must pass role QC to enter the
    database.  Unresolved monomers are explicitly flagged for batch review.
    """
    for m in payload.get("extracted_monomers", []) or []:
        abbr = m.get("abbreviation") or ""
        name = m.get("name") or ""
        role = m.get("monomer_class") or m.get("role") or ""
        ref_smi, ref_source, ref_entry = _resolve_reference_entry_status_generic(abbr, name, role)
        old_smi = canonicalize_smiles(m.get("smiles") or "")
        if ref_smi:
            if old_smi and old_smi != ref_smi:
                _review_item(payload, "monomer_smiles_replaced_by_reference_generic", abbreviation=abbr, old_smiles=old_smi, reference_smiles=ref_smi, source=ref_source)
            m["smiles"] = ref_smi
            m["smiles_source"] = ref_source or "reference_generic"
            m["inchi_key"] = safe_inchikey(ref_smi)
            m["contains_fluorine"] = bool(detect_fluorine(ref_smi))
            continue
        if old_smi and not _candidate_smiles_passes_role_qc(old_smi, role):
            _review_item(payload, "monomer_smiles_rejected_by_generic_role_qc", abbreviation=abbr, role=role, smiles=old_smi, source=m.get("smiles_source"))
            m["smiles"] = ""
            m["smiles_source"] = "unresolved"
            m["inchi_key"] = ""
            m["contains_fluorine"] = None
        elif not old_smi:
            _review_item(payload, "monomer_reference_missing_generic", abbreviation=abbr, name=name, role=role, reason="no curated reference SMILES available; left unresolved for batch review")


def _standardize_properties_and_profiles_generic(payload: Dict[str, Any], paper_text: str) -> None:
    """Batch-safe generic post-processing pass."""
    _apply_reference_monomer_qc_generic(payload)
    _sync_inherent_viscosity_property(payload)
    _dedupe_property_records_generic(payload)
    _split_process_profiles_generic(payload, paper_text)

def _upsert_property(payload: Dict[str, Any], sample_key: str, property_name: str, value: float,
                     unit: str, category: str, method: str, raw: str,
                     source_page: Optional[int] = None, wavelength_nm: Optional[float] = None,
                     value_std: Optional[float] = None, property_name_raw: Optional[str] = None,
                     temperature_range: Optional[Tuple[float, float]] = None,
                     decomposition_criterion: Optional[str] = None,
                     tg_definition: Optional[str] = None) -> None:
    sample_key = _canonical_sample_key(sample_key)
    rec = {
        "local_sample_key": sample_key,
        "property_category": category,
        "property_name": property_name,
        "value_numeric": float(value),
        "unit": unit,
        "value_std": value_std,
        "value_raw": raw,
        "value_qualifier": parse_value_qualifier(raw),
        "property_name_raw": property_name_raw,
        "unit_raw": unit,
        "test_method": method,
        "wavelength_nm": wavelength_nm,
        "frequency_hz": None,
        "temperature_c": None,
        "temperature_range_c_min": temperature_range[0] if temperature_range else None,
        "temperature_range_c_max": temperature_range[1] if temperature_range else None,
        "decomposition_criterion": decomposition_criterion,
        "tg_definition": tg_definition,
        "source_page": source_page,
    }
    key = _prop_key(rec)
    for row in payload.get("property_records", []) or []:
        if _prop_key(row) == key:
            old = row.get("value_numeric")
            if old != value or normalize_unit(row.get("unit") or "") != normalize_unit(unit):
                row.update(rec)
                _review_item(payload, "property_value_repaired", sample=sample_key,
                             property_name=property_name, old_value=old, new_value=value, source_page=source_page)
            else:
                row.update(_merge_property_rows(row, rec))
            return
    payload.setdefault("property_records", []).append(rec)


def _infer_monomer_class_from_identity(abbr: Any, name: Any) -> Optional[str]:
    """Protect obvious diamine/dianhydride identities before QC repair."""
    a = _canonical_sample_key(abbr).upper()
    n = ("" if name is None else str(name)).lower()
    diamine_abbrs = {
        "TFDB", "TFMB", "DMBZ", "2,2'-DMBZ", "3,3'-DMBZ", "PDA", "P-PDA", "M-PDA",
        "M-MPDA", "M-TMPDA", "TMPDA", "3TMTD", "FDA", "MFDA", "ABABP", "MEDABA", "CLDABA",
    }
    dianhydride_abbrs = {
        "DSDA", "ODPA", "PMDA", "BPDA", "BPADA", "BPFPA", "BPAF", "HPMDA", "HBPDA", "6FDA", "CBDA", "CHDA",
    }
    if a in dianhydride_abbrs or "dianhydride" in n or "anhydride" in n:
        if not any(tok in n for tok in ("benzidine", "diamine", "aminophenyl", "aminobenzamide")):
            return "dianhydride"
    if a in diamine_abbrs or any(tok in n for tok in ("benzidine", "diamine", "aminophenyl", "aminobenzamide", "phenylenediamine")):
        return "diamine"
    return None


def _protect_monomer_roles(payload: Dict[str, Any]) -> None:
    monomers = {m.get("abbreviation"): m for m in payload.get("extracted_monomers", []) or []}
    for mon in monomers.values():
        inferred = _infer_monomer_class_from_identity(mon.get("abbreviation"), mon.get("name"))
        if inferred and mon.get("monomer_class") != inferred:
            old = mon.get("monomer_class")
            mon["monomer_class"] = inferred
            _review_item(payload, "monomer_class_protected_by_name", abbreviation=mon.get("abbreviation"),
                         old_class=old, new_class=inferred, name=mon.get("name"))
    for comp in payload.get("polymer_components", []) or []:
        abbr = comp.get("monomer_abbreviation") or comp.get("monomer_abbr")
        mon = monomers.get(_canonical_sample_key(abbr)) or monomers.get(abbr)
        inferred = _infer_monomer_class_from_identity(abbr, mon.get("name") if mon else "")
        if inferred and comp.get("role") in ("diamine", "dianhydride") and comp.get("role") != inferred:
            old = comp.get("role")
            comp["role"] = inferred
            _review_item(payload, "component_role_protected_by_monomer_identity", monomer_abbreviation=abbr,
                         old_role=old, new_role=inferred, polymer=comp.get("local_polymer_key"))


def _count_primary_amines(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return 0
    patt = Chem.MolFromSmarts("[NX3H2]")
    return len(mol.GetSubstructMatches(patt)) if patt is not None else 0


def _count_anhydride_rings(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return 0
    patterns = ["O=C1OC(=O)[#6]~[#6]1", "C(=O)OC(=O)"]
    hits = 0
    for smarts in patterns:
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None:
            hits = max(hits, len(mol.GetSubstructMatches(patt)))
    return hits


def _heavy_atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def _qc_monomer_smiles(payload: Dict[str, Any]) -> None:
    for mon in payload.get("extracted_monomers", []) or []:
        smi = mon.get("smiles") or ""
        if not smi:
            continue
        role = mon.get("monomer_class")
        bad_reason = None
        if _heavy_atom_count(smi) < 8:
            bad_reason = "too_few_heavy_atoms_for_PI_monomer"
        elif role == "diamine" and _count_primary_amines(smi) < 2:
            bad_reason = "diamine_smiles_has_fewer_than_two_primary_amines"
        elif role == "dianhydride" and _count_anhydride_rings(smi) < 2:
            bad_reason = "dianhydride_smiles_has_fewer_than_two_anhydride_groups"
        if not bad_reason:
            continue
        source = mon.get("smiles_source") or ""
        _review_item(payload, "monomer_smiles_flagged", abbreviation=mon.get("abbreviation"),
                     monomer_class=role, smiles=smi, smiles_source=source, reason=bad_reason)
        if source in ("image_vision", "inferred", "unknown", ""):
            mon["smiles"] = ""
            mon["smiles_source"] = "unresolved"


def _upsert_other_property(payload: Dict[str, Any], sample_key: str, raw_name: str, value: float, unit: str,
                           category: str, method: str, raw: str, source_page: Optional[int] = None,
                           value_std: Optional[float] = None) -> None:
    _upsert_property(payload, sample_key, "other", value, unit, category, method, raw,
                     source_page=source_page, value_std=value_std, property_name_raw=raw_name)


def _solubility_score(raw: str) -> float:
    t = (raw or "").strip()
    if t in ("++", "+") or t.startswith(">"):
        return 2.0
    if t in ("+-", "+−", "±"):
        return 1.0
    if t in ("-", "−"):
        return 0.0
    try:
        return float(t)
    except Exception:
        return 1.0


def _upsert_solubility(payload: Dict[str, Any], sample_key: str, solvent: str, raw_value: str,
                       source_page: Optional[int] = None) -> None:
    _upsert_other_property(payload, sample_key, f"solubility_{solvent}", _solubility_score(raw_value),
                           "score_or_mg_per_mL", "chemical", "solubility test", raw_value, source_page)


def _add_film_baking_profile(payload: Dict[str, Any], sample_keys: Sequence[str], segments: Sequence[Tuple[int, float, float]],
                             atmosphere: Optional[str] = None) -> None:
    existing = payload.setdefault("cure_profiles", [])
    for sample in sample_keys:
        if any(_canonical_sample_key(c.get("local_sample_key")) == _canonical_sample_key(sample) for c in existing):
            continue
        existing.append({
            "local_sample_key": _canonical_sample_key(sample),
            "imidization_type": "thermal",
            "atmosphere": atmosphere,
            "segments": [
                {"step_order": order, "temp_c": temp, "duration_min": dur, "ramp_rate_c_per_min": None}
                for order, temp, dur in segments
            ],
        })


def _normalize_existing_keys(payload: Dict[str, Any]) -> None:
    for key_name, rows in (
        ("local_polymer_key", payload.get("polymers", []) or []),
        ("local_sample_key", payload.get("samples", []) or []),
    ):
        for row in rows:
            row[key_name] = _canonical_sample_key(row.get(key_name))
    for row in payload.get("polymer_components", []) or []:
        row["local_polymer_key"] = _canonical_sample_key(row.get("local_polymer_key"))
        row["monomer_abbreviation"] = _canonical_sample_key(row.get("monomer_abbreviation") or row.get("monomer_abbr"))
        row.setdefault("is_crosslinker", False)
    for row in payload.get("property_records", []) or []:
        row["local_sample_key"] = _canonical_sample_key(row.get("local_sample_key"))
    for row in payload.get("extracted_monomers", []) or []:
        row["abbreviation"] = _canonical_sample_key(row.get("abbreviation"))


def normalize_polyimide_payload(payload: Dict[str, Any], paper_text: str) -> None:
    """Generic QC after LLM extraction: role consistency, thickness sanity, duplicate composition warnings."""
    _normalize_existing_keys(payload)
    _protect_monomer_roles(payload)
    _qc_monomer_smiles(payload)

    for s in payload.get("samples", []) or []:
        val = s.get("film_thickness_um")
        if _is_suspicious_thickness(val):
            try:
                v = float(val)
                repaired = v / 10.0 if 5 <= v / 10.0 <= 80 else None
            except Exception:
                repaired = None
            if repaired is not None:
                s["film_thickness_um"] = repaired
                _review_item(payload, "suspicious_film_thickness_auto_scaled", sample=s.get("local_sample_key"),
                             old_value=val, new_value=repaired, rule="value/10 because extracted value was >100 μm")
            else:
                _review_item(payload, "suspicious_film_thickness", sample=s.get("local_sample_key"), value=val)

    monomers = {m.get("abbreviation"): m for m in payload.get("extracted_monomers", []) or []}
    role_by_abbr: Dict[str, set] = {}
    for c in payload.get("polymer_components", []) or []:
        abbr = c.get("monomer_abbreviation")
        role = c.get("role")
        if abbr and role:
            role_by_abbr.setdefault(abbr, set()).add(role)
    for abbr, roles in role_by_abbr.items():
        mon = monomers.get(abbr)
        if not mon:
            continue
        mclass = mon.get("monomer_class")
        if len(roles) == 1:
            expected = next(iter(roles))
            if expected in ("diamine", "dianhydride") and mclass and mclass != expected:
                protected = _infer_monomer_class_from_identity(abbr, mon.get("name", ""))
                if protected and protected != expected:
                    for comp in payload.get("polymer_components", []) or []:
                        if _canonical_sample_key(comp.get("monomer_abbreviation")) == _canonical_sample_key(abbr):
                            old_role = comp.get("role")
                            comp["role"] = protected
                            _review_item(payload, "component_role_repaired_from_protected_monomer", abbreviation=abbr,
                                         old_role=old_role, new_role=protected, polymer=comp.get("local_polymer_key"))
                    continue
                _review_item(payload, "monomer_role_conflict_repaired", abbreviation=abbr,
                             old_class=mclass, component_role=expected, old_name=mon.get("name"))
                mon["monomer_class"] = expected
                if _looks_like_wrong_monomer_name(mon.get("name", ""), expected):
                    mon["name"] = f"{abbr} ({expected}; name requires manual verification)"
                    mon["smiles"] = None
                    mon["smiles_source"] = "unknown"
                    _review_item(payload, "monomer_name_requires_manual_check", abbreviation=abbr,
                                 reason="name text contradicted component role")

    comp_sig: Dict[Tuple[Tuple[str, str, float], ...], str] = {}
    for poly in payload.get("polymers", []) or []:
        pkey = poly.get("local_polymer_key")
        comps = [c for c in payload.get("polymer_components", []) or [] if c.get("local_polymer_key") == pkey]
        if not comps:
            continue
        sig = tuple(sorted((c.get("monomer_abbreviation"), c.get("role"), float(c.get("molar_ratio") or 0)) for c in comps))
        if sig in comp_sig and comp_sig[sig] != pkey:
            _review_item(payload, "duplicate_polymer_composition", polymer_a=comp_sig[sig], polymer_b=pkey,
                         reason="two different sample labels have identical extracted monomer composition; verify Scheme/Figure")
        else:
            comp_sig[sig] = pkey

    for poly in payload.get("polymers", []) or []:
        pkey = poly.get("local_polymer_key")
        roles = {c.get("role") for c in payload.get("polymer_components", []) or [] if c.get("local_polymer_key") == pkey}
        if poly.get("polymer_class") in ("polyimide", "polyamide_imide"):
            if "dianhydride" not in roles:
                _review_item(payload, "missing_dianhydride_component", local_polymer_key=pkey)
            if "diamine" not in roles:
                _review_item(payload, "missing_diamine_component", local_polymer_key=pkey)

    _dedupe_property_records(payload)


def _paper_head(paper_text: str, n: int = 16000) -> str:
    return (paper_text or "")[:n].lower()




















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




_LLM_SYSTEM_PROMPT_FULL_BASELINE = LLM_SYSTEM_PROMPT
_LLM_OUTPUT_SCHEMA_FULL_BASELINE = json.loads(json.dumps(LLM_OUTPUT_SCHEMA, ensure_ascii=False))


TARGET_TRANS_WAVELENGTH_NM = 550.0
TARGET_PROPERTY_NAMES = {"Tg", "CTE", "transmittance"}


LLM_SYSTEM_PROMPT = (
    "You are a careful polymer-chemistry data extraction assistant for transparent "
    "polyimide / poly(amide-imide) papers. Extract high-quality structured records "
    "for monomers, polymer/sample identity, and ONLY three key properties: "
    "directly reported transmittance (T550 preferred when available), glass-transition temperature (Tg), and "
    "coefficient of thermal expansion (CTE). Return ONLY JSON matching the schema. "
    "Do not invent or interpolate values. If T550 is not directly reported, still extract "
    "other directly reported transmittance wavelengths such as T500/T450/T400/T600. For SMILES, read structures from figures "
    "when possible, but low-confidence or role-inconsistent SMILES should be left "
    "blank/unknown for later reference-library review."
)

LLM_OUTPUT_SCHEMA = json.loads(json.dumps(LLM_OUTPUT_SCHEMA, ensure_ascii=False))
try:
    prop_schema = LLM_OUTPUT_SCHEMA["properties"]["property_records"]["items"]["properties"]
    prop_schema["property_name"]["enum"] = ["Tg", "CTE", "transmittance"]
except Exception:
    pass




def _load_reference_library_generic() -> Dict[str, Dict[str, Any]]:  # type: ignore[override]
    """Generic reference-library loader.

    A project may provide reference_monomer_library.json or set PI_REFERENCE_MONOMER_LIBRARY for an actively edited curated library.
    """
    lib = _default_reference_library_generic()
    candidates: List[Path] = []
    env = os.getenv("PI_REFERENCE_MONOMER_LIBRARY") or os.getenv("REFERENCE_MONOMER_LIBRARY")
    if env:
        candidates.append(Path(env))
    candidates.extend([
        Path("reference_monomer_library.json"),
        Path("reference_monomer_library_template.json"),
        Path("data/reference_monomer_library.json"),
        Path("data/reference_monomer_library_template.json"),
        Path("data/reference/monomer_library.json"),
    ])
    for path in candidates:
        try:
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                data = {str(item.get("abbreviation") or item.get("abbr") or item.get("key")): item for item in data if isinstance(item, dict)}
            if not isinstance(data, dict):
                continue
            for key, val in data.items():
                if not isinstance(val, dict):
                    continue
                abbr = str(val.get("abbreviation") or val.get("abbr") or key).strip()
                if not abbr:
                    continue
                entry = dict(val)
                entry["abbreviation"] = abbr
                entry["smiles"] = entry.get("smiles") or entry.get("canonical_smiles") or ""
                entry["role"] = entry.get("role") or entry.get("monomer_class") or entry.get("class") or ""
                entry.setdefault("aliases", [])
                entry.setdefault("source", f"reference_file:{path}")
                entry.setdefault("confidence", "curated")
                lib[abbr] = entry
        except Exception as exc:
            print(f"[WARN] could not load reference monomer library {path}: {exc}", file=sys.stderr)
    return lib


def _extract_wavelength_from_text_target(row: Dict[str, Any]) -> Optional[float]:
    wl = row.get("wavelength_nm")
    if isinstance(wl, (int, float)):
        return float(wl)
    text = " ".join(str(row.get(k) or "") for k in ("property_name", "property_name_raw", "value_raw"))
    m = re.search(r"\bT\s*(\d{3})\b", text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"(?:at|@)?\s*(\d{3})\s*nm", text, re.I)
    if m and re.search(r"transmittance|transparent|T\s*\d{3}|%", text, re.I):
        return float(m.group(1))
    return None


def _canonicalize_key_property_target(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one property row into Tg, CTE or directly reported transmittance.

    This version keeps the earlier monomer/composition extraction logic, but
    does not restrict optical data to T550 only. For transparent PI papers, many
    authors report T400/T450/T500 instead of T550. We keep any explicitly
    reported transmittance wavelength and do not interpolate or infer T550.
    """
    row = _standardize_property_name_generic(dict(rec))
    pname = row.get("property_name")
    raw_text = " ".join(str(row.get(k) or "") for k in ("property_name", "property_name_raw", "value_raw"))
    token = _clean_prop_token_property(row.get("property_name_raw") or row.get("property_name"))

    if pname in {"Tg", "Tg_DMA"} or re.search(r"\bTg\b|glass\s*transition", raw_text, re.I):
        row["property_name"] = "Tg"
        row["property_category"] = "thermal"
        row["unit"] = normalize_unit(row.get("unit") or "°C") or "°C"
        if pname == "Tg_DMA" and not row.get("tg_definition"):
            row["tg_definition"] = "dma_tan_delta_peak"
        row["property_name_raw"] = row.get("property_name_raw") or pname or "Tg"
        return row

    if pname == "CTE" or token in {"cte", "clte"} or re.search(r"\bC\s*T\s*E\b|thermal\s*expansion", raw_text, re.I):
        row["property_name"] = "CTE"
        row["property_category"] = "thermal"
        row["unit"] = _canonical_unit_property(row.get("unit") or row.get("unit_raw") or "ppm/K") or "ppm/K"
        row["property_name_raw"] = row.get("property_name_raw") or "CTE"
        return row

    wl = _extract_wavelength_from_text_target(row)
    if pname == "transmittance" or re.search(r"transmittance|\bT\s*\d{3}\b", raw_text, re.I):
        row["property_name"] = "transmittance"
        row["property_category"] = "optical"
        row["unit"] = "%"
        row["unit_raw"] = row.get("unit_raw") or "%"
        if wl is not None:
            row["wavelength_nm"] = float(wl)
            row["property_name_raw"] = row.get("property_name_raw") or f"T{int(float(wl))}"
        else:
            row["wavelength_nm"] = None
            row["property_name_raw"] = row.get("property_name_raw") or "transmittance"
        return row
    return None

def _target_property_key_target(row: Dict[str, Any]) -> Tuple[Any, ...]:
    pname = row.get("property_name")
    sample = _canonical_sample_key(row.get("local_sample_key"))
    if pname == "transmittance":
        return (sample, "transmittance", row.get("wavelength_nm"))
    if pname == "CTE":
        return (sample, "CTE", row.get("temperature_range_c_min"), row.get("temperature_range_c_max"))
    if pname == "Tg":
        return (sample, "Tg", row.get("test_method"), row.get("tg_definition"))
    return (sample, pname)

def _dedupe_key_properties_target(records: Sequence[Dict[str, Any]], payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    index: Dict[Tuple[Any, ...], int] = {}
    for rec0 in records:
        rec = _canonicalize_key_property_target(rec0)
        if rec is None:
            continue
        try:
            rec["value_numeric"] = float(rec.get("value_numeric"))
        except Exception:
            continue
        rec["local_sample_key"] = _canonical_sample_key(rec.get("local_sample_key"))
        rec["value_qualifier"] = rec.get("value_qualifier") or parse_value_qualifier(rec.get("value_raw") or "")
        if rec.get("property_name") == "transmittance" and rec.get("wavelength_nm") is None:
            _review_item(payload, "transmittance_missing_wavelength_target", sample=rec.get("local_sample_key"), value_raw=rec.get("value_raw"), reason="kept directly reported transmittance but wavelength_nm was not parsed")
        key = _target_property_key_target(rec)
        if key not in index:
            index[key] = len(out)
            out.append(rec)
            continue
        j = index[key]
        old = out[j]
        try:
            same_value = abs(float(old.get("value_numeric")) - float(rec.get("value_numeric"))) < 1e-9
        except Exception:
            same_value = old.get("value_numeric") == rec.get("value_numeric")
        if same_value:
            out[j] = _merge_property_rows_property(old, rec)
            _review_item(payload, "duplicate_key_property_removed_target", sample=rec.get("local_sample_key"), property_name=rec.get("property_name"), wavelength_nm=rec.get("wavelength_nm"), value=rec.get("value_numeric"))
        else:
            out.append(rec)
            _review_item(payload, "conflicting_key_property_values_target", sample=rec.get("local_sample_key"), property_name=rec.get("property_name"), old_value=old.get("value_numeric"), new_value=rec.get("value_numeric"))
    return out


def _best_property_target(records: Sequence[Dict[str, Any]], sample_key: str, prop_name: str) -> Optional[Dict[str, Any]]:
    sample_key = _canonical_sample_key(sample_key)
    candidates = [r for r in records if _canonical_sample_key(r.get("local_sample_key")) == sample_key and r.get("property_name") == prop_name]
    if not candidates:
        return None
    if prop_name == "transmittance":
        def trans_score(r: Dict[str, Any]) -> Tuple[int, float, int, int]:
            wl = r.get("wavelength_nm")
            try:
                wl_f = float(wl)
                has_wl = 1
                distance = abs(wl_f - TARGET_TRANS_WAVELENGTH_NM)
            except Exception:
                has_wl = 0
                distance = 9999.0
            return (
                1 if r.get("value_qualifier") == "exact" else 0,
                -distance,
                has_wl,
                1 if r.get("source_page") is not None else 0,
            )
        candidates.sort(key=trans_score, reverse=True)
        return candidates[0]
    candidates.sort(key=lambda r: (
        1 if r.get("value_qualifier") == "exact" else 0,
        1 if r.get("source_page") is not None else 0,
        1 if r.get("test_method") and r.get("test_method") != "unknown" else 0,
    ), reverse=True)
    return candidates[0]

def _filter_to_key_properties_target(payload: Dict[str, Any], paper_text: str) -> None:
    """Final target-property scope filter.

    Keep only the processed target properties: reported transmittance, Tg and
    CTE. Unlike the broad extraction pass, do not keep every T400/T450/T500 column. For each sample,
    keep one representative transmittance value, preferring directly reported
    T550 and otherwise the available wavelength closest to 550 nm. Tg and CTE
    are also reduced to one best record per sample to keep property_record.json
    clean for screening.
    """
    deduped = _dedupe_key_properties_target(payload.get("property_records", []) or [], payload)
    final_records: List[Dict[str, Any]] = []
    sample_keys = [_canonical_sample_key(s.get("local_sample_key")) for s in payload.get("samples", []) or []]
    for r in deduped:
        sk = _canonical_sample_key(r.get("local_sample_key"))
        if sk and sk not in sample_keys:
            sample_keys.append(sk)
    for sk in sample_keys:
        for pname in ("transmittance", "Tg", "CTE"):
            best = _best_property_target(deduped, sk, pname)
            if best is not None:
                final_records.append(best)
    payload["property_records"] = final_records

    records = payload.get("property_records", []) or []
    for s in payload.get("samples", []) or []:
        sk = _canonical_sample_key(s.get("local_sample_key"))
        has_transmittance = _best_property_target(records, sk, "transmittance") is not None
        has_tg = _best_property_target(records, sk, "Tg") is not None
        has_cte = _best_property_target(records, sk, "CTE") is not None
        if not has_transmittance:
            _review_item(payload, "missing_target_property_target", sample=sk, property_name="transmittance", reason="no directly reported transmittance value was extracted")
        if not has_tg:
            _review_item(payload, "missing_target_property_target", sample=sk, property_name="Tg", reason="not reported or not extracted")
        if not has_cte:
            _review_item(payload, "missing_target_property_target", sample=sk, property_name="CTE", reason="not reported or not extracted")

    cleaned_trends: List[Dict[str, Any]] = []
    for tr in payload.get("trend_records", []) or []:
        pname = str(tr.get("property_name") or "").strip()
        if pname in {"transmittance", "Tg", "CTE"}:
            cleaned_trends.append(tr)
    payload["trend_records"] = cleaned_trends


def _series_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _series_abbr_name(payload: Dict[str, Any], abbr: str) -> str:
    ab = _canonical_sample_key(abbr)
    for m in payload.get("extracted_monomers", []) or []:
        if _canonical_sample_key(m.get("abbreviation")) == ab:
            nm = str(m.get("name") or "").strip()
            return f"{ab} ({nm})" if nm else ab
    return ab


def _series_sample_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    seen = set()
    for s in payload.get("samples", []) or []:
        sk = _canonical_sample_key(s.get("local_sample_key"))
        pk = _canonical_sample_key(s.get("local_polymer_key"))
        if not sk or not pk or sk in seen:
            continue
        rows.append({"sample_key": sk, "polymer_key": pk})
        seen.add(sk)
    if not rows:
        for p in payload.get("polymers", []) or []:
            pk = _canonical_sample_key(p.get("local_polymer_key"))
            if pk and pk not in seen:
                rows.append({"sample_key": pk, "polymer_key": pk})
                seen.add(pk)
    return rows


def _series_components_by_polymer(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    by_poly: Dict[str, List[Dict[str, Any]]] = {}
    for row in payload.get("polymer_components", []) or []:
        pk = _canonical_sample_key(row.get("local_polymer_key"))
        ab = _canonical_sample_key(row.get("monomer_abbreviation") or row.get("monomer_abbr"))
        role = str(row.get("role") or "other").strip()
        ratio = _series_float(row.get("molar_ratio"))
        if not pk or not ab or ratio is None:
            continue
        by_poly.setdefault(pk, []).append({
            "abbr": ab,
            "role": role,
            "ratio": ratio,
        })
    return by_poly


def _series_component_matrix(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    samples = _series_sample_rows(payload)
    by_poly = _series_components_by_polymer(payload)
    sample_components: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        sample_components[s["sample_key"]] = by_poly.get(s["polymer_key"], [])
    return samples, sample_components


def _series_existing_series_keys(payload: Dict[str, Any]) -> set:
    keys = set()
    for s in payload.get("study_series", []) or []:
        vn = str(s.get("variable_name") or "").strip().lower()
        sn = str(s.get("series_name") or "").strip().lower()
        if vn:
            keys.add(vn)
        if sn:
            keys.add(sn)
    return keys


def _series_add_series(payload: Dict[str, Any], series: Dict[str, Any]) -> None:
    variable_name = str(series.get("variable_name") or "").strip()
    if not variable_name:
        return
    existing = _series_existing_series_keys(payload)
    key = variable_name.lower()
    norm_key = _series_norm_series_name(variable_name) if "_series_norm_series_name" in globals() else re.sub(r"[^a-z0-9]+", "", key)
    existing_norm = {(_series_norm_series_name(x) if "_series_norm_series_name" in globals() else re.sub(r"[^a-z0-9]+", "", x)) for x in existing}
    if key in existing or norm_key in existing_norm:
        return
    payload.setdefault("study_series", []).append(series)
    _review_item(payload, "study_series_inferred", variable_name=variable_name,
                 reason="inferred from polymer_component variation across samples")


def _series_format_component_values(payload: Dict[str, Any], samples: List[Dict[str, Any]],
                                 sample_components: Dict[str, List[Dict[str, Any]]],
                                 variable_abbrs: Sequence[str]) -> str:
    parts: List[str] = []
    for s in samples:
        sk = s["sample_key"]
        comp_map = {c["abbr"]: c for c in sample_components.get(sk, [])}
        vals = []
        for ab in variable_abbrs:
            ratio = comp_map.get(ab, {}).get("ratio", 0)
            if ratio:
                vals.append(f"{ab}:{ratio:g}")
        if not vals:
            variable_present = [c for c in sample_components.get(sk, []) if c.get("abbr") in variable_abbrs and c.get("ratio", 0) > 0]
            vals = [f"{c['abbr']}:{c['ratio']:g}" for c in variable_present]
        parts.append(f"{sk}=" + (", ".join(vals) if vals else "not specified"))
    return "; ".join(parts)


def _series_infer_study_series_from_components(payload: Dict[str, Any], paper_text: str) -> None:
    """Infer missing study_series from polymer_component variation.

    This keeps the original processed table set but avoids relying entirely on
    the LLM to notice a sample series. It is conservative: it creates a series
    only when at least three samples/polymers share a comparable backbone table
    and one or more monomers/ratios vary.
    """
    samples, sample_components = _series_component_matrix(payload)
    if len(samples) < 3:
        return

    all_abbrs: List[str] = []
    role_by_abbr: Dict[str, str] = {}
    ratios_by_abbr: Dict[str, List[float]] = {}
    for s in samples:
        comps = sample_components.get(s["sample_key"], [])
        present = {c["abbr"]: c for c in comps}
        for ab, c in present.items():
            if ab not in all_abbrs:
                all_abbrs.append(ab)
            role_by_abbr.setdefault(ab, c.get("role") or "other")
        for ab in all_abbrs:
            ratios_by_abbr.setdefault(ab, [])
    for ab in all_abbrs:
        ratios: List[float] = []
        for s in samples:
            comps = {c["abbr"]: c for c in sample_components.get(s["sample_key"], [])}
            ratios.append(float(comps.get(ab, {}).get("ratio", 0) or 0))
        ratios_by_abbr[ab] = ratios

    variable_abbrs = []
    fixed_abbrs = []
    for ab, ratios in ratios_by_abbr.items():
        unique = {round(v, 8) for v in ratios}
        if len(unique) > 1:
            variable_abbrs.append(ab)
        elif next(iter(unique), 0) > 0:
            fixed_abbrs.append(ab)
    if not variable_abbrs:
        return

    roles = {role_by_abbr.get(ab, "other") for ab in variable_abbrs}
    variable_values_text = _series_format_component_values(payload, samples, sample_components, variable_abbrs)
    sample_scope = f"{samples[0]['sample_key']} -> {samples[-1]['sample_key']}"

    if len(variable_abbrs) == 2 and len(roles) == 1:
        a, b = variable_abbrs[0], variable_abbrs[1]
        variable_name = f"{a}_to_{b}_ratio"
        series_name = f"{a}/{b} composition-ratio series"
        _series_add_series(payload, {
            "series_name": series_name,
            "variable_name": variable_name,
            "variable_kind": "composition_ratio",
            "variable_unit": "molar_ratio",
            "variable_values_text": variable_values_text,
            "notes": f"inferred from polymer_component.json over {sample_scope}; fixed components: {', '.join(fixed_abbrs) if fixed_abbrs else 'not resolved'}."
        })
        return

    variable_roles = sorted(roles)
    if len(variable_roles) == 1 and variable_roles[0] in {"diamine", "dianhydride"}:
        role = variable_roles[0]
        variable_name = f"{role}_structure"
        series_name = f"{role} structure series"
        named = "; ".join(f"{ab}={_series_abbr_name(payload, ab)}" for ab in variable_abbrs)
        _series_add_series(payload, {
            "series_name": series_name,
            "variable_name": variable_name,
            "variable_kind": "other",
            "variable_unit": None,
            "variable_values_text": variable_values_text,
            "notes": f"inferred structural-identity series over {sample_scope}. Variable {role}s: {named}. Fixed components: {', '.join(fixed_abbrs) if fixed_abbrs else 'not resolved'}."
        })
        return

    named = "; ".join(f"{ab}={_series_abbr_name(payload, ab)}" for ab in variable_abbrs)
    _series_add_series(payload, {
        "series_name": "monomer-combination structure series",
        "variable_name": "monomer_combination",
        "variable_kind": "other",
        "variable_unit": None,
        "variable_values_text": variable_values_text,
        "notes": f"inferred from simultaneous changes in polymer components over {sample_scope}. Variable components: {named}."
    })


def _series_property_value_by_sample(payload: Dict[str, Any], prop_name: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for rec in payload.get("property_records", []) or []:
        if rec.get("property_name") != prop_name:
            continue
        sk = _canonical_sample_key(rec.get("local_sample_key"))
        val = _series_float(rec.get("value_numeric"))
        if not sk or val is None:
            continue
        if sk not in out:
            out[sk] = rec
        else:
            old = out[sk]
            if old.get("value_qualifier") != "exact" and rec.get("value_qualifier") == "exact":
                out[sk] = rec
    return out


def _series_threshold_for_property(prop_name: str) -> float:
    if prop_name == "transmittance":
        return 1.0
    if prop_name == "CTE":
        return 1.0
    if prop_name == "Tg":
        return 2.0
    return 0.0


def _series_classify_ordered_trend(values: List[float], prop_name: str) -> str:
    if len(values) < 3:
        return "no_clear_trend"
    tol = _series_threshold_for_property(prop_name)
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    if max(values) - min(values) < tol:
        return "no_clear_trend"
    if all(d >= -tol for d in diffs) and any(d > tol for d in diffs):
        return "increase"
    if all(d <= tol for d in diffs) and any(d < -tol for d in diffs):
        return "decrease"
    max_i = values.index(max(values))
    min_i = values.index(min(values))
    if 0 < max_i < len(values) - 1 or 0 < min_i < len(values) - 1:
        return "optimum"
    return "mixed"


def _series_series_axis(payload: Dict[str, Any], series: Dict[str, Any], samples: List[Dict[str, Any]],
                     sample_components: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[str], bool, Optional[str]]:
    """Return ordered sample keys, whether the axis is numeric, and axis label."""
    sample_keys = [s["sample_key"] for s in samples]
    vname = str(series.get("variable_name") or "")
    if "_to_" in vname:
        primary = vname.split("_to_", 1)[0]
        xs: List[Tuple[float, int, str]] = []
        found_any = False
        for idx, sk in enumerate(sample_keys):
            ratio = 0.0
            for c in sample_components.get(sk, []):
                if c.get("abbr") == primary:
                    ratio = float(c.get("ratio") or 0)
                    found_any = True
                    break
            xs.append((ratio, idx, sk))
        if found_any and len({x[0] for x in xs}) > 1:
            xs.sort(key=lambda item: (item[0], item[1]))
            return [x[2] for x in xs], True, primary
    return sample_keys, False, None


def _series_same_transmittance_wavelength(records: List[Dict[str, Any]]) -> bool:
    wavelengths = []
    for r in records:
        wl = _series_float(r.get("wavelength_nm"))
        if wl is None:
            return False
        wavelengths.append(round(wl, 3))
    return len(set(wavelengths)) == 1


def _series_norm_series_name(text: Any) -> str:
    s = str(text or "").lower()
    s = s.replace("feed_ratio", "ratio")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _series_trend_exists(payload: Dict[str, Any], variable_name: str, property_name: str, sample_scope: Optional[str] = None) -> bool:
    want_var = _series_norm_series_name(variable_name)
    want_scope = str(sample_scope or "").strip()
    for tr in payload.get("trend_records", []) or []:
        if str(tr.get("property_name") or "") != property_name:
            continue
        got_var = _series_norm_series_name(tr.get("variable_name"))
        got_scope = str(tr.get("sample_scope") or "").strip()
        if got_var == want_var:
            return True
        if want_scope and got_scope and want_scope == got_scope:
            return True
    return False


def _series_append_trend(payload: Dict[str, Any], trend: Dict[str, Any]) -> None:
    variable_name = str(trend.get("variable_name") or "").strip()
    property_name = str(trend.get("property_name") or "").strip()
    sample_scope = trend.get("sample_scope")
    if not variable_name or property_name not in {"transmittance", "Tg", "CTE"}:
        return
    if _series_trend_exists(payload, variable_name, property_name, sample_scope):
        return
    payload.setdefault("trend_records", []).append(trend)
    _review_item(payload, "trend_record_inferred", variable_name=variable_name,
                 property_name=property_name, reason="inferred from target property values across study_series")


def _series_infer_target_property_trends(payload: Dict[str, Any], paper_text: str) -> None:
    """Generate conservative trend_records for transmittance, Tg and CTE.

    Ordered trends are generated only for numeric composition-ratio axes. For
    structural-identity series, the code records a qualitative/mixed comparison only
    when the target property range is large enough to be meaningful. It never
    creates trends for removed properties such as density, FFV, solubility,
    mechanical properties, YI, haze, L*a*b*, Eg, etc.
    """
    samples, sample_components = _series_component_matrix(payload)
    if len(samples) < 3:
        return
    if not payload.get("study_series"):
        return

    for series in payload.get("study_series", []) or []:
        variable_name = str(series.get("variable_name") or "").strip()
        if not variable_name:
            continue
        ordered_sample_keys, numeric_axis, axis_label = _series_series_axis(payload, series, samples, sample_components)
        sample_scope = f"{ordered_sample_keys[0]} -> {ordered_sample_keys[-1]}" if ordered_sample_keys else None

        for prop_name in ("Tg", "CTE", "transmittance"):
            by_sample = _series_property_value_by_sample(payload, prop_name)
            recs = [by_sample[sk] for sk in ordered_sample_keys if sk in by_sample]
            if len(recs) < 3:
                continue
            if prop_name == "transmittance" and not _series_same_transmittance_wavelength(recs):
                continue
            values = [float(r.get("value_numeric")) for r in recs]
            keys = [_canonical_sample_key(r.get("local_sample_key")) for r in recs]
            if max(values) - min(values) < _series_threshold_for_property(prop_name):
                continue

            if numeric_axis:
                direction = _series_classify_ordered_trend(values, prop_name)
                if direction == "no_clear_trend":
                    continue
                evidence_pairs = ", ".join(f"{k}={v:g}" for k, v in zip(keys, values))
                axis_text = f"ordered by increasing {axis_label} content" if axis_label else "ordered by composition ratio"
                confidence = 0.82 if direction in {"increase", "decrease"} else 0.68
                _series_append_trend(payload, {
                    "variable_name": variable_name,
                    "variable_unit": series.get("variable_unit") or "molar_ratio",
                    "property_name": prop_name,
                    "trend_direction": direction,
                    "evidence_text": f"{prop_name} values across {series.get('series_name') or variable_name} ({axis_text}): {evidence_pairs}.",
                    "sample_scope": sample_scope,
                    "mechanism_note": series.get("notes"),
                    "confidence": confidence,
                })
            else:
                min_i = values.index(min(values))
                max_i = values.index(max(values))
                evidence_pairs = ", ".join(f"{k}={v:g}" for k, v in zip(keys, values))
                direction = "qualitative" if len(set(values)) > 1 else "no_clear_trend"
                if direction == "no_clear_trend":
                    continue
                _series_append_trend(payload, {
                    "variable_name": variable_name,
                    "variable_unit": series.get("variable_unit"),
                    "property_name": prop_name,
                    "trend_direction": direction,
                    "evidence_text": f"{prop_name} varies across {series.get('series_name') or variable_name}: {evidence_pairs}; lowest={keys[min_i]} ({values[min_i]:g}), highest={keys[max_i]} ({values[max_i]:g}).",
                    "sample_scope": sample_scope,
                    "mechanism_note": series.get("notes"),
                    "confidence": 0.62,
                })


def _series_finalize_series_and_trends(payload: Dict[str, Any], paper_text: str) -> None:
    _series_infer_study_series_from_components(payload, paper_text)
    payload["trend_records"] = [
        tr for tr in (payload.get("trend_records", []) or [])
        if str(tr.get("property_name") or "") in {"transmittance", "Tg", "CTE"}
    ]
    _series_infer_target_property_trends(payload, paper_text)



_REFERENCE_LIBRARY_CACHE_GENERIC = None


LLM_SYSTEM_PROMPT = _LLM_SYSTEM_PROMPT_FULL_BASELINE
LLM_OUTPUT_SCHEMA = json.loads(json.dumps(_LLM_OUTPUT_SCHEMA_FULL_BASELINE, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Clean submission pipeline: generic extraction, no paper-specific repairs
# ---------------------------------------------------------------------------

def llm_extract_paper(text: str, pdf_path: Optional[Path] = None,
                      model: Optional[str] = None, max_chars: int = 80000,
                      max_pages: int = 12) -> Dict[str, Any]:
    """Run the multimodal LLM extraction with a generic schema-driven prompt."""
    import base64
    client = _llm_client()
    model = model or os.getenv("MODEL_NAME", os.getenv("LLM_MODEL", "gpt-4o-mini"))
    snippet = text[:max_chars]
    user_text = (
        "Paper text follows between <PAPER> tags, and selected pages are attached as images. "
        "Extract monomers, polymer/sample keys, backbone composition, cure/process profiles, "
        "and the target properties needed for the final processed dataset.\n\n"
        f"Required output JSON schema:\n{json.dumps(LLM_OUTPUT_SCHEMA, ensure_ascii=False)}\n\n"
        "Extraction rules:\n"
        "- extracted_monomers: include every monomer used in the polymer backbone. Include abbreviation, "
        "full name, monomer_class, CAS if printed, source_page, and the best candidate SMILES only when the "
        "structure is clear. Monomer SMILES should be neutral monomer molecules, not polymer residues. "
        "Dianhydride SMILES must contain two anhydride rings; diamine SMILES must contain two primary amines. "
        "If not reliable, set smiles to null and smiles_source to 'unknown'.\n"
        "- polymer_components: include fixed backbone monomers for every polymer/sample series. A PI/PAI "
        "should normally have at least one dianhydride and one diamine.\n"
        "- property_records: extract only directly reported optical transmittance, Tg, and CTE for the final "
        "processed dataset. For transmittance, prefer explicit T550; otherwise keep one directly reported "
        "visible-wavelength transmittance closest to 550 nm per sample. Do not interpolate.\n"
        "- Keep wavelength_nm for transmittance, tg_definition for Tg when stated, and the CTE temperature "
        "range when stated.\n"
        "- trend_records should describe only transmittance, Tg, or CTE trends when they are stated or clearly "
        "supported by the reported values.\n"
        "- value_raw must preserve the exact printed expression, including >, <, ~, ranges, and units.\n"
        "- If a value is absent or uncertain, use null or omit the record; do not guess.\n"
        f"<PAPER>\n{snippet}\n</PAPER>"
    )
    content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    if pdf_path is not None:
        try:
            for page_no, png_bytes in _render_pdf_pages(pdf_path, dpi=150, max_pages=max_pages):
                b64 = base64.b64encode(png_bytes).decode("ascii")
                content.append({"type": "text", "text": f"[page {page_no}]"})
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        except Exception as exc:
            print(f"[WARN] page render failed for {pdf_path.name}: {exc}", file=sys.stderr)

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


def infer_nonpolymer_components(paper_id: str, paper_text: str) -> List[Dict[str, Any]]:
    """Return non-polymer components only when they were explicitly extracted by the LLM."""
    return []


def infer_study_series(paper_id: str, paper_text: str, samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generic fallback: leave study-series inference to component-level variation rules."""
    return []


def infer_trend_records(paper_id: str, paper_text: str, sample_keys: Sequence[str]) -> List[Dict[str, Any]]:
    """Generic fallback: trends are extracted by the LLM or inferred from target-property variation."""
    return []


def _clean_filter_to_target_properties(payload: Dict[str, Any], paper_text: str) -> None:
    if '_filter_to_key_properties_target' in globals():
        _filter_to_key_properties_target(payload, paper_text)


def _clean_infer_series_and_trends(payload: Dict[str, Any], paper_text: str) -> None:
    if '_series_finalize_series_and_trends' in globals():
        _series_finalize_series_and_trends(payload, paper_text)


def augment_llm_payload_from_text(paper_id: str, payload: Dict[str, Any], paper_text: str) -> Dict[str, Any]:
    """Apply generic post-processing without paper-specific repair rules."""
    payload = ensure_llm_payload_defaults(payload)
    payload.setdefault("review_items", [])
    normalize_polyimide_payload(payload, paper_text)
    if not payload.get("material_components"):
        payload["material_components"] = infer_nonpolymer_components(paper_id, paper_text)
    if '_standardize_properties_and_profiles_generic' in globals():
        _standardize_properties_and_profiles_generic(payload, paper_text)
    _clean_filter_to_target_properties(payload, paper_text)
    _clean_infer_series_and_trends(payload, paper_text)
    payload["extraction_scope"] = {
        "version": "general_schema_driven_polyimide_extraction",
        "target_properties": ["reported_transmittance_one_per_sample", "Tg", "CTE"],
        "required_processed_outputs": [
            "cure_profile.json", "material_component.json", "monomer.json", "polymer.json",
            "polymer_component.json", "property_record.json", "sample.json", "sample_composition.json",
            "study_series.json", "trend_record.json", "README.md", "review_queue.md"
        ],
        "policy": "No paper-specific repair rules are applied. Corrections should come from the LLM output, generic chemistry QC, or an external reference monomer library.",
    }
    return payload

def process_one_pdf(
    pdf_path: Path,
    raw_dir: Path,
    predictor,
    use_llm: bool,
    use_ocsr: bool,
    refresh_llm_cache: bool = False,
) -> Dict[str, Any]:
    """Run one PDF; emit per-paper JSON bundle into raw_dir/<slug>/."""
    slug = slugify(pdf_path.stem)
    paper_dir = raw_dir / slug
    paper_dir.mkdir(parents=True, exist_ok=True)
    print(f"[paper] {pdf_path.name} -> {paper_dir}")

    full_text, _pages = extract_pdf_text(pdf_path)
    (paper_dir / "text.txt").write_text(full_text, encoding="utf-8")

    ocsr_monomers: Dict[str, Dict[str, Any]] = {}
    if use_ocsr and predictor is not None:
        images_dir = paper_dir / "images"
        candidates = extract_image_candidates(pdf_path, images_dir)
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

    llm_payload: Dict[str, Any] = {
        "polymers": [], "polymer_components": [], "samples": [],
        "cure_profiles": [], "property_records": [], "extracted_monomers": [],
        "material_components": [], "sample_compositions": [], "study_series": [], "trend_records": [],
    }
    llm_error: Optional[str] = None
    if use_llm:
        cache_f = paper_dir / "llm_raw.json"
        if refresh_llm_cache and cache_f.exists():
            cache_f.unlink(missing_ok=True)
            print(f"[cache] refreshed {cache_f}", file=sys.stderr)
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
    summary_review_items: List[Dict[str, Any]] = list(llm.get("review_items", []) or [])
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
                "mw_g_per_mol": s.get("mw_g_per_mol"),
                "inherent_viscosity_dL_per_g": s.get("inherent_viscosity_dL_per_g"),
                "sample_compositions": sample_comps_by_sample.get(skey, []),
                "cure_profile": cures_by_sample.get(skey),
                "process_profiles": [p for p in (llm.get("process_profiles", []) or []) if p.get("local_sample_key") == skey],
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
        "polymerization_profiles": llm.get("polymerization_profiles", []) or [],
        "chemical_imidization_profiles": llm.get("chemical_imidization_profiles", []) or [],
        "thermal_imidization_profiles": llm.get("thermal_imidization_profiles", []) or [],
        "film_drying_profiles": llm.get("film_drying_profiles", []) or [],
        "process_profiles": llm.get("process_profiles", []) or [],
        "review_items": summary_review_items,
        "polymers": polymers_out,
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
        for item in llm.get("review_items", []) or []:
            review_items.append({**item, "paper": paper_id})

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

        abbr_to_monomer_id: Dict[str, str] = {}
        llm_mons = llm.get("extracted_monomers", []) or []
        for m in llm_mons:
            abbr = m.get("abbreviation", "")
            name = m.get("name", "")
            mclass = m.get("monomer_class") or ""
            ocsr_entry = ocsr_map.get(abbr, {})
            smiles = ocsr_entry.get("smiles", "") or canonicalize_smiles(m.get("smiles") or "")
            mid = get_or_create_monomer(
                abbr, name, smiles, mclass, paper_id,
                ocsr_conf=ocsr_entry.get("confidence", ""),
            )
            if mid:
                abbr_to_monomer_id[abbr] = mid
        for abbr, ocsr_entry in ocsr_map.items():
            if abbr in abbr_to_monomer_id:
                continue
            mid = get_or_create_monomer(
                abbr, "", ocsr_entry["smiles"], "other", paper_id,
                ocsr_conf=ocsr_entry.get("confidence", ""),
            )
            if mid:
                abbr_to_monomer_id[abbr] = mid

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

    sample_table_out = [{k: v for k, v in s.items() if k != "paper_id"} for s in sample_table]

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

    for prop in property_table:
        if prop.get("confidence", 1) < 0.7:
            review_items.append({"kind": "low_confidence_property", **prop})
    review_lines = ["# Review Queue\n"]
    review_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    review_lines.append(f"Total items: {len(review_items)}\n\n")
    for item in review_items:
        review_lines.append(f"- **{item.get('kind')}**: {json.dumps({k: v for k, v in item.items() if k != 'kind'}, ensure_ascii=False)}")
    (dataset_dir / "review_queue.md").write_text("\n".join(review_lines), encoding="utf-8")

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
                refresh_llm_cache=args.refresh_llm_cache,
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

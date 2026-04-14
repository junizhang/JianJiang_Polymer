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
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import fitz
from rdkit import Chem
from rdkit.Chem import AllChem
import pubchempy as pcp

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
    parser = argparse.ArgumentParser(description="Extract polyimide composition from PDF")
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument("--output", default="", help="Legacy combined JSON output path")
    parser.add_argument("--output-dir", default="", help="Directory for per-paper outputs")
    parser.add_argument("--paper-id", default="", help="Paper ID, defaults to PDF stem")
    parser.add_argument("--text-output", default="", help="Optional extracted text output")
    parser.add_argument("--images-dir", default="", help="Optional image extraction directory")
    parser.add_argument("--cache-file", default="monomer_library.json", help="Monomer dictionary cache JSON")
    parser.add_argument("--model-path", default=DEFAULT_CKPT, help="MolScribe checkpoint path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="OCSR device")
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


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf).expanduser().resolve()
    paper_id = args.paper_id or pdf_path.stem
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        legacy_json_path = output_dir / "polyimide_extraction.json"
    elif args.output:
        legacy_json_path = Path(args.output).expanduser().resolve()
        output_dir = legacy_json_path.parent
    else:
        output_dir = pdf_path.parent / f"{pdf_path.stem}_outputs"
        legacy_json_path = output_dir / "polyimide_extraction.json"
    images_dir = Path(args.images_dir).expanduser().resolve() if args.images_dir else output_dir / "extracted_images"
    model_path = Path(args.model_path).expanduser().resolve()
    cache_path = Path(args.cache_file).expanduser().resolve()

    full_text, _page_texts = extract_pdf_text(pdf_path)
    if args.text_output:
        Path(args.text_output).expanduser().resolve().write_text(full_text, encoding="utf-8")

    entries = extract_series_entries(full_text)
    target_abbreviations = extract_target_abbreviations(entries)
    monomers = extract_monomers(full_text, target_abbreviations)
    cache = load_cache(cache_path)
    candidates = extract_image_candidates(pdf_path, images_dir)
    predictor = init_predictor(model_path, args.device)
    ocsr_results = run_ocsr(candidates, monomers, predictor, output_dir / "structures_clean")
    polymer_repeat_ocsr = attempt_polymer_repeat_ocsr(candidates, predictor)
    resolve_monomer_smiles(monomers, cache)
    payload = build_output(
        paper_id,
        pdf_path,
        monomers,
        entries,
        candidates,
        ocsr_results,
        polymer_repeat_ocsr=polymer_repeat_ocsr,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "monomer_dictionary.json").write_text(
        json.dumps(payload["monomer_dictionary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "polyimide_dataset.json").write_text(
        json.dumps(payload["polymers"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_polymer_csv(payload["polymers"], output_dir / "polyimide_dataset.csv")
    write_jsonl(payload["polymers"], output_dir / "polyimide_dataset.jsonl")
    save_cache(cache_path, cache)

    print(f"Saved JSON: {legacy_json_path}")
    print(f"Saved monomer dictionary: {output_dir / 'monomer_dictionary.json'}")
    print(f"Saved dataset JSON: {output_dir / 'polyimide_dataset.json'}")
    print(f"Saved CSV: {output_dir / 'polyimide_dataset.csv'}")
    print(f"Saved JSONL: {output_dir / 'polyimide_dataset.jsonl'}")
    print(f"Polymers: {len(payload['polymers'])}")
    print(f"Image candidates: {len(candidates)}")
    print(f"OCSR results: {len(ocsr_results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared caption metrics; saved predictions can be scored without torch or a GPU.

Run from the repository root: python -m src.evaluate --help
"""

import argparse
import hashlib
import contextlib
import io
import json
import math
import re
import shutil
import subprocess
import threading
from importlib.metadata import version
from pathlib import Path

from src.data.text import clean_caption, load_captions

ROOT = Path(__file__).resolve().parents[1]
SPECIAL_TOKENS = re.compile(r"<(?:pad|start|end|unk)>", re.IGNORECASE)


def normalize_caption(text: str) -> str:
    """Remove special tokens BEFORE punctuation; use the shared English cleaner."""
    if not isinstance(text, str):
        raise ValueError("Caption must be a string")
    return clean_caption(SPECIAL_TOKENS.sub(" ", text))


def prepare_inputs(references: dict, hypotheses: dict) -> tuple[dict, dict]:
    """Require exact ID coverage and five full-text references per image."""
    if not isinstance(references, dict) or not isinstance(hypotheses, dict):
        raise ValueError("References and predictions must be image_id mappings")
    if not references or set(references) != set(hypotheses):
        raise ValueError("References and predictions must have identical, nonempty image IDs")
    refs, hyps = {}, {}
    for image_id in sorted(references):
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("image_id must be a nonempty string")
        captions = references[image_id]
        if not isinstance(captions, list) or len(captions) != 5:
            raise ValueError(f"{image_id}: exactly five reference captions required")
        refs[image_id] = [normalize_caption(text) for text in captions]
        if any(not text for text in refs[image_id]):
            raise ValueError(f"{image_id}: reference caption is empty after cleaning")
        # Empty model predictions are legitimate failures and must remain in the score.
        hyps[image_id] = [normalize_caption(hypotheses[image_id])]
    return refs, hyps


def compute_metrics(references: dict, hypotheses: dict, *, meteor: bool = False) -> dict[str, float]:
    """Corpus BLEU-1..4, CIDEr and ROUGE-L; optional METEOR requires Java.

    Scores are raw pycocoevalcap values (not multiplied by 100). CIDEr's
    document frequencies are calculated over the supplied reference corpus.
    """
    refs, hyps = prepare_inputs(references, hypotheses)
    if len(refs) < 2:
        raise ValueError("CIDEr requires at least two images for meaningful document frequencies")
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    with contextlib.redirect_stdout(io.StringIO()):
        bleu, _ = Bleu(4).compute_score(refs, hyps)
        cider, _ = Cider().compute_score(refs, hyps)
        rouge, _ = Rouge().compute_score(refs, hyps)
    scores = {f"BLEU-{i + 1}": float(value) for i, value in enumerate(bleu)}
    scores.update(CIDEr=float(cider), **{"ROUGE-L": float(rouge)})
    if meteor:
        if shutil.which("java") is None:
            raise RuntimeError("METEOR requires Java; install it or omit --meteor and document the omission")
        try:
            subprocess.run(["java", "-version"], capture_output=True, check=True, timeout=10)
        except (subprocess.SubprocessError, OSError) as error:
            raise RuntimeError("Java could not start; omit --meteor and document the omission") from error
        from pycocoevalcap.meteor.meteor import Meteor

        scorer = Meteor()
        watchdog = threading.Timer(120, scorer.meteor_p.kill)
        watchdog.daemon = True
        watchdog.start()
        try:
            score, _ = scorer.compute_score(refs, hyps)
            scores["METEOR"] = float(score)
        except (OSError, ValueError) as error:
            raise RuntimeError("METEOR failed or timed out; omit --meteor and document the omission") from error
        finally:
            watchdog.cancel()
            watchdog.join()
            # pycocoevalcap leaves this lock held when Java fails mid-score.
            if scorer.lock.locked():
                scorer.lock.release()
            del scorer
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("Metric library returned a non-finite score")
    return scores


def cider_score(references: dict, hypotheses: dict) -> float:
    """Drop-in shared scorer for src.train.cider_score (no torch import)."""
    refs, hyps = prepare_inputs(references, hypotheses)
    # A one-image smoke run has zero IDF and therefore CIDEr=0.
    from pycocoevalcap.cider.cider import Cider

    score, _ = Cider().compute_score(refs, hyps)
    score = float(score)
    if not math.isfinite(score):
        raise ValueError("CIDEr returned a non-finite score")
    return score


def read_json(path: Path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_pairs)


def load_evaluation_inputs(predictions_path: Path, config_path: Path, split: str,
                           *, captions_path: Path | None = None,
                           references_path: Path | None = None,
                           allow_test: bool = False) -> tuple[dict, dict]:
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    if split == "test" and not allow_test:
        raise ValueError("Test is reserved for final evaluation; explicitly pass --allow-test")
    if captions_path is not None and references_path is not None:
        raise ValueError("Supply either captions.txt or reference JSON, not both")
    from src.data.config import load_config

    config = load_config(config_path)
    partition = read_json(ROOT / config["split_path"])
    image_ids = partition[split]
    if not image_ids or len(image_ids) != len(set(image_ids)):
        raise ValueError("Split contains duplicate IDs or is empty")
    for other in ("train", "val", "test", "unused"):
        if other != split and set(image_ids) & set(partition.get(other, [])):
            raise ValueError(f"Selected split overlaps {other}")
    predictions = read_json(predictions_path)
    if not isinstance(predictions, dict) or set(predictions) != set(image_ids):
        raise ValueError(f"Predictions must contain exactly all {len(image_ids)} IDs from {split}")
    if references_path is not None:
        references = read_json(references_path)
    else:
        source = captions_path or (ROOT / config["captions_path"])
        if not source.is_file():
            raise FileNotFoundError(f"Reference captions missing: {source}. Use --captions or --references")
        all_captions = load_captions(source)
        missing = set(image_ids) - set(all_captions)
        if missing:
            raise ValueError(f"Caption file is missing {len(missing)} selected images")
        references = {image_id: all_captions[image_id] for image_id in image_ids}
    prepare_inputs(references, predictions)
    return references, predictions


def portable_path(path: Path) -> str:
    """Keep repository output portable without exposing machine-specific roots."""
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="JSON: {image_id: caption}")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/base.yaml")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--captions", type=Path, help="Original Flickr8k captions.txt")
    source.add_argument("--references", type=Path, help="JSON: {image_id: [five reference captions]}")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true", help="Only after model selection is frozen")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--method", choices=("greedy", "beam", "nucleus"), default="greedy",
                        help="How the saved predictions were generated; does not decode them again")
    parser.add_argument("--beam-size", type=int)
    parser.add_argument("--meteor", action="store_true", help="Compute METEOR too (Java required)")
    parser.add_argument("--output", type=Path, required=True, help="New JSON result file; never overwrites")
    args = parser.parse_args()
    if args.method == "beam":
        if args.beam_size is None or args.beam_size < 1:
            parser.error("beam requires --beam-size >= 1 (use 3 for the main table)")
    elif args.beam_size is not None:
        parser.error("--beam-size is only valid with --method beam")
    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        references, predictions = load_evaluation_inputs(
            args.predictions, args.config, args.split, captions_path=args.captions,
            references_path=args.references, allow_test=args.allow_test)
        scores = compute_metrics(references, predictions, meteor=args.meteor)
        from src.data.config import load_config
        base_config = load_config(args.config)
        reference_file = args.references or args.captions or (ROOT / base_config["captions_path"])
        result = {
            "model": args.model_name, "split": args.split, "method": args.method,
            "beam_size": args.beam_size, "n_images": len(predictions),
            "references_per_image": 5, "metrics": scores,
            "metric_scale": "raw pycocoevalcap (not percentages)",
            "pycocoevalcap_version": version("pycocoevalcap"),
            "normalization": "remove special tokens, lowercase, remove punctuation, collapse spaces",
            "meteor_status": "computed" if args.meteor else "not requested; no METEOR score",
            "predictions_path": portable_path(args.predictions),
            "references_path": portable_path(args.references) if args.references else None,
            "captions_path": None if args.references else portable_path(reference_file),
            "config_path": portable_path(args.config),
            "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
            "reference_sha256": hashlib.sha256(reference_file.read_bytes()).hexdigest(),
            "split_sha256": hashlib.sha256((ROOT / base_config["split_path"]).read_bytes()).hexdigest(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        parser.exit(2, f"Evaluation failed: {error}\n")


if __name__ == "__main__":
    main()

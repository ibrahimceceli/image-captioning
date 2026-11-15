"""Eğitilmiş bir checkpoint'ten doğrulama (veya final için test) cümleleri üretir.

python -m src.predict --checkpoint checkpoints/model1_lstm/best.pt --method beam --beam-size 3

Çözümleme src/decode.py'deki decode() ile yapılır. Çıktılar results/<ad>/ altına:
<split>_captions_<yöntem>.json ({image_id: cümle}) ve <split>_metrics.json
(yöntem başına CIDEr, görüntü sayısı ve görüntü başına çıkarım süresi).
"""

import argparse
import json
import time
from pathlib import Path

import torch

from src.data.config import load_config, repo_root
from src.data.dataset import build_eval_dataloader, load_references
from src.data.vocab import Vocabulary
from src.decode import decode
from src.train import build_model, cider_score


def method_tag(method: str, beam_size: int, top_p: float | None) -> str:
    if method == "beam":
        return f"beam{beam_size}"
    if method == "nucleus":
        return f"nucleus{top_p}"
    return method


def display_path(path: Path, root: Path) -> str:
    """Use repo-relative paths when possible, absolute paths otherwise."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpoint'ten cümle üretimi")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--allow-test", action="store_true", help="Test kümesi yalnız final tablo için bir kez açılır")
    parser.add_argument("--method", default="greedy", choices=("greedy", "beam", "nucleus"))
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Deneme için ilk N görüntü")
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Separate run directory; refuses to overwrite existing prediction or metric files")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root().resolve()
    if args.output_dir is not None:
        args.output_dir = (root / args.output_dir).resolve()
    base = load_config()
    vocab = Vocabulary.load(root / base["vocab_path"])
    device = torch.device(args.device)

    # Separate experiment runs without changing legacy output paths.
    tag = method_tag(args.method, args.beam_size, args.top_p)
    if args.output_dir is not None:
        for path in (args.output_dir / f"{args.split}_captions_{tag}.json",
                     args.output_dir / f"{args.split}_metrics.json"):
            if path.exists():
                raise FileExistsError(f"Output already exists: {path}")

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = state["config"]
    model = build_model(config, vocab).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    loader = build_eval_dataloader(
        args.split, batch_size=1, features_dir=args.features_dir, allow_test=args.allow_test,
    )
    references = load_references(args.split, allow_test=args.allow_test)
    tag = method_tag(args.method, args.beam_size, args.top_p)

    captions: dict[str, str] = {}
    elapsed = 0.0
    for index, batch in enumerate(loader):
        if args.limit and index >= args.limit:
            break
        features = batch["features"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        captions[batch["image_id"][0]] = decode(
            model, features, args.method, args.beam_size, vocab=vocab, top_p=args.top_p, seed=args.seed,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - started

    results_dir = args.output_dir if args.output_dir is not None else root / "results" / config["name"]
    results_dir.mkdir(parents=True, exist_ok=True)
    captions_path = results_dir / f"{args.split}_captions_{tag}.json"
    captions_path.write_text(json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_path = results_dir / f"{args.split}_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    metrics[tag] = {
        "checkpoint": display_path(args.checkpoint, root),
        "checkpoint_epoch": state["epoch"],
        "images": len(captions),
        "unique_captions": len(set(captions.values())),
        "unique_caption_ratio": len(set(captions.values())) / len(captions),
        "seed": args.seed,
        "top_p": args.top_p,
        "sampling_seed_policy": "decoder resets local CPU generator to the run seed for each image",
        "max_tokens_including_start": 34,
        "batch_size": 1,
        "amp": False,
        "CIDEr": round(cider_score(references, captions), 4),
        "ms_per_image": round(1000 * elapsed / len(captions), 2),
        "device": device.type,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{display_path(captions_path, root)} yazıldı")
    print(json.dumps({tag: metrics[tag]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

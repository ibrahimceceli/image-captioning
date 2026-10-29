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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    base = load_config()
    vocab = Vocabulary.load(root / base["vocab_path"])
    device = torch.device(args.device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
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

    results_dir = root / "results" / config["name"]
    results_dir.mkdir(parents=True, exist_ok=True)
    captions_path = results_dir / f"{args.split}_captions_{tag}.json"
    captions_path.write_text(json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_path = results_dir / f"{args.split}_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    metrics[tag] = {
        "checkpoint": args.checkpoint.resolve().relative_to(root).as_posix(),
        "checkpoint_epoch": state["epoch"],
        "images": len(captions),
        "unique_captions": len(set(captions.values())),
        "CIDEr": round(cider_score(references, captions), 4),
        "ms_per_image": round(1000 * elapsed / len(captions), 2),
        "device": device.type,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{captions_path.relative_to(root).as_posix()} yazıldı")
    print(json.dumps({tag: metrics[tag]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

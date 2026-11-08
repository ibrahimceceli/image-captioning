"""Model 1, 2 ve 3 için ortak eğitim betiği.

python -m src.train --config configs/model1.yaml

Önbellekten okur, teacher forcing ile eğitir, her epoch doğrulama kaybını,
her `cider_every` epoch'ta (ve kayıp iyileştiğinde) greedy CIDEr'i hesaplar.
Çıktılar: checkpoints/<ad>/{best,last}.pt ve results/<ad>/ altında geçmiş,
kayıp eğrisi, özet ve en iyi checkpoint'in doğrulama cümleleri.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from src.data.config import load_config, repo_root
from src.data.dataset import build_dataloader, build_eval_dataloader, collate_captions, load_references
from src.data.vocab import Vocabulary
from src.models.attention_lstm import AttentionLSTM
from src.models.lstm import CaptionLSTM

MODELS = {
    "lstm": CaptionLSTM,
    "attention_lstm": AttentionLSTM,
}
SPECIAL_WORDS = {"<pad>", "<start>", "<end>", "<unk>"}


def load_run_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for key in ("name", "seed", "model", "optimizer", "training", "early_stopping"):
        if key not in config:
            raise KeyError(f"Config alanı eksik: {key} ({path})")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(config: dict, vocab: Vocabulary) -> nn.Module:
    params = dict(config["model"])
    model_type = params.pop("type")
    if model_type not in MODELS:
        raise ValueError(f"Bilinmeyen model türü: {model_type}. Seçenekler: {', '.join(MODELS)}")
    return MODELS[model_type](vocab_size=len(vocab), padding_idx=vocab.pad_idx, **params)


def build_optimizer(config: dict, model: nn.Module) -> torch.optim.Optimizer:
    settings = config["optimizer"]
    name = settings["name"].lower()
    lr = float(settings["lr"])
    weight_decay = float(settings.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Bilinmeyen optimizer: {name}")


def build_scheduler(config: dict, optimizer: torch.optim.Optimizer):
    """warmup_steps verilirse lr ilk adımlarda doğrusal artar (Model 3), yoksa sabittir."""
    warmup = int(config["optimizer"].get("warmup_steps", 0))
    if warmup <= 0:
        return None
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / warmup))


def train_loader(config: dict, args) -> DataLoader:
    """train_fraction < 1 ise görüntü bazlı, seed ile sabit bir alt küme alınır (veri miktarı deneyi)."""
    batch_size = int(config["training"]["batch_size"])
    loader = build_dataloader(
        "train",
        batch_size=batch_size,
        seed=int(config["seed"]),
        num_workers=args.num_workers,
        use_features=True,
        features_dir=args.features_dir,
    )
    fraction = float(config.get("data", {}).get("train_fraction", 1.0))
    if not 0 < fraction <= 1:
        raise ValueError("train_fraction (0, 1] aralığında olmalı")
    if fraction == 1.0:
        return loader
    dataset = loader.dataset
    image_ids = sorted({image_id for image_id, _ in dataset.samples})
    keep = set(random.Random(int(config["seed"])).sample(image_ids, round(len(image_ids) * fraction)))
    indices = [index for index, (image_id, _) in enumerate(dataset.samples) if image_id in keep]
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_captions,
        generator=generator,
    )


def caption_loss(model: nn.Module, batch: dict, criterion: nn.Module, device: torch.device) -> Tensor:
    """Teacher forcing: girdi token_ids[:, :-1], hedef token_ids[:, 1:]."""
    features = batch["features"].to(device, non_blocking=True)
    token_ids = batch["token_ids"].to(device, non_blocking=True)
    logits, _ = model(features, token_ids[:, :-1])
    targets = token_ids[:, 1:]
    return criterion(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1))


@torch.no_grad()
def greedy_batch(model: nn.Module, features: Tensor, vocab: Vocabulary) -> list[str]:
    """Doğrulama izlemesi için toplu greedy çözümleme.

    src/decode.py'deki greedy ile aynı kurallar: <start> ve <pad> seçilemez,
    <end>'de durur, en fazla 33 adım, özel token'lar metinden silinir. Final
    tablo src/decode.py ile üretilir; bu fonksiyon yalnız eğitim sırasında
    1000 görüntüyü hızlı çözmek içindir.
    """
    tokens = torch.full((features.shape[0], 1), vocab.start_idx, dtype=torch.long, device=features.device)
    finished = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
    for _ in range(vocab.max_length - 1):
        logits, _ = model(features, tokens)
        step = logits[:, -1].float()
        step[:, vocab.start_idx] = -torch.inf
        step[:, vocab.pad_idx] = -torch.inf
        next_ids = step.argmax(dim=-1)
        next_ids = torch.where(finished, torch.full_like(next_ids, vocab.pad_idx), next_ids)
        tokens = torch.cat((tokens, next_ids.unsqueeze(1)), dim=1)
        finished |= next_ids == vocab.end_idx
        if bool(finished.all()):
            break
    captions = []
    for row in tokens[:, 1:].tolist():
        words = []
        for token_id in row:
            if token_id == vocab.end_idx:
                break
            word = vocab.itos[token_id]
            if word not in SPECIAL_WORDS:
                words.append(word)
        captions.append(" ".join(words))
    return captions


def cider_score(references: dict[str, list[str]], hypotheses: dict[str, str]) -> float:
    """Shared normalization and CIDEr; keep predict --limit subset support."""
    from src.evaluate import cider_score as shared_cider_score

    refs = {image_id: references[image_id] for image_id in hypotheses}
    return shared_cider_score(refs, hypotheses)


@torch.no_grad()
def validation_loss(model, loader, criterion, device, amp: bool) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            loss = caption_loss(model, batch, criterion, device)
        total += loss.item() * batch["token_ids"].shape[0]
        count += batch["token_ids"].shape[0]
    return total / count


@torch.no_grad()
def validation_captions(model, loader, vocab, device, amp: bool) -> dict[str, str]:
    model.eval()
    captions = {}
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            decoded = greedy_batch(model, features, vocab)
        captions.update(zip(batch["image_id"], decoded))
    return captions


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def plot_history(history: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, [row["train_loss"] for row in history], color="#3B6D8C", label="eğitim kaybı")
    ax.plot(epochs, [row["val_loss"] for row in history], color="#C44E52", label="doğrulama kaybı")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Çapraz entropi")
    ax.spines[["top", "right"]].set_visible(False)
    cider_rows = [row for row in history if row.get("val_cider") is not None]
    if cider_rows:
        twin = ax.twinx()
        twin.plot([row["epoch"] for row in cider_rows], [row["val_cider"] for row in cider_rows],
                  color="#6B8F71", marker="o", linestyle=":", label="doğrulama CIDEr (greedy)")
        twin.set_ylabel("CIDEr")
        twin.spines[["top"]].set_visible(False)
        lines = ax.get_legend_handles_labels()
        twin_lines = twin.get_legend_handles_labels()
        ax.legend(lines[0] + twin_lines[0], lines[1] + twin_lines[1], loc="upper right")
    else:
        ax.legend(loc="upper right")
    ax.set_title("Kayıp eğrileri")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def is_better(monitor: str, value: float | None, best: float | None) -> bool:
    if value is None:
        return False
    if best is None:
        return True
    return value < best if monitor == "val_loss" else value > best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model 1, 2 ve 3 için ortak eğitim")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--name", help="Çıktı klasörü adı (varsayılan: config'teki name)")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--limit-batches", type=int, help="Deneme için epoch başına en fazla N batch")
    parser.add_argument("--features-dir", type=Path, help="Önbellek başka klasördeyse (ör. Kaggle)")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", help="checkpoints/<ad>/last.pt'den devam et")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_run_config(args.config)
    base = load_config()
    name = args.name or config["name"]
    training = config["training"]
    stopping = config["early_stopping"]
    evaluation = config.get("evaluation", {})
    max_epochs = args.max_epochs or int(training["max_epochs"])
    monitor = stopping.get("monitor", "val_loss")
    select_by = evaluation.get("select_by", "val_cider")
    cider_every = int(evaluation.get("cider_every", 2))
    for key in (monitor, select_by):
        if key not in ("val_loss", "val_cider"):
            raise ValueError(f"İzlenen değer val_loss veya val_cider olmalı: {key}")

    set_seed(int(config["seed"]))
    device = torch.device(args.device)
    amp = bool(training.get("amp", True)) and device.type == "cuda"
    root = repo_root()
    vocab = Vocabulary.load(root / base["vocab_path"])
    checkpoint_dir = root / "checkpoints" / name
    results_dir = root / "results" / name

    loader = train_loader(config, args)
    val_loader = build_dataloader(
        "val", batch_size=int(training["batch_size"]), shuffle=False,
        num_workers=args.num_workers, use_features=True, features_dir=args.features_dir,
    )
    val_images = build_eval_dataloader(
        "val", batch_size=int(evaluation.get("batch_size", 64)),
        num_workers=args.num_workers, features_dir=args.features_dir,
    )
    references = load_references("val")

    model = build_model(config, vocab).to(device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_idx,
        label_smoothing=float(config.get("loss", {}).get("label_smoothing", 0.0)),
    )
    n_params = sum(p.numel() for p in model.parameters())

    history: list[dict] = []
    start_epoch = 1
    best = {"monitor": None, "select": None, "epoch": None, "stale": 0}
    if args.resume:
        state = torch.load(checkpoint_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        if scheduler is not None and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        history, best = state["history"], state["best"]
        start_epoch = state["epoch"] + 1
        print(f"{start_epoch - 1}. epoch'tan devam ediliyor")

    print(f"{name}: {n_params:,} parametre, {len(loader.dataset)} eğitim örneği, cihaz {device}, AMP {amp}")
    started = time.time() - (history[-1]["seconds"] if history else 0.0)
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for step, batch in enumerate(loader):
            if args.limit_batches and step >= args.limit_batches:
                break
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = caption_loss(model, batch, criterion, device)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(training["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            total += loss.item() * batch["token_ids"].shape[0]
            count += batch["token_ids"].shape[0]
        train_loss = total / count

        val_loss = validation_loss(model, val_loader, criterion, device, amp)
        loss_improved = best["monitor"] is None or (monitor == "val_loss" and val_loss < best["monitor"])
        val_cider, captions = None, None
        if monitor == "val_cider" or epoch % cider_every == 0 or loss_improved or epoch == max_epochs:
            captions = validation_captions(model, val_images, vocab, device, amp)
            val_cider = cider_score(references, captions)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_cider": None if val_cider is None else round(val_cider, 4),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - started, 1),
        }
        history.append(row)

        monitored = val_loss if monitor == "val_loss" else val_cider
        if is_better(monitor, monitored, best["monitor"]):
            best["monitor"], best["stale"] = monitored, 0
        else:
            best["stale"] += 1
        selected = val_loss if select_by == "val_loss" else val_cider
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "epoch": epoch,
            "config": config,
            "vocab_size": len(vocab),
            "history": history,
            "best": best,
        }
        if is_better(select_by, selected, best["select"]):
            best["select"], best["epoch"] = selected, epoch
            state["best"] = best
            save_checkpoint(checkpoint_dir / "best.pt", state)
            if captions is not None:
                results_dir.mkdir(parents=True, exist_ok=True)
                (results_dir / "val_captions_best.json").write_text(
                    json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        save_checkpoint(checkpoint_dir / "last.pt", state)

        cider_text = "-" if val_cider is None else f"{val_cider:.4f}"
        print(f"epoch {epoch}: train {train_loss:.4f}, val {val_loss:.4f}, CIDEr {cider_text}, "
              f"en iyi epoch {best['epoch']}, sabır {best['stale']}/{stopping['patience']}")
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        plot_history(history, results_dir / "loss_curve.png")
        if best["stale"] >= int(stopping["patience"]):
            print(f"Erken durdurma: {monitor} {stopping['patience']} epoch iyileşmedi")
            break

    best_row = next(row for row in history if row["epoch"] == best["epoch"])
    summary = {
        "name": name,
        "model": config["model"]["type"],
        "parameters": n_params,
        "train_examples": len(loader.dataset),
        "epochs_run": history[-1]["epoch"],
        "best_epoch": best["epoch"],
        "select_by": select_by,
        "early_stopping_monitor": monitor,
        "best_val_loss": best_row["val_loss"],
        "best_val_cider_greedy": best_row["val_cider"],
        "train_minutes": round(history[-1]["seconds"] / 60, 1),
        "checkpoint": (checkpoint_dir / "best.pt").relative_to(root).as_posix(),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

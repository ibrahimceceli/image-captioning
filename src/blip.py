"""BLIP fine-tuning and validation. No test split is exposed by this workflow."""
import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from src.data.config import repo_root
from src.data.text import clean_caption, load_captions
from src.evaluate import cider_score, normalize_caption, read_json

MODEL_ID = "Salesforce/blip-image-captioning-base"


def selected_data(captions_path, images_dir, split_path, split):
    if split not in ("train", "val"):
        raise ValueError("Only train/val are allowed; final test is a separate team step")
    partition = read_json(Path(split_path))
    for key, count in (("train", 6000), ("val", 1000)):
        if len(partition[key]) != count or len(set(partition[key])) != count:
            raise ValueError(f"Expected {count} unique {key} images")
    groups = [set(partition.get(key, [])) for key in ("train", "val", "test", "unused")]
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1:]):
        raise ValueError("Split contains overlapping images")
    all_captions = load_captions(Path(captions_path))
    selected = {}
    for image_id in partition[split]:
        if len(all_captions.get(image_id, [])) != 5:
            raise ValueError(f"Five references required: {image_id}")
        if not (Path(images_dir) / image_id).is_file():
            raise FileNotFoundError(Path(images_dir) / image_id)
        selected[image_id] = [clean_caption(c) for c in all_captions[image_id]]
    return selected


def generation_options(method, beam_size=3):
    if method not in ("greedy", "beam") or not 1 <= beam_size <= 5:
        raise ValueError("Use greedy or beam with beam_size 1..5")
    return {"do_sample": False, "num_beams": 1 if method == "greedy" else beam_size,
            "max_new_tokens": 33, "min_length": 0, "length_penalty": 1.0,
            "repetition_penalty": 1.0, "no_repeat_ngram_size": 0}


def load_blip(source, device, dropout=None):
    from transformers import BlipConfig, BlipForConditionalGeneration, BlipProcessor
    processor = BlipProcessor.from_pretrained(source)
    config = BlipConfig.from_pretrained(source)
    if dropout is not None:
        config.text_config.hidden_dropout_prob = dropout
        config.text_config.attention_probs_dropout_prob = dropout
        config.vision_config.attention_dropout = dropout
    model = BlipForConditionalGeneration.from_pretrained(source, config=config).to(device)
    return processor, model


def sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict(model, processor, references, images_dir, device, method):
    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    model.eval()
    captions = {}
    elapsed = 0.0
    with torch.inference_mode():
        # Untimed warmup, using a validation image (never the test set).
        with Image.open(Path(images_dir) / next(iter(references))) as image:
            inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)
            model.generate(**inputs, **generation_options(method))
        sync(device)
        for image_id in tqdm(references, desc=method):
            with Image.open(Path(images_dir) / image_id) as image:
                rgb = image.convert("RGB")
            sync(device)
            started = time.perf_counter()
            inputs = processor(images=rgb, return_tensors="pt").to(device)
            tokens = model.generate(**inputs, **generation_options(method))
            caption = normalize_caption(processor.decode(tokens[0], skip_special_tokens=True))
            sync(device)
            elapsed += time.perf_counter() - started
            captions[image_id] = caption
    return captions, 1000 * elapsed / len(captions)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def collate_training(batch, processor, bos_token_id):
    images, captions = zip(*batch)
    inputs = processor(images=list(images), text=list(captions), padding=True,
                       truncation=True, max_length=34, return_tensors="pt")
    inputs["input_ids"][:, 0] = bos_token_id
    inputs["labels"] = inputs["input_ids"].clone().masked_fill(inputs["attention_mask"] == 0, -100)
    return inputs


def train(args, processor, model, device, train_refs, val_refs):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm

    class Captions(Dataset):
        def __init__(self, references):
            self.samples = [(key, caption) for key, captions in references.items() for caption in captions]
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, index):
            key, caption = self.samples[index]
            with Image.open(args.images_dir / key) as image:
                return image.convert("RGB"), caption

    generator = torch.Generator().manual_seed(42)
    from functools import partial
    collate = partial(collate_training, processor=processor, bos_token_id=model.config.text_config.bos_token_id)
    loader = DataLoader(Captions(train_refs), batch_size=8, shuffle=True, num_workers=0,
                        generator=generator, collate_fn=collate)
    val_loader = DataLoader(Captions(val_refs), batch_size=8, shuffle=False, num_workers=0,
                            collate_fn=collate)
    # Full fine-tuning: both vision encoder and text decoder remain trainable.
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)
    amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    history, best, stale, best_loss = [], float('-inf'), 0, float('inf')
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in tqdm(loader, desc=f'Train {epoch}'):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * batch['labels'].shape[0]
            count += batch['labels'].shape[0]
        model.eval()
        val_total, val_count = 0.0, 0
        with torch.inference_mode():
            for batch in val_loader:
                batch = batch.to(device)
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    loss = model(**batch).loss
                val_total += loss.item() * batch['labels'].shape[0]
                val_count += batch['labels'].shape[0]
        val_loss = val_total / val_count
        improved_loss = val_loss < best_loss
        best_loss = min(best_loss, val_loss)
        stale = 0 if improved_loss else stale + 1
        score = None
        if improved_loss or epoch % 2 == 0 or epoch == args.epochs:
            captions, _ = predict(model, processor, val_refs, args.images_dir, device, 'greedy')
            score = cider_score(val_refs, captions)
            if score > best:
                best = score
                model.save_pretrained(args.checkpoint / 'best', safe_serialization=True)
                processor.save_pretrained(args.checkpoint / 'best')
                save_json(args.checkpoint / 'selection.json', {'epoch': epoch, 'val_cider': score})
        history.append({'epoch': epoch, 'train_loss': total / count, 'val_loss': val_loss,
                        'val_cider': score, 'seconds': time.perf_counter() - started})
        save_json(args.output / 'history.json', history)
        model.save_pretrained(args.checkpoint / 'last', safe_serialization=True)
        processor.save_pretrained(args.checkpoint / 'last')
        if stale >= 5:
            break
    return args.checkpoint / 'best'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['pretrained', 'finetune'])
    parser.add_argument('--images-dir', type=Path, required=True)
    parser.add_argument('--captions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=repo_root() / 'checkpoints/model4_blip')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.images_dir = args.images_dir.resolve()
    args.captions = args.captions.resolve()
    args.output = args.output.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if not 1 <= args.epochs <= 5:
        parser.error('epochs must be 1..5')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new empty output directory; existing runs are preserved')
    if args.action == 'finetune' and args.checkpoint.exists() and any(args.checkpoint.iterdir()):
        parser.error('Use a new checkpoint directory; this workflow does not resume optimizer state')
    import numpy as np
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable: enable a GPU or explicitly select --device cpu')
    split_path = repo_root() / 'src/data/artifacts/flickr8k_split.json'
    val_refs = selected_data(args.captions, args.images_dir, split_path, 'val')
    train_refs = selected_data(args.captions, args.images_dir, split_path, 'train') if args.action == 'finetune' else None
    args.output.mkdir(parents=True, exist_ok=True)
    processor, model = load_blip(MODEL_ID, device, dropout=0.1 if train_refs else None)
    resolved_revision = getattr(model.config, '_commit_hash', None)
    source = MODEL_ID
    if train_refs:
        source = train(args, processor, model, device, train_refs, val_refs)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        processor, model = load_blip(str(source), device)
    summary = {'name': args.output.name, 'model': 'blip', 'variant': args.action,
               'parameters': sum(p.numel() for p in model.parameters()),
               'checkpoint': str(source), 'pretrained_model': MODEL_ID,
               'pretrained_revision': resolved_revision, 'seed': 42,
               'split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
               'generation': generation_options('beam'), 'epochs': args.epochs if train_refs else 0}
    if train_refs:
        summary['best_epoch'] = read_json(args.checkpoint / 'selection.json')['epoch']
    save_json(args.output / 'summary.json', summary)
    (args.output / 'environment.txt').write_bytes(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']))
    metadata = {}
    for method, tag in [('greedy', 'greedy'), ('beam', 'beam3')]:
        captions, milliseconds = predict(model, processor, val_refs, args.images_dir, device, method)
        predictions = args.output / f'val_captions_{tag}.json'
        save_json(predictions, captions)
        metadata[tag] = {'checkpoint': str(source), 'checkpoint_epoch': summary.get('best_epoch'),
                         'images': len(captions), 'unique_captions': len(set(captions.values())),
                         'CIDEr': cider_score(val_refs, captions), 'ms_per_image': milliseconds,
                         'device': device.type}
        save_json(args.output / 'val_metrics.json', metadata)
        command = [sys.executable, '-m', 'src.evaluate', '--predictions', str(predictions),
                   '--captions', str(args.captions), '--model-name', args.output.name,
                   '--method', method, '--output', str(args.output / f'val_metrics_{tag}.json')]
        if method == 'beam':
            command += ['--beam-size', '3']
        subprocess.run(command, check=True, cwd=repo_root())


if __name__ == '__main__':
    main()

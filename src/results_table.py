"""Birleşik doğrulama tablosu: results/ altındaki JSON'lardan ana tablo + ablation tabloları.

    python -m src.results_table [--results-dir results] [--out-md results/val_table.md]
                                [--out-csv results/val_table.csv]

Sayılar yalnız JSON'lardan okunur (elle yazılmaz). Kaynaklar:
  - src/evaluate.py çıktısı  <run>/val_metrics_{greedy,beam3}.json  -> BLEU/METEOR/ROUGE-L/CIDEr
  - src/predict.py çıktısı   <run>/val_metrics.json                 -> farklı cümle, ms/görüntü, epoch
  - eğitim özeti             <run>/summary.json (veya eşlemede verilen yol) -> parametre sayısı
Sonucu olmayan koşular "henüz yok" olarak görünür; betik çökmez.
Tutarlılık uyarıları md'nin sonuna ve konsola yazılır.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Eşlemeler: hangi koşu hangi modeli temsil ediyor. Yeni koşu gelince yalnız burası değişir.
# "summary": summary.json'un results/ altındaki yolu; verilmezse <run>/summary.json.
# ---------------------------------------------------------------------------
MAIN_RUNS = [
    {"label": "Model 1 — CNN-LSTM", "run": "model1_lstm"},
    {"label": "Model 2 — Attention LSTM", "run": "model2_attention_lstm",
     "summary": "model2_attention_lstm_full_20261104_01/summary.json"},
    {"label": "Model 3 — Transformer", "run": "model3_transformer"},
    {"label": "Model 4 — BLIP", "run": "model4_blip"},
]

# (a) veri miktarı: aynı config, tek fark data.train_fraction. Δ referansı: %100.
DATA_FRACTION_RUNS = [
    {"label": "%25", "run": "model1_lstm_frac25"},
    {"label": "%50", "run": "model1_lstm_frac50"},
    {"label": "%100", "run": "model1_lstm"},
]
DATA_FRACTION_BASELINE = "model1_lstm"
DATA_FRACTION_CONFIGS = [("configs/model1.yaml", "configs/model1_frac25.yaml"),
                         ("configs/model1.yaml", "configs/model1_frac50.yaml")]

# (b) çözümleme yöntemi: aynı checkpoint, tek fark kod çözme. Δ referansı: aynı modelin greedy'si.
DECODING_RUNS = [
    {"label": "Model 1 — CNN-LSTM", "run": "model1_lstm"},
    {"label": "Model 2 — Attention LSTM", "run": "model2_attention_lstm"},
]
NUCLEUS = {
    "label": "Model 2 — Attention LSTM",
    "run": "model2_attention_lstm",
    "predict_key": "nucleus0.9",
    "seed_glob": "model2_attention_lstm/nucleus_p09_seed*",   # her birinde quality_metrics.json + val_metrics.json
    "summary": "model2_attention_lstm/nucleus_p09_summary.json",
}

# (c) attention var/yok: Δ referansı Model 1. Config farkı configs/ dosyalarından otomatik yazılır.
ATTENTION_RUNS = [
    {"label": "Model 1 — attention yok", "run": "model1_lstm"},
    {"label": "Model 2 — attention var", "run": "model2_attention_lstm"},
]
ATTENTION_BASELINE = "model1_lstm"
ATTENTION_CONFIGS = ("configs/model1.yaml", "configs/model2.yaml")   # repo köküne göre
ATTENTION_EXPECTED_DIFF = {"name", "model.type", "model.attention_dim", "model.num_layers"}

# (d) embedding: 300 boyutlu rastgele başlatma vs GloVe-300. Klasörler bu öneklerle otomatik bulunur.
EMBEDDING_PREFIXES = ["model1_lstm_emb300", "model1_lstm_glove300"]
EMBEDDING_BASELINE = "model1_lstm_emb300"
EMBEDDING_CONFIGS = ("configs/model1_emb300.yaml", "configs/model1_glove300.yaml")
EMBEDDING_EXPECTED_DIFF = {"name", "embeddings.path", "embeddings.dim", "embeddings.freeze"}

METHODS = [("greedy", "greedy", "val_metrics_greedy.json", None),
           ("beam3", "beam-3", "val_metrics_beam3.json", 3)]
METRIC_KEYS = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]
EXPECTED_IMAGES = 1000
EXPECTED_REFS = 5
EXPECTED_SPLIT = "val"
MISSING = "henüz yok"
DASH = "—"


@dataclass
class Row:
    label: str
    run: str
    method: str                     # greedy | beam3 | nucleus0.9
    available: bool = False
    metrics: dict = field(default_factory=dict)
    metrics_std: dict = field(default_factory=dict)   # yalnız çok-seed satırları
    unique: float | None = None
    unique_ratio: float | None = None
    unique_ratio_std: float | None = None
    ms: float | None = None
    epoch: int | None = None
    params: int | None = None
    device: str | None = None
    n_seeds: int | None = None


class Collector:
    """JSON okur; okunan/eksik dosyaları ve uyarıları toplar."""

    def __init__(self, results_dir: Path):
        self.root = results_dir
        self.sources: set[str] = set()
        self.missing: set[str] = set()
        self.warnings: list[str] = []
        self._checked: set[str] = set()
        self._cache: dict[str, dict | None] = {}
        self.hashes: dict[str, set[str]] = {"reference_sha256": set(), "split_sha256": set()}

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def load(self, rel_path: str) -> dict | None:
        if rel_path in self._cache:
            return self._cache[rel_path]
        path = self.root / rel_path
        data = None
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.sources.add(rel_path)
            except (OSError, json.JSONDecodeError) as error:
                self.warn(f"`{rel_path}` okunamadı: {error}")
        else:
            self.missing.add(rel_path)
        self._cache[rel_path] = data
        return data

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    # --- evaluate.py çıktısı kontrolleri ---
    def check_eval(self, rel_path: str, data: dict, method: str, beam_size: int | None) -> None:
        if rel_path in self._checked:
            return
        self._checked.add(rel_path)
        if data.get("split") != EXPECTED_SPLIT:
            self.warn(f"`{rel_path}`: split = {data.get('split')!r} (beklenen {EXPECTED_SPLIT!r})")
        if data.get("n_images") != EXPECTED_IMAGES:
            self.warn(f"`{rel_path}`: n_images = {data.get('n_images')} (beklenen {EXPECTED_IMAGES})")
        if data.get("references_per_image", EXPECTED_REFS) != EXPECTED_REFS:
            self.warn(f"`{rel_path}`: references_per_image = {data.get('references_per_image')} "
                      f"(beklenen {EXPECTED_REFS})")
        if data.get("method") != method:
            self.warn(f"`{rel_path}`: method = {data.get('method')!r} (beklenen {method!r})")
        if beam_size is not None and data.get("beam_size") != beam_size:
            self.warn(f"`{rel_path}`: beam_size = {data.get('beam_size')} (beklenen {beam_size})")
        for key in self.hashes:
            if data.get(key):
                self.hashes[key].add(data[key])

    def check_predict(self, rel_path: str, key: str, entry: dict) -> None:
        tag = f"{rel_path}[{key}]"
        if tag in self._checked:
            return
        self._checked.add(tag)
        if entry.get("images") != EXPECTED_IMAGES:
            self.warn(f"`{rel_path}` [{key}]: images = {entry.get('images')} (beklenen {EXPECTED_IMAGES})")

    def check_cider(self, eval_path: str, eval_cider: float | None, pred_path: str, key: str,
                    pred_cider: float | None) -> None:
        tag = f"cider:{eval_path}:{pred_path}:{key}"
        if tag in self._checked or eval_cider is None or pred_cider is None:
            return
        self._checked.add(tag)
        if f"{eval_cider:.4f}" != f"{pred_cider:.4f}":
            self.warn(f"CIDEr tutarsız: `{eval_path}` {eval_cider:.4f} ≠ `{pred_path}` [{key}] {pred_cider:.4f}")


# summary.json yolu koşu adına göre tek yerden: MAIN_RUNS'taki "summary" alanı tüm tablolarda geçerli.
SUMMARY_PATHS = {s["run"]: s["summary"] for s in MAIN_RUNS if s.get("summary")}


def _summary_path(spec: dict) -> str:
    return spec.get("summary") or SUMMARY_PATHS.get(spec["run"]) or f"{spec['run']}/summary.json"


def load_row(col: Collector, spec: dict, method_key: str) -> Row:
    """evaluate.py + predict.py + summary.json -> tek satır (greedy veya beam3)."""
    _, _, eval_name, beam_size = next(m for m in METHODS if m[0] == method_key)
    run = spec["run"]
    row = Row(label=spec["label"], run=run, method=method_key)
    eval_path = f"{run}/{eval_name}"
    pred_path = f"{run}/val_metrics.json"
    evaluation = col.load(eval_path)
    predict = col.load(pred_path)
    summary = col.load(_summary_path(spec))

    if evaluation is not None:
        row.available = True
        col.check_eval(eval_path, evaluation, "greedy" if beam_size is None else "beam", beam_size)
        row.metrics = dict(evaluation.get("metrics", {}))
    entry = (predict or {}).get(method_key)
    if predict is not None and entry is None:
        col.warn(f"`{pred_path}` içinde `{method_key}` anahtarı yok")
    if entry is not None:
        col.check_predict(pred_path, method_key, entry)
        row.unique = entry.get("unique_captions")
        images = entry.get("images")
        row.unique_ratio = entry.get("unique_caption_ratio")
        if row.unique_ratio is None and row.unique is not None and images:
            row.unique_ratio = row.unique / images
        row.ms = entry.get("ms_per_image")
        row.epoch = entry.get("checkpoint_epoch")
        row.device = entry.get("device")
        if evaluation is not None:
            col.check_cider(eval_path, row.metrics.get("CIDEr"), pred_path, method_key, entry.get("CIDEr"))
    if summary is not None:
        row.params = summary.get("parameters")
        best = summary.get("best_epoch")
        if row.epoch is not None and best is not None and best != row.epoch:
            col.warn(f"`{run}`: predict checkpoint_epoch {row.epoch} ≠ `{_summary_path(spec)}` best_epoch {best}")
        ckpt_sum = summary.get("checkpoint")
        ckpt_pred = (entry or {}).get("checkpoint")
        if ckpt_sum and ckpt_pred and Path(ckpt_sum).as_posix() != Path(ckpt_pred).as_posix():
            col.warn(f"`{run}`: predict checkpoint `{ckpt_pred}` ≠ summary checkpoint `{ckpt_sum}`")
    return row


def load_nucleus(col: Collector, spec: dict) -> Row:
    """Çok-seed nucleus: seed klasörlerinden ortalama ± örneklem std (ddof=1)."""
    row = Row(label=spec["label"], run=spec["run"], method=spec["predict_key"])
    seed_dirs = sorted(p for p in col.root.glob(spec["seed_glob"]) if p.is_dir())
    per_metric: dict[str, list[float]] = {}
    ratios, ms, epochs, devices = [], [], set(), set()
    for seed_dir in seed_dirs:
        rel = col.rel(seed_dir)
        evaluation = col.load(f"{rel}/quality_metrics.json")
        predict = col.load(f"{rel}/val_metrics.json")
        if evaluation is None:
            continue
        col.check_eval(f"{rel}/quality_metrics.json", evaluation, "nucleus", None)
        for key, value in evaluation.get("metrics", {}).items():
            per_metric.setdefault(key, []).append(value)
        entry = (predict or {}).get(spec["predict_key"])
        if entry is not None:
            col.check_predict(f"{rel}/val_metrics.json", spec["predict_key"], entry)
            col.check_cider(f"{rel}/quality_metrics.json", evaluation["metrics"].get("CIDEr"),
                            f"{rel}/val_metrics.json", spec["predict_key"], entry.get("CIDEr"))
            if entry.get("unique_caption_ratio") is not None:
                ratios.append(entry["unique_caption_ratio"])
            if entry.get("ms_per_image") is not None:
                ms.append(entry["ms_per_image"])
            epochs.add(entry.get("checkpoint_epoch"))
            devices.add(entry.get("device"))
    if not seed_dirs:
        col.missing.add(spec["seed_glob"])
    n = len(per_metric.get("CIDEr", []))
    if n:
        row.available = True
        row.n_seeds = n
        row.metrics = {k: statistics.fmean(v) for k, v in per_metric.items()}
        row.metrics_std = {k: statistics.stdev(v) if len(v) > 1 else 0.0 for k, v in per_metric.items()}
        if ratios:
            row.unique_ratio = statistics.fmean(ratios)
            row.unique_ratio_std = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
        row.ms = statistics.fmean(ms) if ms else None
        row.epoch = epochs.pop() if len(epochs) == 1 else None
        row.device = devices.pop() if len(devices) == 1 else None
    # Özet dosyasıyla çapraz kontrol (ortalama/std 4 basamakta aynı olmalı).
    summary = col.load(spec["summary"])
    if summary is not None and n:
        if summary.get("split") != EXPECTED_SPLIT or summary.get("n_images") != EXPECTED_IMAGES:
            col.warn(f"`{spec['summary']}`: split/n_images = {summary.get('split')!r}/{summary.get('n_images')}")
        if len(summary.get("seeds", [])) != n:
            col.warn(f"`{spec['summary']}`: {len(summary.get('seeds', []))} seed listeliyor, {n} seed klasörü okundu")
        for key, stats in summary.get("quality_across_seeds", {}).items():
            mean, std = row.metrics.get(key), row.metrics_std.get(key)
            if mean is None:
                continue
            if f"{mean:.4f}" != f"{stats.get('mean', float('nan')):.4f}" or \
                    f"{std:.4f}" != f"{stats.get('sample_std_ddof1', float('nan')):.4f}":
                col.warn(f"Nucleus {key}: seed klasörlerinden {mean:.4f} ± {std:.4f}, "
                         f"`{spec['summary']}` {stats.get('mean'):.4f} ± {stats.get('sample_std_ddof1'):.4f}")
    return row


def _flatten(data, prefix: str = "") -> dict:
    if not isinstance(data, dict):
        return {prefix: data}
    out = {}
    for key, value in data.items():
        out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    return out


def config_diff(repo_root: Path, paths: tuple[str, str]) -> list[str] | None:
    """İki config arasında değeri farklı (ya da yalnız birinde olan) anahtarlar; okunamazsa None."""
    try:
        import yaml
        a, b = (_flatten(yaml.safe_load((repo_root / p).read_text(encoding="utf-8")) or {}) for p in paths)
    except Exception:   # yaml yok / dosya yok: cümle sabit metinle kalır
        return None
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def config_note(col: "Collector", repo_root: Path, paths: tuple[str, str], expected: set[str], what: str) -> str:
    """Ablation cümlesi için config farkı; beklenmeyen fark varsa uyarı."""
    diff = config_diff(repo_root, paths)
    if diff is None:
        return f"`{paths[0]}` / `{paths[1]}` okunamadı, config farkı doğrulanmadı"
    extra = [k for k in diff if k not in expected]
    if extra:
        col.warn(f"{what} ablation'ı tek değişkenli değil; `{paths[0]}` / `{paths[1]}` ayrıca farklı: "
                 + ", ".join(f"`{k}`" for k in extra))
    return (f"`{paths[0]}` → `{paths[1]}` farklı anahtarlar: "
            + ", ".join(f"`{k}`" for k in diff) + " (diğer tüm ayarlar aynı)")


def collapse_missing(root: Path, missing: set[str]) -> list[str]:
    """Klasörü hiç olmayan koşular tek satır; diğerleri dosya dosya."""
    out, absent = [], set()
    for rel in sorted(missing):
        top = rel.split("/", 1)[0]
        if "/" in rel and "*" not in rel and not (root / top).is_dir():
            absent.add(top)
        else:
            out.append(f"`{rel}`")
    return [f"`{top}/` (klasör yok — koşu henüz yapılmadı)" for top in sorted(absent)] + out


def discover_embedding_runs(root: Path) -> list[dict]:
    found = []
    for prefix in EMBEDDING_PREFIXES:
        dirs = sorted(p.name for p in root.glob(f"{prefix}*") if p.is_dir())
        found.extend(dirs or [prefix])
    return [{"label": name, "run": name} for name in found]


# ---------------------------------------------------------------------------
# Biçimlendirme
# ---------------------------------------------------------------------------
def f4(value: float | None) -> str:
    return DASH if value is None else f"{value:.4f}"


def fmt_metric(row: Row, key: str) -> str:
    value = row.metrics.get(key)
    if value is None:
        return DASH
    if key in row.metrics_std:
        return f"{value:.4f} ± {row.metrics_std[key]:.4f}"
    return f"{value:.4f}"


def fmt_delta(value: float | None, ref: float | None) -> str:
    if value is None or ref is None:
        return DASH
    return f"{value - ref:+.4f}"


def fmt_int(value) -> str:
    return DASH if value is None else f"{int(value):,}"


def fmt_ms(value) -> str:
    return DASH if value is None else f"{value:.2f}"


def method_label(key: str) -> str:
    for m_key, label, _, _ in METHODS:
        if m_key == key:
            return label
    if key.startswith("nucleus"):
        return f"nucleus p={key[len('nucleus'):]}"
    return key


def md_table(header: list[str], rows: list[list[str]], align: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---:" if a == "r" else "---" for a in align) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _has_meteor(rows: list[Row]) -> bool:
    return any("METEOR" in r.metrics for r in rows)


def main_table_rows(rows: list[Row]) -> list[list[str]]:
    out = []
    for r in rows:
        if not r.available:
            out.append([r.label, method_label(r.method)] + [MISSING] + [""] * 10)
            continue
        out.append([r.label, method_label(r.method)] + [fmt_metric(r, k) for k in METRIC_KEYS] +
                   [fmt_int(r.unique), fmt_ms(r.ms), fmt_int(r.params), DASH if r.epoch is None else str(r.epoch)])
    return out


MAIN_HEADER = ["Model", "Çözme", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr",
               "Farklı cümle (/1000)", "ms/görüntü", "Parametre", "Checkpoint epoch"]


def ablation_table(rows: list[Row], refs: list[Row | None], *, ratio: bool = False) -> str:
    keys = [k for k in METRIC_KEYS if k != "METEOR" or _has_meteor(rows)]
    header = ["Koşu", "Çözme"] + keys + ["Δ CIDEr", "Farklı cümle oranı" if ratio else "Farklı cümle (/1000)",
                                         "ms/görüntü", "Epoch"]
    body = []
    for r, ref in zip(rows, refs):
        name = r.label if r.run == r.label else f"{r.label} (`{r.run}`)"
        if not r.available:
            body.append([name, method_label(r.method), MISSING] + [""] * (len(header) - 3))
            continue
        if ref is None or ref is r:
            delta = "ref." if ref is r else DASH
        else:
            delta = fmt_delta(r.metrics.get("CIDEr"), ref.metrics.get("CIDEr") if ref.available else None)
        if ratio:
            uniq = f4(r.unique_ratio)
            if r.unique_ratio_std is not None and r.n_seeds:
                uniq = f"{r.unique_ratio:.4f} ± {r.unique_ratio_std:.4f}"
        else:
            uniq = fmt_int(r.unique)
        body.append([name, method_label(r.method)] + [fmt_metric(r, k) for k in keys] +
                    [delta, uniq, fmt_ms(r.ms), DASH if r.epoch is None else str(r.epoch)])
    return md_table(header, body, ["l", "l"] + ["r"] * (len(header) - 2))


def pair_rows(col: Collector, specs: list[dict], baseline_run: str | None) -> tuple[list[Row], list[Row | None]]:
    """Her spec için greedy + beam3; Δ referansı aynı yöntemdeki baseline koşusu."""
    rows, refs = [], []
    for method_key, *_ in METHODS:
        group = [load_row(col, s, method_key) for s in specs]
        base = next((r for r in group if r.run == baseline_run), None)
        rows += group
        refs += [base] * len(group)
    return rows, refs


def devices_note(rows: list[Row]) -> str:
    devices = sorted({r.device for r in rows if r.device})
    return ", ".join(devices) if devices else DASH


def build(results_dir: Path, repo_root: Path | None = None) -> tuple[str, list[list[str]], list[str]]:
    """Markdown metni, ana tablo CSV satırları ve konsol uyarılarını döndürür (dosya yazmaz)."""
    repo_root = results_dir.parent if repo_root is None else repo_root
    col = Collector(results_dir)

    # Ana tablo
    main_rows: list[Row] = []
    for spec in MAIN_RUNS:
        for method_key, *_ in METHODS:
            main_rows.append(load_row(col, spec, method_key))

    # (a) veri miktarı
    frac_rows, frac_refs = pair_rows(col, DATA_FRACTION_RUNS, DATA_FRACTION_BASELINE)
    # (b) çözümleme
    dec_rows, dec_refs = [], []
    for spec in DECODING_RUNS:
        greedy = load_row(col, spec, "greedy")
        beam = load_row(col, spec, "beam3")
        dec_rows += [greedy, beam]
        dec_refs += [greedy, greedy]
        if spec["run"] == NUCLEUS["run"]:
            nucleus = load_nucleus(col, NUCLEUS)
            dec_rows.append(nucleus)
            dec_refs.append(greedy)
    # (c) attention
    att_rows, att_refs = pair_rows(col, ATTENTION_RUNS, ATTENTION_BASELINE)
    # (d) embedding
    emb_specs = discover_embedding_runs(results_dir)
    emb_rows, emb_refs = pair_rows(col, emb_specs, EMBEDDING_BASELINE)

    # Karşılaştırılabilirlik: tüm koşular aynı referans ve bölme dosyasıyla mı değerlendirildi?
    for key, values in col.hashes.items():
        if len(values) > 1:
            col.warn(f"Koşular farklı `{key}` değerleriyle değerlendirilmiş ({len(values)} farklı değer)")

    nucleus_row = next((r for r in dec_rows if r.method.startswith("nucleus")), None)
    n_seeds = nucleus_row.n_seeds if nucleus_row and nucleus_row.n_seeds else 0

    md: list[str] = ["# Birleşik doğrulama tablosu", ""]
    md.append("Doğrulama kümesi, 1000 görüntü × 5 referans, ham pycocoevalcap ölçeği (yüzde değil), "
              "test kümesi açılmadı. Sayılar `python -m src.results_table` ile aşağıdaki JSON'lardan "
              "okunur, elle yazılmaz; metrikler 4 ondalık. Nucleus örnekleme ana tablonun yerine geçmez, "
              "yalnız (b) çeşitlilik deneyinde raporlanır.")
    md += ["", "**Kaynak dosyalar** (`results/` altında):", ""]
    md += [f"- `{s}`" for s in sorted(col.sources)] or ["- (hiçbiri)"]
    md += ["", "## Ana tablo — 4 model × greedy / beam-3", ""]
    md.append(md_table(MAIN_HEADER, main_table_rows(main_rows), ["l", "l"] + ["r"] * 11))
    md += ["", "Metrikler `val_metrics_{greedy,beam3}.json` (src/evaluate.py); farklı cümle, ms/görüntü ve "
           "checkpoint epoch `val_metrics.json` (src/predict.py, cihaz: " + devices_note(main_rows) + "); "
           "parametre sayısı eğitim `summary.json`. METEOR hesaplanmadıysa \"—\". Koşu eşlemesi: " +
           ", ".join(f"{s['label']} = `{s['run']}`" for s in MAIN_RUNS) + "."]

    md += ["", "## Ablation tabloları", "",
           "Δ CIDEr her tabloda \"ref.\" satırına göre (aynı çözme yöntemiyle) hesaplanır.", ""]
    frac_notes = [config_note(col, repo_root, pair, {"name", "data.train_fraction"}, "Veri miktarı")
                  for pair in DATA_FRACTION_CONFIGS]
    md += ["### (a) Veri miktarı", "",
           "Tek değişen ayar `data.train_fraction` (%25 / %50 / %100, görüntü bazlı alt küme, seed 42); "
           "model ve çözme aynı; " + "; ".join(frac_notes) + ".", ""]
    md.append(ablation_table(frac_rows, frac_refs))
    md += ["", "### (b) Çözümleme yöntemi", "",
           "Tek değişen ayar kod çözme yöntemi (greedy / beam-3 / nucleus p=0.9); aynı checkpoint.", ""]
    md.append(ablation_table(dec_rows, dec_refs, ratio=True))
    if n_seeds:
        md += ["", f"Nucleus satırı {n_seeds} seed'in ortalaması ± örneklem standart sapması (ddof=1); "
               "farklı cümle oranı = farklı altyazı / 1000 görüntü."]
    att_note = config_note(col, repo_root, ATTENTION_CONFIGS, ATTENTION_EXPECTED_DIFF, "Attention")
    md += ["", "### (c) Attention var / yok", "",
           f"Tek değişen ayar kod çözücüde görsel attention (Model 1 → Model 2); {att_note}.", ""]
    md.append(ablation_table(att_rows, att_refs))
    emb_note = config_note(col, repo_root, EMBEDDING_CONFIGS, EMBEDDING_EXPECTED_DIFF, "Embedding")
    md += ["", "### (d) Embedding (GloVe)", "",
           "Tek değişen ayar kelime embedding'inin başlatılması: 300 boyutlu rastgele (`model1_lstm_emb300`) "
           f"vs GloVe-300 (`model1_lstm_glove300*`); {emb_note}.", ""]
    md.append(ablation_table(emb_rows, emb_refs))

    # Uyarılar
    md += ["", "## Tutarlılık kontrolleri", ""]
    md.append(f"Kontrol edilenler: her koşuda split == \"{EXPECTED_SPLIT}\", n_images == {EXPECTED_IMAGES}, "
              f"{EXPECTED_REFS} referans; method/beam_size dosya adıyla uyumlu; predict.py ile evaluate.py "
              "CIDEr'i 4 basamakta aynı; checkpoint epoch summary.json best_epoch ile aynı; nucleus ortalama/std "
              "özet dosyasıyla aynı; tüm koşularda referans ve bölme SHA-256'sı aynı.")
    md += ["", "**Uyarılar:**", ""]
    md += [f"- {w}" for w in col.warnings] or ["- Yok."]
    md += ["", "**Eksik dosyalar** (ilgili satırlar \"henüz yok\" ya da \"—\"):", ""]
    missing = collapse_missing(results_dir, col.missing)
    md += [f"- {m}" for m in missing] or ["- Yok."]
    md.append("")

    csv_rows = [["model", "run", "decoding", "status"] + METRIC_KEYS +
                ["unique_captions", "ms_per_image", "parameters", "checkpoint_epoch"]]
    for r in main_rows:
        csv_rows.append([r.label, r.run, method_label(r.method), "ok" if r.available else MISSING] +
                        [("" if r.metrics.get(k) is None else f"{r.metrics[k]:.4f}") for k in METRIC_KEYS] +
                        ["" if r.unique is None else str(r.unique), "" if r.ms is None else f"{r.ms:.2f}",
                         "" if r.params is None else str(r.params), "" if r.epoch is None else str(r.epoch)])

    console = list(col.warnings) + [f"eksik: {m.replace('`', '')}" for m in missing]
    return "\n".join(md), csv_rows, console


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=repo / "results")
    parser.add_argument("--out-md", type=Path, default=None, help="varsayılan: <results-dir>/val_table.md")
    parser.add_argument("--out-csv", type=Path, default=None, help="varsayılan: <results-dir>/val_table.csv")
    args = parser.parse_args(argv)
    out_md = args.out_md or args.results_dir / "val_table.md"
    out_csv = args.out_csv or args.results_dir / "val_table.csv"

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # Windows konsolunda "≠", "Δ" çökertmesin
    md, csv_rows, console = build(args.results_dir)
    out_md.write_text(md, encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(csv_rows)
    print(f"yazıldı: {out_md}\nyazıldı: {out_csv}")
    for line in console:
        print(f"UYARI: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

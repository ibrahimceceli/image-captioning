# Birleşik doğrulama tablosu

Doğrulama kümesi, 1000 görüntü × 5 referans, ham pycocoevalcap ölçeği (yüzde değil), test kümesi açılmadı. Sayılar `python -m src.results_table` ile aşağıdaki JSON'lardan okunur, elle yazılmaz; metrikler 4 ondalık. Nucleus örnekleme ana tablonun yerine geçmez, yalnız (b) çeşitlilik deneyinde raporlanır.

**Kaynak dosyalar** (`results/` altında):

- `model1_lstm/summary.json`
- `model1_lstm/val_metrics.json`
- `model1_lstm/val_metrics_beam3.json`
- `model1_lstm/val_metrics_greedy.json`
- `model1_lstm_emb300/summary.json`
- `model1_lstm_emb300/val_metrics.json`
- `model1_lstm_emb300/val_metrics_beam3.json`
- `model1_lstm_emb300/val_metrics_greedy.json`
- `model1_lstm_frac25/summary.json`
- `model1_lstm_frac25/val_metrics.json`
- `model1_lstm_frac25/val_metrics_beam3.json`
- `model1_lstm_frac25/val_metrics_greedy.json`
- `model1_lstm_frac50/summary.json`
- `model1_lstm_frac50/val_metrics.json`
- `model1_lstm_frac50/val_metrics_beam3.json`
- `model1_lstm_frac50/val_metrics_greedy.json`
- `model1_lstm_glove300/summary.json`
- `model1_lstm_glove300/val_metrics.json`
- `model1_lstm_glove300/val_metrics_beam3.json`
- `model1_lstm_glove300/val_metrics_greedy.json`
- `model2_attention_lstm/nucleus_p09_seed42/quality_metrics.json`
- `model2_attention_lstm/nucleus_p09_seed42/val_metrics.json`
- `model2_attention_lstm/nucleus_p09_seed43/quality_metrics.json`
- `model2_attention_lstm/nucleus_p09_seed43/val_metrics.json`
- `model2_attention_lstm/nucleus_p09_seed44/quality_metrics.json`
- `model2_attention_lstm/nucleus_p09_seed44/val_metrics.json`
- `model2_attention_lstm/nucleus_p09_summary.json`
- `model2_attention_lstm/val_metrics.json`
- `model2_attention_lstm/val_metrics_beam3.json`
- `model2_attention_lstm/val_metrics_greedy.json`
- `model2_attention_lstm_full_20261104_01/summary.json`
- `model3_transformer/summary.json`
- `model3_transformer/val_metrics.json`
- `model3_transformer/val_metrics_beam3.json`
- `model3_transformer/val_metrics_greedy.json`
- `model4_blip_finetuned/summary.json`
- `model4_blip_finetuned/val_metrics.json`
- `model4_blip_finetuned/val_metrics_beam3.json`
- `model4_blip_finetuned/val_metrics_greedy.json`
- `model4_blip_pretrained/summary.json`
- `model4_blip_pretrained/val_metrics.json`
- `model4_blip_pretrained/val_metrics_beam3.json`
- `model4_blip_pretrained/val_metrics_greedy.json`

## Ana tablo — 4 model (BLIP iki varyant) × greedy / beam-3

| Model | Çözme | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | Farklı cümle (/1000) | ms/görüntü | Parametre | Checkpoint epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model 1 — CNN-LSTM | greedy | 0.4691 | 0.2907 | 0.1725 | 0.1063 | 0.1555 | 0.3718 | 0.2306 | 176 | 15.05 | 5,653,004 | 10 |
| Model 1 — CNN-LSTM | beam-3 | 0.5297 | 0.3368 | 0.2131 | 0.1389 | 0.1590 | 0.3860 | 0.2778 | 164 | 45.96 | 5,653,004 | 10 |
| Model 2 — Attention LSTM | greedy | 0.5638 | 0.3871 | 0.2566 | 0.1698 | 0.1867 | 0.4312 | 0.4409 | 877 | 63.20 | 11,159,565 | 10 |
| Model 2 — Attention LSTM | beam-3 | 0.6170 | 0.4355 | 0.2988 | 0.2030 | 0.1861 | 0.4396 | 0.4697 | 813 | 182.65 | 11,159,565 | 10 |
| Model 3 — Transformer | greedy | 0.6013 | 0.4153 | 0.2818 | 0.1872 | 0.1923 | 0.4441 | 0.4763 | 942 | 53.21 | 5,016,588 | 8 |
| Model 3 — Transformer | beam-3 | 0.6050 | 0.4232 | 0.2925 | 0.1984 | 0.1869 | 0.4418 | 0.4778 | 900 | 160.59 | 5,016,588 | 8 |
| Model 4 — BLIP (fine-tune edilmeden) | greedy | 0.6336 | 0.4839 | 0.3470 | 0.2427 | 0.2040 | 0.4875 | 0.6337 | 808 | 156.37 | 223,971,644 | — |
| Model 4 — BLIP (fine-tune edilmeden) | beam-3 | 0.6553 | 0.4986 | 0.3653 | 0.2621 | 0.2093 | 0.4777 | 0.6833 | 772 | 205.57 | 223,971,644 | — |
| Model 4 — BLIP (fine-tune edilmiş) | greedy | 0.7606 | 0.6034 | 0.4622 | 0.3463 | 0.2797 | 0.5733 | 1.0141 | 969 | 216.94 | 223,971,644 | 2 |
| Model 4 — BLIP (fine-tune edilmiş) | beam-3 | 0.7482 | 0.5973 | 0.4634 | 0.3522 | 0.2910 | 0.5862 | 1.0571 | 978 | 291.60 | 223,971,644 | 2 |

Metrikler `val_metrics_{greedy,beam3}.json` (src/evaluate.py); farklı cümle, ms/görüntü ve checkpoint epoch `val_metrics.json` (src/predict.py, cihaz: cuda); parametre sayısı eğitim `summary.json`. METEOR hesaplanmadıysa "—". Koşu eşlemesi: Model 1 — CNN-LSTM = `model1_lstm`, Model 2 — Attention LSTM = `model2_attention_lstm`, Model 3 — Transformer = `model3_transformer`, Model 4 — BLIP (fine-tune edilmeden) = `model4_blip_pretrained`, Model 4 — BLIP (fine-tune edilmiş) = `model4_blip_finetuned`.

## Ablation tabloları

Δ CIDEr her tabloda "ref." satırına göre (aynı çözme yöntemiyle) hesaplanır.

### (a) Veri miktarı

Tek değişen ayar `data.train_fraction` (%25 / %50 / %100, görüntü bazlı alt küme, seed 42); model ve çözme aynı; `configs/model1.yaml` → `configs/model1_frac25.yaml` farklı anahtarlar: `data.train_fraction`, `name` (diğer tüm ayarlar aynı); `configs/model1.yaml` → `configs/model1_frac50.yaml` farklı anahtarlar: `data.train_fraction`, `name` (diğer tüm ayarlar aynı).

| Koşu | Çözme | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | Δ CIDEr | Farklı cümle (/1000) | ms/görüntü | Epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| %25 (`model1_lstm_frac25`) | greedy | 0.4414 | 0.2571 | 0.1399 | 0.0801 | 0.1419 | 0.3536 | 0.1642 | -0.0664 | 93 | 18.24 | 10 |
| %50 (`model1_lstm_frac50`) | greedy | 0.4994 | 0.2990 | 0.1700 | 0.1019 | 0.1440 | 0.3628 | 0.1943 | -0.0362 | 127 | 16.08 | 12 |
| %100 (`model1_lstm`) | greedy | 0.4691 | 0.2907 | 0.1725 | 0.1063 | 0.1555 | 0.3718 | 0.2306 | ref. | 176 | 15.05 | 10 |
| %25 (`model1_lstm_frac25`) | beam-3 | 0.4890 | 0.2979 | 0.1817 | 0.1175 | 0.1491 | 0.3610 | 0.2178 | -0.0600 | 88 | 77.05 | 10 |
| %50 (`model1_lstm_frac50`) | beam-3 | 0.5020 | 0.2994 | 0.1729 | 0.1042 | 0.1425 | 0.3594 | 0.2086 | -0.0692 | 124 | 64.10 | 12 |
| %100 (`model1_lstm`) | beam-3 | 0.5297 | 0.3368 | 0.2131 | 0.1389 | 0.1590 | 0.3860 | 0.2778 | ref. | 164 | 45.96 | 10 |

### (b) Çözümleme yöntemi

Tek değişen ayar kod çözme yöntemi (greedy / beam-3 / nucleus p=0.9); aynı checkpoint.

| Koşu | Çözme | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | Δ CIDEr | Farklı cümle oranı | ms/görüntü | Epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model 1 — CNN-LSTM (`model1_lstm`) | greedy | 0.4691 | 0.2907 | 0.1725 | 0.1063 | 0.1555 | 0.3718 | 0.2306 | ref. | 0.1760 | 15.05 | 10 |
| Model 1 — CNN-LSTM (`model1_lstm`) | beam-3 | 0.5297 | 0.3368 | 0.2131 | 0.1389 | 0.1590 | 0.3860 | 0.2778 | +0.0472 | 0.1640 | 45.96 | 10 |
| Model 2 — Attention LSTM (`model2_attention_lstm`) | greedy | 0.5638 | 0.3871 | 0.2566 | 0.1698 | 0.1867 | 0.4312 | 0.4409 | ref. | 0.8770 | 63.20 | 10 |
| Model 2 — Attention LSTM (`model2_attention_lstm`) | beam-3 | 0.6170 | 0.4355 | 0.2988 | 0.2030 | 0.1861 | 0.4396 | 0.4697 | +0.0288 | 0.8130 | 182.65 | 10 |
| Model 2 — Attention LSTM (`model2_attention_lstm`) | nucleus p=0.9 | 0.4768 ± 0.0428 | 0.2776 ± 0.0492 | 0.1577 ± 0.0430 | 0.0889 ± 0.0328 | 0.1476 ± 0.0130 | 0.3432 ± 0.0477 | 0.2519 ± 0.0421 | -0.1890 | 0.9920 ± 0.0026 | 45.00 | 10 |

Nucleus satırı 3 seed'in ortalaması ± örneklem standart sapması (ddof=1); farklı cümle oranı = farklı altyazı / 1000 görüntü.

### (c) Attention var / yok

Tek değişen ayar kod çözücüde görsel attention (Model 1 → Model 2); `configs/model1.yaml` → `configs/model2.yaml` farklı anahtarlar: `model.attention_dim`, `model.num_layers`, `model.type`, `name` (diğer tüm ayarlar aynı).

| Koşu | Çözme | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | Δ CIDEr | Farklı cümle (/1000) | ms/görüntü | Epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model 1 — attention yok (`model1_lstm`) | greedy | 0.4691 | 0.2907 | 0.1725 | 0.1063 | 0.1555 | 0.3718 | 0.2306 | ref. | 176 | 15.05 | 10 |
| Model 2 — attention var (`model2_attention_lstm`) | greedy | 0.5638 | 0.3871 | 0.2566 | 0.1698 | 0.1867 | 0.4312 | 0.4409 | +0.2104 | 877 | 63.20 | 10 |
| Model 1 — attention yok (`model1_lstm`) | beam-3 | 0.5297 | 0.3368 | 0.2131 | 0.1389 | 0.1590 | 0.3860 | 0.2778 | ref. | 164 | 45.96 | 10 |
| Model 2 — attention var (`model2_attention_lstm`) | beam-3 | 0.6170 | 0.4355 | 0.2988 | 0.2030 | 0.1861 | 0.4396 | 0.4697 | +0.1919 | 813 | 182.65 | 10 |

### (d) Embedding (GloVe)

Tek değişen ayar kelime embedding'inin başlatılması: 300 boyutlu rastgele (`model1_lstm_emb300`) vs GloVe-300 (`model1_lstm_glove300*`); `configs/model1_emb300.yaml` → `configs/model1_glove300.yaml` farklı anahtarlar: `embeddings.dim`, `embeddings.freeze`, `embeddings.path`, `name` (diğer tüm ayarlar aynı).

| Koşu | Çözme | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | Δ CIDEr | Farklı cümle (/1000) | ms/görüntü | Epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model1_lstm_emb300 | greedy | 0.5150 | 0.3252 | 0.2004 | 0.1260 | 0.1565 | 0.3874 | 0.2365 | ref. | 154 | 16.37 | 12 |
| model1_lstm_glove300 | greedy | 0.5173 | 0.3333 | 0.2054 | 0.1312 | 0.1603 | 0.3865 | 0.2610 | +0.0245 | 217 | 14.93 | 12 |
| model1_lstm_emb300 | beam-3 | 0.5251 | 0.3340 | 0.2117 | 0.1384 | 0.1567 | 0.3872 | 0.2843 | ref. | 168 | 51.65 | 12 |
| model1_lstm_glove300 | beam-3 | 0.5431 | 0.3536 | 0.2278 | 0.1499 | 0.1606 | 0.3912 | 0.2919 | +0.0076 | 212 | 51.36 | 12 |

## Tutarlılık kontrolleri

Kontrol edilenler: her koşuda split == "val", n_images == 1000, 5 referans; method/beam_size dosya adıyla uyumlu; predict.py ile evaluate.py CIDEr'i 4 basamakta aynı; checkpoint epoch summary.json best_epoch ile aynı; nucleus ortalama/std özet dosyasıyla aynı; tüm koşularda referans ve bölme SHA-256'sı aynı.

**Uyarılar:**

- Yok.

**Eksik dosyalar** (ilgili satırlar "henüz yok" ya da "—"):

- Yok.

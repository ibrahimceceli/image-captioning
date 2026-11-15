[English](README.md) · **Türkçe**

# Derin Öğrenme Tabanlı Görüntü Altyazılama

**CNN-LSTM, Dikkat Mekanizması, Transformer ve Önceden Eğitilmiş Modellerin Karşılaştırmalı Analizi**

Bu proje bir fotoğraftan onu anlatan İngilizce bir cümle üretir. Flickr8k üzerinde (8091 görüntü, görüntü başına 5 altyazı) dört model karşılaştırılır: Show-and-Tell tarzı CNN-LSTM (Model 1), Bahdanau attention'lı LSTM (Model 2), Transformer decoder (Model 3) ve önceden eğitilmiş BLIP (Model 4; fine-tune edilmeden ve fine-tune edilerek). Model 1–3 aynı bölmeyi, aynı sözlüğü ve aynı dondurulmuş ResNet50 öznitelik önbelleğini kullanır. Karşılaştırma doğruluk (BLEU, METEOR, ROUGE-L, CIDEr), cümle çeşitliliği, hesaplama maliyeti ve yorumlanabilirlik (attention haritaları) üzerinden yapılır.

## Sonuçların özeti

Doğrulama kümesi: 1000 görüntü × 5 referans. Metrikler ham pycocoevalcap ölçeğindedir (yüzde değil). Sayılar `results/val_table.md` / `results/val_table.csv` dosyasından alınmıştır; o tablo `python -m src.results_table` ile `results/` altındaki JSON'lardan üretilir, elle yazılmaz. Parametre sayısına paylaşılan, dondurulmuş ResNet50 (23,508,032) dahil değildir.

| Model | Çözme | BLEU-4 | METEOR | ROUGE-L | CIDEr | Farklı cümle (/1000) | Parametre |
|---|---|---:|---:|---:|---:|---:|---:|
| Model 1 — CNN-LSTM | greedy | 0.1063 | 0.1555 | 0.3718 | 0.2306 | 176 | 5,653,004 |
| Model 1 — CNN-LSTM | beam-3 | 0.1389 | 0.1590 | 0.3860 | 0.2778 | 164 | 5,653,004 |
| Model 2 — Attention LSTM | greedy | 0.1698 | 0.1867 | 0.4312 | 0.4409 | 877 | 11,159,565 |
| Model 2 — Attention LSTM | beam-3 | 0.2030 | 0.1861 | 0.4396 | 0.4697 | 813 | 11,159,565 |
| Model 3 — Transformer | greedy | 0.1872 | 0.1923 | 0.4441 | 0.4763 | 942 | 5,016,588 |
| Model 3 — Transformer | beam-3 | 0.1984 | 0.1869 | 0.4418 | 0.4778 | 900 | 5,016,588 |
| Model 4 — BLIP (fine-tune edilmeden) | greedy | 0.2427 | 0.2040 | 0.4875 | 0.6337 | 808 | 223,971,644 |
| Model 4 — BLIP (fine-tune edilmeden) | beam-3 | 0.2621 | 0.2093 | 0.4777 | 0.6833 | 772 | 223,971,644 |
| Model 4 — BLIP (fine-tune'lu) | greedy | 0.3463 | 0.2797 | 0.5733 | 1.0141 | 969 | 223,971,644 |
| Model 4 — BLIP (fine-tune'lu) | beam-3 | 0.3522 | 0.2910 | 0.5862 | 1.0571 | 978 | 223,971,644 |

### Test kümesi sonuçları

Test kümesi: 1000 görüntü × 5 referans. Değerler `results/model1_lstm/test/`, `results/model2_attention_lstm/test/`, `results/model3_transformer/test/` ve `results/model4_blip_pretrained/test/` altındaki `test_metrics*.json` dosyalarından alınmıştır (Model 1–3 için `predict --split test --allow-test`, tümü için `evaluate --split test --allow-test --meteor`, ham pycocoevalcap ölçeği). BLIP'in iki varyantı da test kümesinde değerlendirilmiştir: fine-tune edilmeden `results/model4_blip_pretrained/test/`, fine-tune'lu (Kaggle'da eğitilen epoch 2 checkpoint'i, `python -m src.blip evaluate --split test --allow-test`) `results/model4_blip_finetuned/test/`.

| Model | Çözme | BLEU-4 | METEOR | ROUGE-L | CIDEr | Farklı cümle (/1000) | Parametre |
|---|---|---:|---:|---:|---:|---:|---:|
| Model 1 — CNN-LSTM | greedy | 0.1020 | 0.1541 | 0.3740 | 0.2273 | 171 | 5,653,004 |
| Model 1 — CNN-LSTM | beam-3 | 0.1302 | 0.1571 | 0.3818 | 0.2653 | 155 | 5,653,004 |
| Model 2 — Attention LSTM | greedy | 0.1715 | 0.1893 | 0.4446 | 0.4446 | 864 | 11,159,565 |
| Model 2 — Attention LSTM | beam-3 | 0.2026 | 0.1893 | 0.4503 | 0.4789 | 817 | 11,159,565 |
| Model 3 — Transformer | greedy | 0.1859 | 0.1938 | 0.4464 | 0.4535 | 941 | 5,016,588 |
| Model 3 — Transformer | beam-3 | 0.1956 | 0.1892 | 0.4491 | 0.4704 | 885 | 5,016,588 |
| Model 4 — BLIP (fine-tune edilmeden) | greedy | 0.2482 | 0.2114 | 0.5017 | 0.6323 | 790 | 223,971,644 |
| Model 4 — BLIP (fine-tune edilmeden) | beam-3 | 0.2648 | 0.2125 | 0.4876 | 0.6744 | 786 | 223,971,644 |
| Model 4 — BLIP (fine-tune'lu) | greedy | 0.3504 | 0.2835 | 0.5845 | 1.0119 | 961 | 223,971,644 |
| Model 4 — BLIP (fine-tune'lu) | beam-3 | 0.3520 | 0.2923 | 0.5859 | 1.0456 | 975 | 223,971,644 |

Ablation tabloları (veri miktarı %25 / %50 / %100, çözme yöntemi greedy / beam-3 / nucleus, attention var / yok, rastgele vs GloVe embedding) ve tutarlılık kontrolleri [results/val_table.md](results/val_table.md) içindedir. Deney, kalitatif analiz ve doğrulama raporları: [results/model1_data_fraction.md](results/model1_data_fraction.md), [results/model1_glove.md](results/model1_glove.md), [report/model2_durum_20261108.md](report/model2_durum_20261108.md), [report/model2_b_tamamlayici_20261112.md](report/model2_b_tamamlayici_20261112.md), [report/kalitatif_karsilastirma_4model.md](report/kalitatif_karsilastirma_4model.md), [report/demo_dogrulama.md](report/demo_dogrulama.md).

## Repo yapısı

```
image-captioning/
├── README.md, README.tr.md
├── CONTRIBUTING.md, CONTRIBUTING.tr.md   # ekip içi çalışma kuralları
├── requirements.txt
├── configs/                     # base.yaml (ortak, kilitli ayarlar) + model ve deney config'leri
│   ├── base.yaml
│   ├── model1.yaml, model1_frac25.yaml, model1_frac50.yaml
│   ├── model1_emb300.yaml, model1_glove300.yaml
│   ├── model2.yaml
│   └── model3.yaml
├── src/
│   ├── data/                    # veri hattı
│   │   ├── artifacts/           # flickr8k_split.json, vocab.json (repoda, yeniden üretilmez)
│   │   ├── prepare.py           # bölme + sözlük + EDA (çalıştırmanız gerekmez)
│   │   ├── cache_features.py    # ResNet50 layer4 öznitelik önbelleği
│   │   ├── features.py          # kilitli ön işleme + dondurulmuş encoder
│   │   ├── dataset.py, vocab.py, text.py, split.py, config.py, cache.py, eda.py
│   │   └── glove.py             # GloVe embedding matrisi
│   ├── models/
│   │   ├── lstm.py              # Model 1
│   │   ├── attention_lstm.py    # Model 2
│   │   ├── transformer.py       # Model 3
│   │   └── attention_captioner.py   # Model 2 için generate(image) adaptörü
│   ├── blip.py                  # Model 4: BLIP fine-tuning ve değerlendirme
│   ├── train.py                 # ortak eğitim (Model 1–3)
│   ├── decode.py                # greedy, beam, nucleus
│   ├── predict.py               # checkpoint → doğrulama cümleleri + CIDEr
│   ├── evaluate.py              # BLEU-1..4, METEOR, ROUGE-L, CIDEr
│   ├── results_table.py         # birleşik doğrulama tablosu
│   └── visualize_attention.py   # kelime başına attention haritası (kütüphane fonksiyonu)
├── app/                         # Gradio demosu (app.py, captioners.py, config.yaml, README.md)
├── tests/                       # pytest testleri
├── results/                     # metrik JSON'ları, cümleler, kayıp eğrileri, tablolar (test sonuçları <ad>/test/ altında)
│   ├── eda/
│   ├── model1_lstm/, model1_lstm_frac25/, model1_lstm_frac50/
│   ├── model1_lstm_emb300/, model1_lstm_glove300/
│   ├── model2_attention_lstm/                    # Model 2 tahmin ve metrikleri, nucleus, attention örnekleri
│   ├── model2_attention_lstm_full_20261104_01/   # Model 2 eğitim günlükleri
│   ├── model2_val_study_20261112/                # Model 2 beam-5 ve 40 görüntülük hata incelemesi
│   ├── model3_transformer/
│   ├── model4_blip_pretrained/, model4_blip_finetuned/
│   ├── qualitative_4models/                      # dört modelin 40 görüntüdeki cümleleri
│   ├── val_table.md, val_table.csv
│   └── model1_data_fraction.md, model1_glove.md
├── report/                      # deney, kalitatif analiz ve demo doğrulama notları
└── notebooks/                   # BLIP fine-tuning için Kaggle notebook'u
```

Repoda **olmayanlar** (`.gitignore`): `data/` (görüntüler, altyazılar, öznitelik önbelleği, GloVe), `checkpoints/` (`best.pt`, `last.pt`, BLIP ağırlıkları), eğitim logları (`results/*_train.log`) ve `kaggle.json`.

## Modeller

| Model | Tanım | Kod | Config | Parametre |
|---|---|---|---|---:|
| Model 1 — CNN-LSTM | Show-and-Tell tarzı alt sınır: 49 bölgenin ortalaması (2048) LSTM'in başlangıç durumuna verilir; embedding 256, LSTM 512, 1 katman | `src/models/lstm.py` | `configs/model1.yaml` | 5,653,004 |
| Model 2 — Attention LSTM | Show-Attend-and-Tell tarzı: 7×7 = 49 bölge üzerinde her kelimede Bahdanau attention | `src/models/attention_lstm.py` | `configs/model2.yaml` | 11,159,565 |
| Model 3 — Transformer | 49 bölge → d_model 256; 3 katman, 8 kafa, self- + cross-attention decoder | `src/models/transformer.py` | `configs/model3.yaml` | 5,016,588 |
| Model 4 — BLIP | `Salesforce/blip-image-captioning-base` (COCO'da altyazılama için eğitilmiş); fine-tune edilmeden ve Flickr8k'da fine-tune edilerek | `src/blip.py` | `src.blip` komut satırı | 223,971,644 |


Parametre sayıları eğitim `summary.json` dosyalarındandır ve paylaşılan ResNet50 encoder'ı içermez. Embedding boyutu 300 olan deney koşuları (`emb300`, `glove300`) 5,856,284 parametredir.

## Kurulum

Python 3.12 ile denenmiştir. Model 1–3 için CUDA destekli GPU önerilir; önbellek ve eğitim CPU'da da çalışır ama yavaştır.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

- GPU için PyTorch'u CUDA sürümüyle kurun, örneğin CUDA 12.4: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` ([pytorch.org/get-started](https://pytorch.org/get-started/locally/)). Önbellek bu projede `torch 2.6.0+cu124` ve `torchvision 0.21.0+cu124` ile üretildi.
- METEOR Java ister (`java -version` çalışmalı). Java yoksa BLEU, ROUGE-L ve CIDEr yine hesaplanır. Türkçe yerel ayarlı Windows'ta bilinen sorun ve çözümü: [Değerlendirme](#değerlendirme).
- Test edilmiş paket sürümleri Python 3.12.3 ile doğrulanıp [requirements.txt](requirements.txt) içinde sabitlenmiştir ([demo ve ortam doğrulama raporu](report/demo_dogrulama.md)).
- Kurulumu doğrulamak için: `python -m pytest tests` (`pytest==9.1.1`, [requirements.txt](requirements.txt) içinde yer alır).

## Veri

Flickr8k'nın Kaggle sürümü kullanılır: [adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k). İçinde 8091 görüntü (`Images/`) ve 40.455 altyazı (`captions.txt`, başlıkları `image,caption`) vardır.

Zip olarak indirip açın ya da Kaggle API kullanın. API anahtarı `kaggle.json` kullanıcı klasöründe durur (`~/.kaggle/kaggle.json`, Windows'ta `C:\Users\<ad>\.kaggle\kaggle.json`) ve repoya konmaz.

```bash
pip install kaggle
kaggle datasets download -d adityajn105/flickr8k -p data/flickr8k --unzip
```

Beklenen yapı:

```
data/flickr8k/
├── Images/          # 8091 .jpg
└── captions.txt     # image,caption
```

Sonuçlarda kullanılan `captions.txt` dosyasının SHA-256'sı `e6cf5e8d8025d25b50a4945c38580d96a940b78644f31a90368dfcdca3c7e036` (`results/model1_lstm/reference_source.json`). Bölme dosyası ve sözlük repoda hazırdır, yeniden üretilmez.

## Hızlı başlangıç

Kurulum ve veriden sonra Model 1'i baştan sona çalıştırma:

```bash
# 1. ResNet50 öznitelik önbelleği (GPU'da yaklaşık 1 dakika, ~1,6 GB)
python -m src.data.cache_features

# 2. Eğitim (RTX 3050'de yaklaşık 6 dakika, erken durdurmayla)
python -m src.train --config configs/model1.yaml

# 3. Doğrulama cümleleri ve CIDEr: greedy ve beam-3
python -m src.predict --checkpoint checkpoints/model1_lstm/best.pt --method greedy
python -m src.predict --checkpoint checkpoints/model1_lstm/best.pt --method beam --beam-size 3
```

Tüm sonuçların yeniden üretimi bir sonraki bölümdedir.

## Tekrar çalıştırma

Bu bölüm `results/` altındaki her sayının nasıl üretildiğini sırasıyla verir. Komutlar depo kökünden çalıştırılır. Döngüler bash sözdizimindedir (Linux/macOS ya da Git Bash); PowerShell'de komutları tek tek çalıştırın.

**Üzerine yazma kuralları.** `src.train` ve `src.predict` (varsayılan klasörde) `results/<ad>/` altındaki commit'li dosyaların üzerine yazar; sonra `git diff --stat results/` ile farkı görün, geri dönmek için `git checkout -- results/`. `src.evaluate --output` ve `src.predict --output-dir` var olan dosyanın üzerine **yazmaz**: yerinde yeniden puanlamak için önce eski `val_metrics_*.json` dosyasını silin ya da farklı bir yol verin.

**Süreler.** Eğitim süreleri `results/<ad>/summary.json` içindeki `train_minutes` alanıdır (CUDA, AMP); Model 1 koşusu RTX 3050'de alınmıştır, başka GPU'da süreler değişir. Tahmin süreleri `val_metrics.json` içindeki `ms_per_image` × 1000 görüntüdür (yalnız çözme; model ve önbellek yükleme hariç).

### 1. Öznitelik önbelleği

```bash
python -m src.data.cache_features
```

Çıktı `data/features/` (~1,6 GB). RTX 3050'de yaklaşık 50 saniye. Tüm Model 1–3 koşularının ön koşuludur.

### 2. Model 1 ve deney koşularının eğitimi

```bash
python -m src.train --config configs/model1.yaml            # model1_lstm, ~5,7 dk
python -m src.train --config configs/model1_frac25.yaml     # model1_lstm_frac25, ~2,3 dk
python -m src.train --config configs/model1_frac50.yaml     # model1_lstm_frac50, ~4,8 dk
python -m src.train --config configs/model1_emb300.yaml     # model1_lstm_emb300, ~7,0 dk
python -m src.train --config configs/model1_glove300.yaml   # model1_lstm_glove300, ~6,8 dk (GloVe dosyası gerekir)
```

GloVe koşusu `data/glove/glove.6B.300d.txt` dosyasını ister (GloVe 6B, 300 boyut, 400.000 satır, ~1 GB; repoda yoktur). Resmi kaynak [nlp.stanford.edu/projects/glove](https://nlp.stanford.edu/projects/glove/). Bu projede resmi sunucuya erişilemediği için Kaggle kopyası [`thanakomsn/glove6b300dtxt`](https://www.kaggle.com/datasets/thanakomsn/glove6b300dtxt) kullanıldı ve resmi dosyayla tutarlılığı doğrulandı (ayrıntı `results/model1_glove.md`). Dosya hangi kaynaktan gelirse gelsin `data/glove/glove.6B.300d.txt` yolunda durmalıdır. Kapsama istatistikleri eğitim sırasında `results/model1_lstm_glove300/embedding_stats.json` dosyasına yazılır.

### 3. Model 2 eğitimi ya da checkpoint'i

Model 2 checkpoint'i (`best.pt`) repoda yoktur; ekip içinde paylaşılır (Üye B). Sonuçları yeniden puanlamak ya da demoda kullanmak için şu yola koyun:

```
checkpoints/model2_attention_lstm_full_20261104_01/best.pt
```

Baştan eğitmek için (CUDA'da ~19,4 dk, 11 epoch):

```bash
python -m src.train --config configs/model2.yaml --name model2_attention_lstm_full_20261104_01
```

`--name` yalnız eğitim çıktılarının klasörünü belirler (`checkpoints/<ad>/`, `results/<ad>/`). `src.predict` checkpoint'in içindeki config adını kullandığı için Model 2'nin tahmin ve metrik dosyaları `results/model2_attention_lstm/` altına yazılır.

### 4. Tahmin: greedy, beam-3 ve nucleus

Her koşu için `results/<ad>/val_captions_{greedy,beam3}.json` ve `val_metrics.json` (CIDEr, farklı cümle sayısı, ms/görüntü):

```bash
for run in model1_lstm model1_lstm_frac25 model1_lstm_frac50 model1_lstm_emb300 model1_lstm_glove300 model2_attention_lstm_full_20261104_01; do
  python -m src.predict --checkpoint checkpoints/$run/best.pt --method greedy
  python -m src.predict --checkpoint checkpoints/$run/best.pt --method beam --beam-size 3
done
```

Model 2 nucleus çeşitlilik deneyi (top-p 0.9, seed 42 / 43 / 44, her seed ayrı klasöre):

```bash
for seed in 42 43 44; do
  python -m src.predict --checkpoint checkpoints/model2_attention_lstm_full_20261104_01/best.pt \
    --method nucleus --top-p 0.9 --seed $seed \
    --output-dir results/model2_attention_lstm/nucleus_p09_seed$seed
done
```

Model 1 için greedy bir koşu ~15 sn, beam-3 ~46 sn; Model 2 için greedy ~63 sn, beam-3 ~3 dk, nucleus seed başına ~45 sn sürmüştür.

### 5. Değerlendirme (METEOR dahil)

Kayıtlı cümle dosyaları GPU ve checkpoint olmadan puanlanır. Türkçe yerel ayarlı Windows'ta METEOR'dan önce şu değişkeni ayarlayın ([neden](#değerlendirme)):

```bash
export JAVA_TOOL_OPTIONS="-Duser.language=en -Duser.country=US"     # PowerShell: $env:JAVA_TOOL_OPTIONS = "-Duser.language=en -Duser.country=US"
```

```bash
for run in model1_lstm model1_lstm_frac25 model1_lstm_frac50 model1_lstm_emb300 model1_lstm_glove300 model2_attention_lstm; do
  python -m src.evaluate --predictions results/$run/val_captions_greedy.json --model-name $run \
    --meteor --output results/$run/val_metrics_greedy.json
  python -m src.evaluate --predictions results/$run/val_captions_beam3.json --model-name $run \
    --method beam --beam-size 3 --meteor --output results/$run/val_metrics_beam3.json
done

for seed in 42 43 44; do
  d=results/model2_attention_lstm/nucleus_p09_seed$seed
  python -m src.evaluate --predictions $d/val_captions_nucleus0.9.json --model-name model2_attention_lstm \
    --method nucleus --meteor --output $d/quality_metrics.json
done
```

Nucleus ortalaması ve örneklem standart sapması (ddof=1) birleşik tabloda seed klasörlerinden hesaplanır. `results/model2_attention_lstm/nucleus_p09_summary.json` Üye B'nin deney özetidir (METEOR içermez); tablo yalnız onunla çapraz kontrol yapar, bu dosyayı üreten betik repoda yoktur.

### 6. Birleşik tablo

```bash
python -m src.results_table
```

`results/val_table.md` ve `results/val_table.csv` yeniden yazılır. Tablonun sonundaki "Tutarlılık kontrolleri" bölümünde uyarı olmamalıdır. Deney yorum dosyaları (`results/model1_data_fraction.md`, `results/model1_glove.md`) elle yazılmış metin içerir ve bu komutla üretilmez.

### 7. Demo

```bash
python -m app.app
```

Model 1–3 checkpoint'leri yukarıdaki yollardaysa demoda çalışır. BLIP kutusu varsayılan olarak fine-tune edilmemiş modeli Hugging Face'ten yükler; fine-tune'lu modeli göstermek için `app/config.yaml` içindeki BLIP `model_id` alanını `checkpoints/model4_blip/best` yapın ve etiketi güncelleyin.

### 8. Model 3 ve BLIP

Önbellek hazır olduğunda Model 3 için depo kökünden (CUDA'da ~8,3 dk, 11 epoch, en iyi epoch 8):

```bash
python -m src.train --config configs/model3.yaml
python -m src.predict --checkpoint checkpoints/model3_transformer/best.pt --method greedy
python -m src.predict --checkpoint checkpoints/model3_transformer/best.pt --method beam --beam-size 3
python -m src.evaluate --predictions results/model3_transformer/val_captions_greedy.json --model-name model3_transformer --output results/model3_transformer/val_metrics_greedy.json
python -m src.evaluate --predictions results/model3_transformer/val_captions_beam3.json --model-name model3_transformer --method beam --beam-size 3 --output results/model3_transformer/val_metrics_beam3.json
python -m src.results_table
```

BLIP iki koşuyla değerlendirilir (fine-tuning için GPU gerekir; bu projede Kaggle T4 üzerinde [notebook](notebooks/member_c_blip_kaggle.ipynb) ile yapıldı, 5 epoch ~4 saat, en iyi checkpoint epoch 2):

```bash
python -m src.blip pretrained --images-dir data/flickr8k/Images --captions data/flickr8k/captions.txt --output results/model4_blip_pretrained
python -m src.blip finetune --images-dir data/flickr8k/Images --captions data/flickr8k/captions.txt --output results/model4_blip_finetuned --checkpoint checkpoints/model4_blip --epochs 5
```

İki koşu da greedy ve beam-3 tahminlerini, metrikleri, parametre sayısını, çıkarım süresini ve çalışan ortamın sürümlerini (`environment.txt`) yazar. Fine-tune'lu checkpoint (`checkpoints/model4_blip/best`, ~900 MB) repoda yoktur. METEOR bu komutlarla hesaplanmaz; [Değerlendirme](#değerlendirme) adımıyla ayrıca hesaplanır.

### 9. Test kümesi (final, bir kez)

Model seçimi dondurulduktan sonra her model için greedy ve beam-3 koşuları bir kez çalıştırılır:

```bash
python -m src.predict --checkpoint checkpoints/<ad>/best.pt --method greedy --split test --allow-test --output-dir results/<ad>/test
python -m src.predict --checkpoint checkpoints/<ad>/best.pt --method beam --beam-size 3 --split test --allow-test --output-dir results/<ad>/test
python -m src.evaluate --predictions results/<ad>/test/test_captions_greedy.json --model-name <ad> --split test --allow-test --meteor --output results/<ad>/test/test_metrics_greedy.json
python -m src.evaluate --predictions results/<ad>/test/test_captions_beam3.json --model-name <ad> --method beam --beam-size 3 --split test --allow-test --meteor --output results/<ad>/test/test_metrics_beam3.json
```

Model 1 (`model1_lstm`), Model 2 (`checkpoints/model2_attention_lstm_full_20261104_01/best.pt` → `results/model2_attention_lstm/test/`, `--model-name model2_attention_lstm`), Model 3 (`model3_transformer`) ve BLIP'in iki varyantı (`results/model4_blip_pretrained/test/`, `results/model4_blip_finetuned/test/`) test değerlendirmeleri tamamlanarak `results/<ad>/test/` altına kaydedilmiştir. Fine-tune'lu BLIP için checkpoint klasörü (`checkpoints/model4_blip/best`, repoda yok) ile:

```bash
python -m src.blip evaluate --source checkpoints/model4_blip/best --checkpoint-epoch 2 --model-name model4_blip_finetuned --images-dir data/flickr8k/Images --captions data/flickr8k/captions.txt --output results/model4_blip_finetuned/test --split test --allow-test --meteor
```

`python -m src.results_table` yalnız doğrulama tablosunu ürettiğinden test metrikleri doğrudan `results/<ad>/test/test_metrics*.json` dosyalarından raporlanır.

## Sabit kararlar

Aşağıdaki kararlar proje boyunca sabittir; tüm modeller ve deneyler bunlara uyar.

- **Ortak veri:** Model 1–3 aynı bölmeyi, aynı sözlüğü ve aynı ResNet50 önbelleğini kullanır. BLIP kendi görüntü işlemcisini ve tokenizer'ını kullanır.
- **Bölme:** görüntü bazlı, train 6000 / val 1000 / test 1000; bir kez üretilir, değiştirilmez.
- **Metin:** küçük harf, noktalama silinir. Sözlük yalnız train altyazılarından, minimum kelime sıklığı 5. Token'lar `<pad>`, `<start>`, `<end>`, `<unk>`.
- **Uzunluk:** en fazla 34 token, `<start>` ve `<end>` dahil. Pad sağa eklenir. Üretim `<start>` ile başlar, `<end>` token'ında ya da 34 token'da durur.
- **Görüntü (Model 1–3):** kısa kenar 256, merkezden 224×224 kırpma, ImageNet normalizasyonu; dondurulmuş ResNet50 (`IMAGENET1K_V1`) layer4 çıktısı, 2048×7×7.
- **Seed:** 42.
- **Eğitim:** teacher forcing, pad yok sayılan çapraz entropi, gradyan kırpma 5, AMP. Erken durdurma doğrulama kaybıyla (sabır 5), `best.pt` seçimi greedy doğrulama CIDEr'iyle.
- **Metrikler:** düz cümle üzerinden; `<start>`, `<end>`, `<pad>` ve `<unk>` silinir, çıktı ve referanslar küçük harfe çevrilir, noktalama silinir. Ana tablo her model için greedy ve beam-3; nucleus ayrı bir çeşitlilik deneyidir.
- **Arayüz:** her model dışarıya `generate(image) -> caption` verir. Eğitim önbellekten okur, demo ham görüntü alır (`app/captioners.py`, `src/models/attention_captioner.py`).
- **Test kümesi** yalnız final tablo için bir kez kullanılır.
- **Deneyler:** her deneyde tek ayar değişir ve değişiklik deney config'inde durur.
- Veri, önbellek, checkpoint ve `kaggle.json` repoya konmaz.

## Ayrıntılar

### Veri hattı

- Bölme: `src/data/artifacts/flickr8k_split.json`. Train 6000, val 1000, test 1000; 91 görüntü `unused` içindedir ve eğitime girmez.
- Sözlük: `src/data/artifacts/vocab.json`, 2572 token (2568 kelime + 4 özel token). Sıklığı 5'in altındaki kelime `<unk>` olur.
- Yollar `configs/base.yaml` içindedir. Yerel veri `data/flickr8k/Images` ve `data/flickr8k/captions.txt`.
- `token_ids` uzunluğu 34'tür: `<start>`, kelimeler, `<end>`, sağda `<pad>`. Eğitim `build_dataloader(split)` ile okur. Önbellek `image_id` (dosya adı) ile eşlenir.
- Veri istatistikleri: `results/eda/summary.json`, `results/eda/sentence_lengths.png`, `results/eda/word_frequencies.png`.

#### Veri hazırlama betiği

```bash
python -m src.data.prepare
```

Bölme dosyası, sözlük ve `results/eda/` altındaki dosyalar bu betikle üretildi. Normal kullanımda **çalıştırılması gerekmez**. Betik komut satırı seçeneği almaz; `--help` verilse bile çalışır.

- Betik `flickr8k_split.json` ve `vocab.json` dosyalarının **üzerine yazar**. Seed 42 sabit olduğu için aynı veriyle aynı sonucu verir. Çalıştırdıktan sonra `git status` bu iki dosyada değişiklik gösteriyorsa veri farklıdır: değişikliği commit etmeyin, `git checkout -- src/data/artifacts` ile geri alın.
- Ne yapar: görüntü ve altyazıları eşleştirir, metin temizliğini sınar, bölmeyi ve sözlüğü kurar, EDA grafiklerini ve `results/eda/summary.json` dosyasını yazar, train ve val için örnek batch'i denetler (şekil, `<start>`, sağa pad, kümeler arası sızıntı).

### Öznitelik önbelleği

Model 1–3 görüntü yerine dondurulmuş ResNet50 layer4 çıktısını okur. Dönüşüm ve encoder `src/data/features.py` içindedir; demo aynı modülü `extract_image` ile kullanır.

- Çıktılar `data/features/` altındadır: `resnet50_layer4.npy` (fp16, `[8091, 2048, 7, 7]`, yaklaşık 1,6 GB), `resnet50_layer4_index.json` (`image_id -> satır`) ve `resnet50_layer4.meta.json` (ağırlık, dtype, şekil, sürümler). Yollar `configs/base.yaml` içindeki `features_path` ve `features_index_path` alanlarıdır.
- Seçenekler: `--images-dir`, `--out-dir`, `--batch-size`, `--limit` (deneme için ilk N görüntü), `--device`, `--num-workers`, `--no-amp`. GPU'da autocast kullanılır, kayıt her zaman fp16'dır.
- Kaggle'da girdi salt okunurdur, çıktıyı çalışma klasörüne yazın:

```bash
python -m src.data.cache_features --images-dir /kaggle/input/flickr8k/Images --out-dir /kaggle/working/features
```

- Eğitim: `build_dataloader("train", use_features=True)` batch'e `features` (float32, `[B, 2048, 7, 7]`) ekler. Önbellek başka klasördeyse `src.train` / `src.predict` için `--features-dir` verin.
- Doğrulama: `build_eval_dataloader("val")` görüntü başına bir örnek (`image_id`, `features`) verir. `load_references("val")` her görüntünün 5 temiz altyazısını döndürür. İkisi de test kümesini yalnız `allow_test=True` ile açar.

### Eğitim

```bash
python -m src.train --config configs/model1.yaml
```

- Model 1, 2 ve 3 aynı betikle eğitilir; her modelin ayarları `configs/` altındaki kendi dosyasındadır, ortak ayarlar `configs/base.yaml` içindedir.
- Önbellekten okur; teacher forcing, pad yok sayılan çapraz entropi, gradyan kırpma 5, AMP (GPU'da), seed 42.
- Her epoch doğrulama kaybı hesaplanır. Greedy CIDEr her `cider_every` epoch'ta, kayıp iyileştiğinde ve son epoch'ta hesaplanır. Erken durdurma `early_stopping.monitor` (varsayılan `val_loss`, sabır 5), `best.pt` seçimi `evaluation.select_by` (varsayılan `val_cider`) ile yapılır.
- Çıktılar: `checkpoints/<ad>/best.pt` ve `last.pt` (repoya girmez), `results/<ad>/history.json`, `loss_curve.png`, `summary.json`, `val_captions_best.json`.
- Seçenekler: `--name`, `--max-epochs`, `--limit-batches` (deneme), `--features-dir` (Kaggle), `--num-workers`, `--device`, `--resume` (`last.pt`'den devam).
- Veri miktarı deneyi config'te `data.train_fraction` ile yapılır (ör. `0.25`); alt küme görüntü bazlıdır ve seed ile sabittir. GloVe başlatma config'in üst düzey `embeddings` bölümüyle açılır (`src/data/glove.py`).
- Eğitim sırasındaki doğrulama CIDEr'i toplu greedy ile hesaplanır (`src/decode.py` ile aynı kurallar); final tablo `src.predict` ve `src.evaluate` ile üretilir.
- Yeni model `src/train.py` içindeki `MODELS` sözlüğüne eklenir. Arayüz: `forward(features, caption_inputs) -> (logits, attention veya None)`.

### Deneyler

Her deneyde referans config'ten tek ayar değişir.

| Deney | Config | Değişen ayar |
|---|---|---|
| Veri miktarı | `configs/model1_frac25.yaml`, `configs/model1_frac50.yaml` | `data.train_fraction` (`configs/model1.yaml`'a göre) |
| Embedding (kontrol) | `configs/model1_emb300.yaml` | `model.embedding_dim: 300`, rastgele başlatma |
| Embedding (GloVe) | `configs/model1_glove300.yaml` | `emb300` + GloVe başlatma (`embeddings` bölümü, `freeze: false`) |
| Çözme yöntemi | — (aynı checkpoint) | greedy / beam-3 / nucleus p=0.9 |
| Attention var / yok | `configs/model1.yaml` → `configs/model2.yaml` | decoder'da görsel attention |

GloVe karşılaştırması `emb300` ile `glove300` arasındadır; 256 boyutlu `model1_lstm` yalnız referanstır.

### Cümle üretimi

```bash
python -m src.predict --checkpoint checkpoints/model1_lstm/best.pt --method greedy
python -m src.predict --checkpoint checkpoints/model1_lstm/best.pt --method beam --beam-size 3
```

- Cümleler `src/decode.py` içindeki `decode()` ile üretilir. Çıktılar `results/<ad>/` altındadır: `val_captions_<yöntem>.json` (`{image_id: cümle}`) ve `val_metrics.json` (yöntem başına CIDEr, farklı cümle sayısı, görüntü başına çıkarım süresi). `<ad>` checkpoint'teki config adıdır.
- Seçenekler: `--method {greedy,beam,nucleus}`, `--beam-size`, `--top-p`, `--seed`, `--limit`, `--features-dir`, `--output-dir` (ayrı klasör; var olan dosyanın üzerine yazmaz), `--device`, `--split`, `--allow-test`.
- Test kümesi yalnız final tablo için `--split test --allow-test` ile bir kez açılır.
- Model 2 attention haritaları `src/visualize_attention.py` içindeki `visualize_attention()` fonksiyonuyla çizilir (komut satırı arayüzü yok). Örnekler: `results/model2_attention_lstm/attention_20261108_163624_317288/`.

### Değerlendirme

```bash
python -m src.evaluate --predictions results/model1_lstm/val_captions_greedy.json --model-name model1_lstm --meteor --output results/model1_lstm/val_metrics_greedy.json
python -m src.evaluate --predictions results/model1_lstm/val_captions_beam3.json --model-name model1_lstm --method beam --beam-size 3 --meteor --output results/model1_lstm/val_metrics_beam3.json
```

- Metrikler: BLEU-1..4, METEOR, ROUGE-L, CIDEr (pycocoevalcap 1.2, ham ölçek). Referanslar `data/flickr8k/captions.txt` dosyasından okunur (ya da `--references` ile JSON).
- `--meteor` Java ister; Java yoksa bu seçenek olmadan çalıştırın.
- **Türkçe yerel ayar sorunu:** Java Türkçe yerel ayarla çalışınca METEOR jar'ı ondalık sayıları okuyamaz ve `evaluate.py` "METEOR failed or timed out" verir (zaman aşımı değildir, birkaç saniyede düşer). Kurulum değiştirmeden çözüm, komutu `JAVA_TOOL_OPTIONS="-Duser.language=en -Duser.country=US"` ile çalıştırmaktır (PowerShell: `$env:JAVA_TOOL_OPTIONS = "-Duser.language=en -Duser.country=US"`). Bu ayar BLEU, ROUGE-L ve CIDEr'i değiştirmez.
- `--output` var olan bir dosyanın üzerine yazmaz.

### Birleşik tablo

`python -m src.results_table` (seçenekler `--results-dir`, `--out-md`, `--out-csv`) ana tabloyu (dört model × greedy / beam-3) ve ablation tablolarını üretir. Sonucu olmayan model "henüz yok" görünür. Koşu klasörü eşlemeleri `src/results_table.py` başındaki `MAIN_RUNS` ve deney sözlüklerindedir; ablation tablolarındaki "tek fark" cümleleri config YAML farkından otomatik yazılır. Tablo split, görüntü sayısı, yöntem, predict–evaluate CIDEr eşliği, checkpoint epoch ve referans/bölme SHA-256'sını denetler.

### Demo

```bash
python -m app.app
```

Tarayıcıda `http://127.0.0.1:7860` açılır. Fotoğraf yüklenir; dört modelin altyazısı, çıkarım süresi ve parametre sayısı yan yana görünür, attention destekleyen modelde kelime başına harita çizilir. Greedy ve beam arasında geçiş ve beam genişliği ayarı vardır. Checkpoint'i olmayan model kutusunda açık bir mesaj görünür.

- Checkpoint yolları yalnız `app/config.yaml` içindedir. Ayrıntı ve Model 3 / BLIP bağlama adımları: `app/README.md`. Uçtan uca CPU doğrulama sonuçları ve ortam raporu: [report/demo_dogrulama.md](report/demo_dogrulama.md).
- Seçenekler: `--config`, `--device` (`auto` / `cpu` / `cuda`), `--host`, `--port`, `--share`.
- Sunucu varsayılan olarak yalnız bu bilgisayarda (`127.0.0.1`) açılır. `--share` herkese açık geçici bir bağlantı oluşturur; yalnız sunum sırasında kullanın.

### Testler

```bash
pip install pytest
python -m pytest tests
```

Test dosyaları: `tests/test_lstm.py` (Model 1), `tests/test_transformer.py` (Model 3), `tests/test_blip.py` ve `tests/test_blip_runtime.py` (BLIP), `tests/test_decode.py` (çözme), `tests/test_evaluate.py` (metrikler), `tests/test_glove.py` (GloVe yükleme), `tests/test_results_table.py` (birleşik tablo, geçici sahte `results/` klasörüyle), `tests/test_model2_review.py` (Model 2 hata incelemesi).

## Sınırlılıklar ve notlar

- **Tek seed:** her eğitim koşusu seed 42 ile bir kez yapıldı. Tablolar varyans ya da güven aralığı içermez; küçük farklar (ör. GloVe beam-3'te +0.008 CIDEr) gürültü sınırında olabilir. Yalnız nucleus deneyi üç seed ile tekrarlandı ve bu tekrar yalnız örneklemeyi değiştirir, eğitimi değil.
- **Belirlenimcilik:** eğitim GPU'da AMP ile yapılır ve cuDNN belirlenimci kipe alınmaz; aynı komutlar başka donanımda (hatta aynı donanımda) birebir aynı sayıları vermeyebilir.
- **Test kümesi bir kez açıldı:** Dört modelin (BLIP iki varyantıyla) test sonuçları `results/<ad>/test/` altındadır. Tüm modellerde checkpoint seçimi doğrulama kümesindeki greedy CIDEr üzerinden yapılmıştır. Fine-tune'lu BLIP Kaggle'da (`transformers 4.57.1`) eğitildi, test çıkarımı yerelde (`transformers 5.17.0`) yapıldı; 100 doğrulama görüntüsünde greedy cümleler 100/100, beam-3 cümleler 96/100 aynı çıktı.
- **Zero-shot değil:** "fine-tune edilmeden" denen BLIP, COCO üzerinde altyazılama için zaten eğitilmiştir; yalnız Flickr8k'yı görmemiştir.
- **GloVe Kaggle kopyası:** resmi Stanford sunucusuna erişilemediği için Kaggle'daki bir kopya kullanıldı. Kopya dosya boyutu, satır sayısı, ilk satır ve basit analoji / komşu testleriyle doğrulandı (`results/model1_glove.md`); resmi dosya indirilip doğrudan karşılaştırılmadı.
- **METEOR Java'ya bağlı:** Java yoksa METEOR hesaplanmaz; Türkçe yerel ayarda yukarıdaki ortam değişkeni gerekir.
- **Sabitlenmiş paket sürümleri:** Test edilen paket sürümleri Python 3.12.3 ortamında [requirements.txt](requirements.txt) içinde sabitlenmiştir ([report/demo_dogrulama.md](report/demo_dogrulama.md)); yine de farklı donanım veya CUDA sürümleri küçük sayısal farklar doğurabilir.
- **Model 2 checkpoint'i repoda yok:** Model 2 sonuçları commit'li JSON'lardan okunabilir; yeniden puanlama ve demo için checkpoint ekip içinden alınmalı ya da yeniden eğitilmelidir.
- **Genel sınırlılıklar:** modeller görüntüde olmayan nesneleri anlatabilir (hallucination), eğitim verisindeki yanlılık altyazılara yansır ve küçük veri jenerik cümlelere yol açar. Kalitatif karşılaştırma ve 40 görüntülük hata incelemesi: [report/kalitatif_karsilastirma_4model.md](report/kalitatif_karsilastirma_4model.md).
- **Örnek görseller:** `results/` altındaki attention görsellerinde kullanılan fotoğraflar Flickr8k veri setinden alınmıştır (Hodosh vd., 2013). Fotoğrafların hakları Flickr'daki sahiplerine aittir; burada yalnız araştırma ve eğitim amaçlı örnek olarak gösterilmiştir.
- **Checkpoint yükleme:** tüm checkpoint'ler `torch.load(..., weights_only=True)` ile açılır; dosya yalnız tensör ve temel veri tipleri içerebilir, kod çalıştıramaz.

## Kaynakça

1. O. Vinyals, A. Toshev, S. Bengio, D. Erhan. *Show and Tell: A Neural Image Caption Generator.* CVPR, 2015. [arXiv:1411.4555](https://arxiv.org/abs/1411.4555)
2. K. Xu, J. Ba, R. Kiros, K. Cho, A. Courville, R. Salakhutdinov, R. Zemel, Y. Bengio. *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention.* ICML, 2015. [arXiv:1502.03044](https://arxiv.org/abs/1502.03044)
3. D. Bahdanau, K. Cho, Y. Bengio. *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR, 2015. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
4. M. Cornia, M. Stefanini, L. Baraldi ve R. Cucchiara. [Meshed-Memory Transformer for Image Captioning](https://arxiv.org/abs/1912.08226), CVPR 2020. İlgili çalışma. Model 3 ortak ResNet50 önbelleğiyle standart bir Transformer decoder kullanır.
5. J. Li, D. Li, C. Xiong, S. Hoi. *BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation.* ICML, 2022. [arXiv:2201.12086](https://arxiv.org/abs/2201.12086)
6. M. Hodosh, P. Young, J. Hockenmaier. *Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics.* Journal of Artificial Intelligence Research, 47, 2013. (Flickr8k)
7. K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR, 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
8. J. Pennington, R. Socher, C. D. Manning. *GloVe: Global Vectors for Word Representation.* EMNLP, 2014.
9. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin. *Attention Is All You Need.* NeurIPS, 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
10. K. Papineni, S. Roukos, T. Ward, W.-J. Zhu. *BLEU: a Method for Automatic Evaluation of Machine Translation.* ACL, 2002.
11. S. Banerjee, A. Lavie. *METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments.* ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, 2005.
12. C.-Y. Lin. *ROUGE: A Package for Automatic Evaluation of Summaries.* Text Summarization Branches Out (ACL Workshop), 2004.
13. R. Vedantam, C. L. Zitnick, D. Parikh. *CIDEr: Consensus-based Image Description Evaluation.* CVPR, 2015. [arXiv:1411.5726](https://arxiv.org/abs/1411.5726)
14. Flickr8k, Kaggle sürümü: [kaggle.com/datasets/adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)

## Ekip içi çalışma

Dallar, dosya sahipliği ve Pull Request kuralları: [CONTRIBUTING.tr.md](CONTRIBUTING.tr.md).

## Lisans

Bu repodaki kod [MIT Lisansı](LICENSE) ile yayımlanmıştır. Lisans yalnız bu proje için yazılan kodu ve belgeleri kapsar. Flickr8k görüntüleri (`results/` altındaki şekillerde görünenler dahil), önceden eğitilmiş BLIP ve ResNet50 ağırlıkları ve GloVe vektörleri kendi koşullarına tabidir.

# Görüntü Altyazılama

Flickr8k üzerinde CNN-LSTM, dikkat mekanizmalı LSTM, Transformer decoder ve BLIP karşılaştırması.

## Kurulum

Python 3.12 ile denenmiştir. Model 1, 2 ve 3 için CUDA destekli bir GPU önerilir; önbellek ve eğitim CPU'da da çalışır ama yavaştır.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

- GPU için PyTorch'u CUDA sürümüyle kurun, örneğin CUDA 12.4: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`. Ayrıntı: [pytorch.org/get-started](https://pytorch.org/get-started/locally/). Önbellek bu projede `torch 2.6.0+cu124` ve `torchvision 0.21.0+cu124` ile üretildi.

- METEOR metriği Java ister (`java -version` çalışmalı). Java yoksa BLEU, ROUGE-L ve CIDEr yine hesaplanır.

- `requirements.txt` içindeki sürümler henüz sabitlenmedi. Son sürümler proje sonunda yazılacak.

- Kurulumu doğrulamak için testler: `pip install pytest` ve `python -m pytest tests`.

## Veri

Flickr8k'nın Kaggle sürümü kullanılır: [adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k). İçinde 8091 görüntü (`Images/`) ve görüntü başına 5 altyazı (`captions.txt`, başlıkları `image,caption`) vardır.

Kaggle sayfasından zip olarak indirip açın ya da Kaggle API kullanın. API anahtarı `kaggle.json` kullanıcı klasöründe durur (`~/.kaggle/kaggle.json`, Windows'ta `C:\Users\<ad>\.kaggle\kaggle.json`) ve repoya konmaz.

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

`data/` klasörü `.gitignore` içindedir. Bölme dosyası ve sözlük repoda hazırdır, yeniden üretilmez.

## Hızlı başlangıç

Kurulum ve veriden sonra Model 1'i baştan sona çalıştırma sırası:

```bash
# 1. ResNet50 öznitelik önbelleği (GPU'da yaklaşık 1 dakika, ~1,6 GB)
python -m src.data.cache_features

# 2. Eğitim (RTX 3050'de yaklaşık 6 dakika, erken durdurmayla)
python -m src.train --config configs/model1.yaml

# 3. Doğrulama cümleleri ve CIDEr: greedy ve beam-3
python -m src.predict --checkpoint checkpoints/model1_lstm/best.pt --method greedy
python -m src.predict --checkpoint checkpoints/model1_lstm/best.pt --method beam --beam-size 3
```

Model 2 için `configs/model2.yaml` ve `checkpoints/model2_attention_lstm/best.pt` kullanılır. Sonuçlar `results/<ad>/` altına yazılır. Her adımın ayrıntısı aşağıdaki bölümlerdedir.

## Dalları

- `main`: çalışan ortak kod

- `data`: Üye A

- `attention`: Üye B

- `transformer-demo`: Üye C

Çalışmaya başlamadan `main` dalını çekip kendi dalınıza alın. Bitince kendi dalınıza commit atın ve `main` dalına Pull Request açın.

## Dosya sahipliği

- `train.py` ve `src/data/`: Üye A

- `decode.py` ve attention modeli: Üye B

- `evaluate.py` ve `app/`: Üye C

Başkasının dosyasında değişiklik gerekiyorsa dosyanın sahibi Pull Request'i birleştirir.

## Sözleşme

- Model 1, 2 ve 3 aynı bölmeyi, aynı sözlüğü ve aynı ResNet50 önbelleğini kullanır.

- BLIP kendi görüntü işlemcisini ve tokenizer'ını kullanır.

- Her model dışarıya `generate(image) -> caption` verecek (demo aşamasında eklenecek, henüz yok). Eğitim önbellekten okur. Demo ham görüntü alır.

- Sözlük token'ları: `<pad>`, `<start>`, `<end>`, `<unk>`.

- Minimum kelime sıklığı 5. Maksimum dizi uzunluğu 34. `<start>` ve `<end>` bu sayıya dahildir.

- Üretim `<start>` ile başlar, `<end>` token'ında veya 34 token'da durur. Pad sağa eklenir.

- Seed 42'dir.

- Metrikler düz cümle üzerinden hesaplanır. `<start>`, `<end>`, `<pad>` ve `<unk>` silinir. Çıktı ve referanslar küçük harfe çevrilir, noktalama silinir.

- Test kümesi yalnızca final tablo için bir kez kullanılır.

- Veri, önbellek, kontrol noktası ve `kaggle.json` repoya konmaz.

## Veri hattı

Üye A'nın Hafta 1 çıktısı. Model 1, 2 ve 3 bu dosyaları kullanır. Bölme yeniden üretilmez.

- Bölme: `src/data/artifacts/flickr8k_split.json`. Train 6000, val 1000, test 1000. 91 görüntü `unused` içindedir ve eğitime girmez.

- Sözlük: `src/data/artifacts/vocab.json`. Yalnız train altyazılarından kurulur. Sıklığı 5'in altındaki kelime `<unk>` olur.

- Yollar `configs/base.yaml` içindedir. Yerel veri `data/flickr8k/Images` ve `data/flickr8k/captions.txt` olarak durur; repoda yoktur.

- `token_ids` uzunluğu 34'tür: `<start>`, kelimeler, `<end>`, sağda `<pad>`. Eğitim `build_dataloader(split)` ile okur. Önbellek `image_id` (dosya adı) ile eşlenir.

- Test kümesi metrik için açılmaz.

### Veri hazırlama betiği

```bash
python -m src.data.prepare
```

Bölme dosyası, sözlük ve `results/eda/` altındaki grafikler bu betikle üretildi. Normal kullanımda **çalıştırılması gerekmez**.

- Betik `flickr8k_split.json` ve `vocab.json` dosyalarının **üzerine yazar**. Seed 42 sabit olduğu için aynı veriyle aynı sonucu verir. Çalıştırdıktan sonra `git status` bu iki dosyada değişiklik gösteriyorsa veri farklıdır: değişikliği commit etmeyin, `git checkout -- src/data/artifacts` ile geri alın.

- Ne yapar: görüntü ve altyazıları eşleştirir, metin temizliğini sınar, bölmeyi ve sözlüğü kurar, EDA grafiklerini ve `results/eda/summary.json` dosyasını yazar, train ve val için örnek batch'i denetler (şekil, `<start>`, sağa pad, kümeler arası sızıntı).

### Öznitelik önbelleği

Model 1, 2 ve 3 görüntü yerine dondurulmuş ResNet50 (`IMAGENET1K_V1`) layer4 çıktısını okur. Dönüşüm ve encoder `src/data/features.py` içindedir; demo aynı modülü `extract_image` ile kullanır.

```bash
python -m src.data.cache_features
```

- Çıktılar `data/features/` altındadır: `resnet50_layer4.npy` (fp16, `[8091, 2048, 7, 7]`, yaklaşık 1,6 GB), `resnet50_layer4_index.json` (`image_id -> satır`) ve `resnet50_layer4.meta.json` (ağırlık, dtype, şekil, sürümler). Yollar `configs/base.yaml` içindeki `features_path` ve `features_index_path` alanlarıdır.

- Seçenekler: `--images-dir`, `--out-dir`, `--batch-size`, `--limit` (deneme için ilk N görüntü), `--device`, `--num-workers`, `--no-amp`. GPU'da autocast kullanılır, kayıt her zaman fp16'dır.

- Kaggle: girdi salt okunurdur, çıktıyı çalışma klasörüne yazın.

```bash
python -m src.data.cache_features --images-dir /kaggle/input/flickr8k/Images --out-dir /kaggle/working/features
```

- Eğitim: `build_dataloader("train", use_features=True)` batch'e `features` (float32, `[B, 2048, 7, 7]`) ekler. Önbellek başka klasördeyse `features_dir=Path("/kaggle/working/features")` verin.

- Doğrulama: `build_eval_dataloader("val")` görüntü başına bir örnek (`image_id`, `features`) verir. `load_references("val")` her görüntünün 5 temiz altyazısını döndürür. İkisi de test kümesini yalnız `allow_test=True` ile açar.

## Eğitim

Model 1, 2 ve 3 aynı betikle eğitilir. Her modelin ayarları `configs/` altındaki kendi dosyasındadır.

```bash
python -m src.train --config configs/model1.yaml
```

- Önbellekten okur, teacher forcing, pad yok sayılan çapraz entropi, gradyan kırpma 5, AMP (GPU'da), seed 42.

- Her epoch doğrulama kaybı hesaplanır. Greedy CIDEr her `cider_every` epoch'ta, kayıp iyileştiğinde ve son epoch'ta hesaplanır. Erken durdurma `early_stopping.monitor` (varsayılan `val_loss`, sabır 5), `best.pt` seçimi `evaluation.select_by` (varsayılan `val_cider`) ile yapılır.

- Çıktılar: `checkpoints/<ad>/best.pt` ve `last.pt` (repoya girmez), `results/<ad>/history.json`, `loss_curve.png`, `summary.json`, `val_captions_best.json`.

- Seçenekler: `--name`, `--max-epochs`, `--limit-batches` (deneme), `--features-dir` (Kaggle), `--num-workers`, `--device`, `--resume` (`last.pt`'den devam).

- Veri miktarı deneyi: config'te `data.train_fraction: 0.25` gibi. Alt küme görüntü bazlıdır ve seed ile sabittir.

- Doğrulama CIDEr'i eğitim sırasında toplu greedy ile hesaplanır (`src/decode.py` ile aynı kurallar). Final tablo `src/decode.py` ve `src/evaluate.py` ile üretilir.

- Yeni model: `src/train.py` içindeki `MODELS` sözlüğüne eklenir. Arayüz: `forward(features, caption_inputs) -> (logits, attention veya None)`.

## Cümle üretimi ve doğrulama satırları

Eğitilmiş checkpoint'ten cümleler `src/decode.py` içindeki `decode()` ile üretilir.

```bash
python -m src.predict --checkpoint checkpoints/model1_lstm/best.pt --method greedy
python -m src.predict --checkpoint checkpoints/model1_lstm/best.pt --method beam --beam-size 3
```

- Çıktılar `results/<ad>/` altındadır: `val_captions_<yöntem>.json` (`{image_id: cümle}`) ve `val_metrics.json` (yöntem başına CIDEr, farklı cümle sayısı, görüntü başına çıkarım süresi).

- Ana tablo her model için iki satırdır: `greedy` ve `beam3`. Nucleus (`--method nucleus --top-p 0.9`) çeşitlilik deneyi olarak ayrı raporlanır.

- Test kümesi yalnız final tablo için `--split test --allow-test` ile bir kez açılır.

- BLEU, METEOR ve ROUGE-L `src/evaluate.py` ile bu cümle dosyalarından hesaplanır.

## Değerlendirme ve birleşik tablo

Kayıtlı cümle dosyaları GPU ve checkpoint olmadan puanlanır (`src/evaluate.py`). Referanslar `data/flickr8k/captions.txt` dosyasından okunur.

```bash
python -m src.evaluate --predictions results/model1_lstm/val_captions_greedy.json --model-name model1_lstm --meteor --output results/model1_lstm/val_metrics_greedy.json
python -m src.evaluate --predictions results/model1_lstm/val_captions_beam3.json --model-name model1_lstm --method beam --beam-size 3 --meteor --output results/model1_lstm/val_metrics_beam3.json
```

- Metrikler: BLEU-1..4, METEOR, ROUGE-L, CIDEr (ham pycocoevalcap ölçeği). `--meteor` Java ister; Java yoksa bu seçenek olmadan çalıştırın.
- `--output` var olan bir dosyanın üzerine yazmaz. Yeniden hesaplamak için farklı bir yol verin.

Tüm koşuların sonuçları tek tabloda toplanır:

```bash
python -m src.results_table
```

- Çıktı: `results/val_table.md` ve `results/val_table.csv`. Ana tablo (dört model × greedy / beam-3) ve ablation tabloları (veri miktarı, çözümleme yöntemi, attention var/yok, GloVe). Sayılar `results/` altındaki JSON'lardan okunur; sonucu olmayan model "henüz yok" görünür.
- Deney özetleri: `results/model1_data_fraction.md` (veri miktarı), `results/model1_glove.md` (GloVe).

## Deneyler

Her deneyde `configs/model1.yaml` üzerinden tek ayar değişir; değişiklik deney config'indedir.

| Deney | Config | Değişen ayar |
|---|---|---|
| Veri miktarı | `configs/model1_frac25.yaml`, `configs/model1_frac50.yaml` | `data.train_fraction` |
| Embedding (kontrol) | `configs/model1_emb300.yaml` | `model.embedding_dim: 300` (rastgele) |
| Embedding (GloVe) | `configs/model1_glove300.yaml` | `emb300` + GloVe başlatma (`embeddings` bölümü) |

GloVe deneyi `data/glove/glove.6B.300d.txt` dosyasını ister (GloVe 6B, 400.000 satır, ~1 GB; repoda yoktur). Resmi kaynak: [nlp.stanford.edu/projects/glove](https://nlp.stanford.edu/projects/glove/). Bu projede resmi sunucuya erişilemediği için doğrulanmış bir Kaggle kopyası kullanıldı; ayrıntı `results/model1_glove.md` içindedir.

## Demo

```bash
python -m app.app
```

Tarayıcıda `http://127.0.0.1:7860` açılır. Fotoğraf yüklenir; dört modelin altyazısı, çıkarım süresi ve parametre sayısı yan yana görünür, attention destekleyen modelde kelime başına harita çizilir. Checkpoint'i olmayan model kutusunda açık bir mesaj görünür. Checkpoint yolları `app/config.yaml` içindedir. Ayrıntı ve model bağlama adımları: `app/README.md`.

- Sunucu varsayılan olarak yalnız bu bilgisayarda (`127.0.0.1`) açılır. `--share` herkese açık geçici bir bağlantı oluşturur; yalnız sunum sırasında kullanın.
- Model 2 checkpoint'i repoda yoktur; `checkpoints/model2_attention_lstm_full_20261104_01/best.pt` yoluna konunca demoda çalışır.

## Testler

```bash
pip install pytest
python -m pytest tests
```

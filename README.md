# # Görüntü Altyazılama

Flickr8k üzerinde CNN-LSTM, dikkat mekanizmalı LSTM, Transformer decoder ve BLIP karşılaştırması.

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

- Her model `generate(image) -> caption` verir. Eğitim önbellekten okur. Demo ham görüntü alır.

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

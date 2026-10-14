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


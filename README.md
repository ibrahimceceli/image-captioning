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


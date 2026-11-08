# Model 2 nucleus deney tasarımı

README nucleus top_p=0.9 çeşitlilik deneyinin ayrı raporlanmasını ister. Seed 42, 43 ve 44 ile üç tekrar bizim deney tercihimizdir; README zorunluluğu değildir.

## Sabit ayarlar

- Checkpoint: checkpoints/model2_attention_lstm_full_20261104_01/best.pt (epoch 10).
- Aynı 1000 val görüntüsü; limit yok, test kullanılmaz.
- Nucleus top_p=0.9; temperature dönüşümü yok (softmax doğrudan logits üzerinden).
- CUDA, model.eval(), no_grad; AMP kapalı, batch_size=1, num_workers=0.
- Aynı ortak vocab, split ve ResNet50 layer4 önbelleği.
- En fazla start dahil 34 token; end durdurur; start/pad seçimden maskelenir; özel tokenlar çıktı metninden çıkarılır.
- Mevcut decoder her görüntüde yerel CPU rastgele üretecini koşunun seed değerine sıfırlar. Üç koşu yalnız bu seed ile farklılaşır; algoritma değiştirilmez.

## Dosya düzeni ve raporlama

Her seed için results/model2_attention_lstm/nucleus_p09_seed<seed>/ dizini kullanılır. predict --output-dir mevcut tahmin veya val_metrics.json varsa başlamayı reddeder; legacy greedy/beam çıktıları değişmez.

Her dizinde val_captions_nucleus0.9.json ve val_metrics.json üretilir. İkincisi CIDEr, farklı açıklama sayısı/oranı, seed, top_p ve üretim ayarlarını kaydeder. Üretimden sonra ortak src.evaluate ile kalite metrikleri ayrı quality_metrics.json dosyasına hesaplanacaktır. METEOR bu deneyde dahil değildir; BLEU-1..4, CIDEr, ROUGE-L kullanılacaktır.

Üç koşu tamamlanınca aynı val ID kapsamı doğrulanarak:

1. Her koşuda benzersiz açıklama sayısı / 1000 ve oranı raporlanır (boş açıklamalar dışlanmaz).
2. Her görüntü için üç metnin oluşturduğu kümenin büyüklüğü hesaplanır; bu büyüklüklerin 1000 görüntü üzerindeki ortalaması raporlanır (1–3 aralığı).
3. Her kalite metriği için üç koşunun aritmetik ortalaması ve örnek standart sapması (n-1, ddof=1) raporlanır.

Çeşitlilik eşitliği üretilen açıklama metninin tam eşitliğidir. Kalite metrikleri ortak evaluate normalizasyonunu kullanır. Tekrarlar en iyi seed seçmek için değil değişkenliği göstermek içindir. Deney henüz çalıştırılmamıştır.

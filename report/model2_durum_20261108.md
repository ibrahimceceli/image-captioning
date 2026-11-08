# Model 2 durum raporu — 8 Kasım 2026

Model 2'nin eğitimi, greedy/beam3 doğrulaması, nucleus top_p=0.9 çeşitlilik deneyi ve üç attention görselleştirmesi tamamlandı. Test değerlendirmesi yapılmadı. METEOR hesaplanmadı.

## B üyesinin katkıları

- `src/models/attention_lstm.py`: 49 uzamsal bölge üzerinde Bahdanau attention kullanan LSTM decoder.
- `src/decode.py`: greedy, beam search ve nucleus cümle üretimi; kelimelerle hizalı attention döndürme desteği.
- `src/visualize_attention.py`: modelin gördüğü RGB kırpımı üzerinde kelime başına attention haritaları.
- Model 2 eğitim/doğrulama sonuçları, nucleus tekrarları ve örnek hata analizi. Ortak eğitim, veri hattı ve metrik altyapısı ekip bileşenleridir; yalnız B katkısı olarak sunulmaz.

## Eğitim ve veri

- Flickr8k: 6000 train, 1000 val; ortak sözlük 2572 token.
- ResNet50 IMAGENET1K_V1 layer4 önbelleği: data/features/resnet50_layer4.npy; [8091,2048,7,7], float16. Veri yükleyicisi float32 döndürür.
- Attention LSTM: 11.159.565 parametre; CUDA, AMP, train batch 32, num_workers 0, Adam lr 0.001, seed 42.
- Koşu: model2_attention_lstm_full_20261104_01. Eğitim 19,4 dakika ve 11 epoch sürdü. Val loss beş epoch iyileşmediği için erken durdu.
- Checkpoint: checkpoints/model2_attention_lstm_full_20261104_01/best.pt. Greedy val CIDEr ile seçilen epoch 10; eğitim içi CIDEr 0.4416. Seçilen epoch loss'u 2.8238; en düşük val loss epoch 6'da 2.7861.

Erken durdurma ile checkpoint seçimi farklıdır: `val_loss` beş ardışık epoch iyileşmeyince eğitim durur; `best.pt` ise hesaplanan en yüksek greedy `val_cider` değerine göre seçilir. Dolayısıyla en düşük loss epoch 6 iken seçilen checkpoint epoch 10’dur. Config üst sınırı 30 epoch, gradyan kırpma 5.0, teacher forcing 1.0 ve pad yok sayılan çapraz entropidir. Eğitim komutunda Windows için num_workers=0 kullanılmıştır.

Sayısal eğitim kaynakları: `configs/model2.yaml`, `results/model2_attention_lstm_full_20261104_01/summary.json` ve `history.json`.

## Doğrulama sonuçları

1000 val görüntüsü, görüntü başına beş referans. Metrikler ham pycocoevalcap ölçeğindedir.

| Metrik | Greedy | Beam 3 |
|---|---:|---:|
| BLEU-1 | 0.5638 | 0.6170 |
| BLEU-2 | 0.3871 | 0.4355 |
| BLEU-3 | 0.2566 | 0.2988 |
| BLEU-4 | 0.1698 | 0.2030 |
| CIDEr | 0.4409 | 0.4697 |
| ROUGE-L | 0.4312 | 0.4396 |
| Süre, ms/görüntü | 63.20 | 182.65 |
| Farklı açıklama | 877 | 813 |

Beam3 kalite metriklerinde daha yüksek; yaklaşık 2,9 kat yavaş. METEOR hesaplanmadı. Eğitim içi AMP/batch greedy CIDEr ile ayrı tahmin koşusunun CIDEr'i küçük fark gösterir; nedenini izole eden deney yapılmadı.

Sonuçlar results/model2_attention_lstm/ altında; eğitim günlükleri results/model2_attention_lstm_full_20261104_01/ altında. predict.py checkpoint config adını kullandığından klasör adları farklıdır.

## Nucleus çeşitlilik deneyi — tamamlandı

Kaynak: `results/model2_attention_lstm/nucleus_p09_summary.json`. Aynı epoch 10 checkpoint’i ve aynı 1000 val görüntüsü kullanıldı. Top_p=0.9; seed 42, 43 ve 44 ile üç tekrar bizim deney tercihimizdir, README zorunluluğu değildir. Diğer üretim ayarları sabit: CUDA, eval/no_grad, AMP kapalı, batch 1, num_workers=0; en fazla start dahil 34 token. Mevcut decoder her görüntüde yerel CPU üretecini koşu seed’ine sıfırlar.

| Metrik | Üç koşunun ortalaması | Örnek standart sapma (ddof=1) |
|---|---:|---:|
| BLEU-1 | 0.4768 | 0.0428 |
| BLEU-2 | 0.2776 | 0.0492 |
| BLEU-3 | 0.1577 | 0.0430 |
| BLEU-4 | 0.0889 | 0.0328 |
| CIDEr | 0.2519 | 0.0421 |
| ROUGE-L | 0.3432 | 0.0477 |

Standart sapma üç seed arasındaki değişkenliği gösterir; görüntü başına belirsizlik veya güven aralığı değildir.

### Görüntüler arasında farklı açıklamalar

Her koşudaki 1000 açıklamanın farklı tam metinlerinin sayısı ve bu sayının 1000’e oranı:

| Seed | Farklı açıklama / 1000 | Oran |
|---|---:|---:|
| 42 | 989 | %98.9 |
| 43 | 993 | %99.3 |
| 44 | 994 | %99.4 |

### Aynı görüntü için seed’ler arasında çeşitlilik

Her görüntüde üç seed’in açıklamaları bir küme olarak karşılaştırıldı. Görüntü başına farklı açıklama sayısının ortalaması **3**; 1000 görüntünün tamamında üç farklı tam metin var. Bu ölçü yukarıdaki görüntüler arası orandan ayrıdır. Boş açıklamalar dışlanmaz. Tam metin farklılığı anlamsal çeşitlilik kanıtı değildir; aynı anlam farklı sözcüklerle ifade edilmiş olabilir.

Nucleus ortalama CIDEr 0.2519 ile greedy (0.4409) ve beam3’ten (0.4697) düşük; bu koşulda yüksek metinsel çeşitlilik daha yüksek referans uyumu sağlamamıştır.

Kalite kaynakları: her `nucleus_p09_seed42/`, `nucleus_p09_seed43/`, `nucleus_p09_seed44/` klasöründeki `quality_metrics.json`; tahminler ve üretim ayarları aynı klasörlerde korunur. Ortak evaluate temizliği: özel tokenları kaldırma, küçük harfe çevirme, noktalama kaldırma ve boşluk birleştirme. Greedy/beam3 ile referans ve split SHA256 değerleri aynıdır.

## Attention örnekleri ve kontroller

results/model2_attention_lstm/attention_20261108_163624_317288/ içinde üç PNG ve captions.json üretildi. İlk üç val ID'si kaliteye göre filtrelenmeden kullanıldı. Aynı kayıtlar için mevcut beam3 cümleleriyle birebir eşleşme doğrulandı. Attention [kelime_sayısı,49] biçiminde; değerler sonlu ve negatif değil, satır toplamları 1. Görüntüler ortak Resize/CenterCrop dönüşümünün normalize edilmemiş 224x224 RGB çıktısıdır.

- [1015584366_dfcec3c85a.jpg](../results/model2_attention_lstm/attention_20261108_163624_317288/1015584366_dfcec3c85a_beam3.png) — `results/model2_attention_lstm/attention_20261108_163624_317288/1015584366_dfcec3c85a_beam3.png`
- [102455176_5f8ead62d5.jpg](../results/model2_attention_lstm/attention_20261108_163624_317288/102455176_5f8ead62d5_beam3.png) — `results/model2_attention_lstm/attention_20261108_163624_317288/102455176_5f8ead62d5_beam3.png`
- [1028205764_7e8df9a2ea.jpg](../results/model2_attention_lstm/attention_20261108_163624_317288/1028205764_7e8df9a2ea_beam3.png) — `results/model2_attention_lstm/attention_20261108_163624_317288/1028205764_7e8df9a2ea_beam3.png`

- Köpek: nesne tanınıyor, fakat referanstaki kütük üzerinden atlama yerine ormanda koşma deniyor.
- Buz tırmanıcısı: referans buz tırmanışı, tahmin snowboarding; belirgin eylem hatası.
- Kayak: kayak etkinliği tanınıyor; çocuk atlanıyor ve referanslarda olmayan pool sözcüğü ekleniyor.

Attention haritaları bölge ağırlıklarını gösterir; doğru nesne tespiti veya nedensel açıklama garantisi değildir. Üç örnek genel kaliteyi temsil eden istatistiksel örneklem değildir.

Attention üretiminde mevcut model/decoder/görselleştirme fonksiyonları kullanıldı. Nucleus hazırlığında predict.py için ayrı çıktı dizini ve güvenli yol gösterimi desteği eklenmişti. Bu rapor güncellemesinde yalnız bu Markdown dosyası değiştirildi; deney, eğitim, test değerlendirmesi veya Git işlemi yapılmadı.

## Teslim durumu ve açık işler

| İş | Durum |
|---|---|
| Model 2 eğitimi, greedy/beam3 metrikleri | Tamamlandı |
| Nucleus top_p=0.9, üç seed, kalite/çeşitlilik özeti | Tamamlandı |
| Üç attention PNG ve örnek gözlemleri | Tamamlandı |
| METEOR | Hesaplanmadı; zorunluluğu mevcut belgelerde net değil |
| Test final değerlendirmesi | Yapılmadı; model/ayar seçimi sonrasında ayrıca planlanmalı |
| Demo arayüzü ve Model 2 entegrasyonu | Açık; tamamlandığı doğrulanmadı |
| B teslim PR’ı ve inceleme/birleştirme | Açık; tamamlandığı doğrulanmadı |

Diğer ekip üyelerinin güncel ilerlemesi bu yerel belgelerden varsayılmaz. Veri, ResNet50 önbelleği, checkpoint ve gizli anahtarlar Git teslimine dahil edilmemelidir.

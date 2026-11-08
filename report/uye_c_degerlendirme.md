# Üye C: ortak değerlendirme teslimi

Güncel main (350d927) üzerine metrik kodu entegre edilmiştir. Ayça'nın decode
uyumluluk düzeltmeleri bu sürümde mevcuttur; src/decode.py değiştirilmemiştir.

## Kullanım

Kayıtlı tahminleri değerlendirmek için GPU veya checkpoint gerekmez.
Flickr8k captions.txt dosyası data/flickr8k/captions.txt konumunda bulunmalıdır.
Dosya https://www.kaggle.com/datasets/adityajn105/flickr8k kaynağındandır.
Veri dosyası Git'e eklenmez; reference_source.json kaynağı ve dosya özetini tutar.

```bash
python -m src.evaluate --predictions results/model1_lstm/val_captions_greedy.json --model-name model1_lstm --output results/model1_lstm/val_metrics_greedy.json
python -m src.evaluate --predictions results/model1_lstm/val_captions_beam3.json --model-name model1_lstm --method beam --beam-size 3 --output results/model1_lstm/val_metrics_beam3.json
```

Çıktılar repoda zaten varsa yeniden hesaplamak için farklı bir --output yolu kullanın;
mevcut sonuçların üzerine yazılmaz. Alternatif veri yolu --captions ile, beş referanslı
JSON dosyası --references ile verilebilir. --method üretim yapmaz; girdi tahminlerinin
hangi yöntemle üretildiğini kaydeder.

## Doğrulama sonuçları

Her yöntem için 1000 görüntü, görüntü başına beş referans. Ham pycocoevalcap ölçeği.

| Yöntem | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr | ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| greedy | 0.4691 | 0.2907 | 0.1725 | 0.1063 | 0.2306 | 0.3718 |
| beam3 | 0.5297 | 0.3368 | 0.2131 | 0.1389 | 0.2778 | 0.3860 |

CIDEr skorları mevcut val_metrics.json ile dört ondalık basamakta eşleşir.
Yeni greedy dosyası kullanıldığı için skor, eski val_captions_best.json üzerindeki
0.2302 değerinden biraz farklıdır. Test kümesi değerlendirilmemiştir.
METEOR isteğe bağlı --meteor ile çalışır; yerel Java başlangıç sorunu nedeniyle bu
sonuçlarda yoktur. Hatalı METEOR çalışması sessizce sıfıra dönüştürülmez.

## Entegrasyon ve kontroller

- evaluate.py: ortak temizlik, beş referans, tam bölme kapsamı, yinelenen anahtar kontrolü,
  BLEU/CIDEr/ROUGE-L ve isteğe bağlı METEOR; kaynak dosyaların SHA256 özetleri.
- train.py: CIDEr ortak fonksiyona yönlendirilir; predict.py --limit desteği korunur.
  Tek görüntülü denemede CIDEr=0 anlamlı bir kalite ölçüsü değildir. Tam değerlendirme
  komutu bir görüntülü korpusu reddeder.
- data/__init__.py: mevcut dışa aktarımlar korunarak PyTorch bağımlılıkları gerektiğinde yüklenir.
- 10 metrik/entegrasyon testi geçti. Eğitim sarmalayıcısı kısmi tahminlerle de sınandı.
- PyTorch bulunmadığından model/decoder testleri ve eğitim bu ortamda çalıştırılmadı.
- --allow-test final değerlendirmeye açık geçiştir; geçmiş kullanımları izleyen bir kilit değildir.
- train.py ve src/data değişikliklerini Üye A inceleyip birleştirmelidir.

```bash
python -m unittest discover -s tests -p test_evaluate.py -v
```

Sonraki işler: Model 2 sonuçlarının ortak metriklerle değerlendirilmesi, BLIP,
Transformer ve modellerin Gradio demosuna bağlanması.

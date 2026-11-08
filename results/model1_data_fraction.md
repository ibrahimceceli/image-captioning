# Model 1 — veri miktarı deneyi (%25 / %50 / %100)

Model 1 (CNN-LSTM) `configs/model1.yaml` ayarlarıyla; tek değişen `data.train_fraction` (`configs/model1_frac25.yaml`, `configs/model1_frac50.yaml`). Alt küme görüntü bazlı, seed 42 ile sabit. Tüm metrikler doğrulama kümesinden (1000 görüntü × 5 referans); test kümesi açılmadı. Checkpoint seçimi greedy val CIDEr, erken durdurma val loss (sabır 5).

## Eğitim

| Veri | Eğitim görüntü | Eğitim örnek | Best epoch | Çalışılan epoch | Süre (dk) | Val loss (best epoch) | En düşük val loss (epoch) | Son epoch train / val loss | Son epoch val − train | Val loss artışı (en düşükten sona) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| %25 | 1500 | 7500 | 10 | 12 | 2.3 | 3.2776 | 3.2245 (7) | 1.9657 / 3.3577 | 1.3920 | +0.1332 |
| %50 | 3000 | 15000 | 12 | 13 | 4.8 | 3.1258 | 3.0354 (8) | 1.9855 / 3.1526 | 1.1671 | +0.1172 |
| %100 | 6000 | 30000 | 10 | 12 | 5.7 | 2.8761 | 2.8429 (7) | 2.1780 / 2.9139 | 0.7359 | +0.0710 |

## Doğrulama metrikleri (best.pt)

| Veri | Çözme | BLEU-1 | BLEU-4 | CIDEr | ROUGE-L | Farklı cümle (/1000) |
|---|---|---:|---:|---:|---:|---:|
| %25 | greedy | 0.4414 | 0.0801 | 0.1642 | 0.3536 | 93 |
| %50 | greedy | 0.4994 | 0.1019 | 0.1943 | 0.3628 | 127 |
| %100 | greedy | 0.4691 | 0.1063 | 0.2306 | 0.3718 | 176 |
| %25 | beam-3 | 0.4890 | 0.1175 | 0.2178 | 0.3610 | 88 |
| %50 | beam-3 | 0.5020 | 0.1042 | 0.2086 | 0.3594 | 124 |
| %100 | beam-3 | 0.5297 | 0.1389 | 0.2778 | 0.3860 | 164 |

## Epoch başına val loss / val CIDEr (greedy)

| Epoch | %25 val loss | %25 CIDEr | %50 val loss | %50 CIDEr | %100 val loss | %100 CIDEr |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.7824 | 0.0522 | 3.4861 | 0.0793 | 3.2284 | 0.1180 |
| 2 | 3.4963 | 0.1442 | 3.2460 | 0.1371 | 3.0249 | 0.1644 |
| 3 | 3.3642 | 0.1443 | 3.1370 | 0.0619 | 2.9315 | 0.1694 |
| 4 | 3.2964 | 0.1128 | 3.0729 | 0.1334 | 2.8896 | 0.1932 |
| 5 | 3.2529 | 0.1427 | 3.0485 | 0.1875 | 2.8593 | 0.2018 |
| 6 | 3.2312 | 0.1560 | 3.0359 | 0.1433 | 2.8523 | 0.2029 |
| 7 | 3.2245 | 0.1408 | 3.0392 | - | 2.8429 | 0.2027 |
| 8 | 3.2327 | 0.1400 | 3.0354 | 0.1309 | 2.8518 | 0.2232 |
| 9 | 3.2434 | - | 3.0663 | - | 2.8642 | - |
| 10 | 3.2776 | 0.1644 | 3.0732 | 0.1638 | 2.8761 | 0.2302 |
| 11 | 3.3163 | - | 3.1037 | - | 2.8950 | - |
| 12 | 3.3577 | 0.1516 | 3.1258 | 0.1946 | 2.9139 | 0.2218 |
| 13 |  |  | 3.1526 | - |  |  |

## Yorum

Greedy çözmede veri arttıkça CIDEr, BLEU-4 ve ROUGE-L düzenli olarak yükseliyor ve farklı cümle sayısı da artıyor; tek istisna greedy BLEU-1, %50'de %100'den yüksek. Beam-3'te %100 bütün metriklerde en iyi, ancak %50 koşusu BLEU-4, CIDEr ve ROUGE-L'de %25'in altında kalıyor; %50'de beam-3'ün greedy'ye göre kazancı diğer iki koşudan belirgin şekilde küçük. Val loss her epoch'ta veri arttıkça daha düşük; üç koşuda da en düşük değere epoch 7–8'de ulaşılıyor ve sonrasında yükseliyor, bu artış ve son epoch'taki val−train farkı veri azaldıkça büyüyor (küçük alt kümelerde aşırı öğrenme daha güçlü). Erken durdurma üç koşuda da 12–13. epoch'ta devreye giriyor; checkpoint greedy CIDEr ile seçildiği için best epoch (10–12) val loss minimumundan sonra geliyor. Epoch'lar arası greedy CIDEr dalgalı (ör. %50'de epoch 3), bu nedenle %25 ile %50 arasındaki beam-3 farkı gibi küçük farklar seçim gürültüsü içinde kalabilir.

Not: Her oran tek seed (42) ile bir kez eğitildi; sonuçlar tek koşudur, varyans ya da güven aralığı tahmini içermez.

## Nasıl üretildi

- Eğitim: `python -m src.train --config configs/model1_frac{25,50}.yaml` (log: `results/model1_lstm_frac{25,50}_train.log`).
- Tahmin: `python -m src.predict --checkpoint checkpoints/<ad>/best.pt --method greedy` ve `--method beam --beam-size 3`.
- Metrik: `python -m src.evaluate --predictions results/<ad>/val_captions_{greedy,beam3}.json --model-name <ad> [--method beam --beam-size 3] --output results/<ad>/val_metrics_{greedy,beam3}.json`.
- Tablolar kısa bir Python betiğiyle şu dosyalardan okundu: `summary.json` (epoch, süre, val loss), `history.json` (en düşük val loss, son epoch, epoch tablosu), `val_metrics_{greedy,beam3}.json` (BLEU/CIDEr/ROUGE-L, evaluate.py), `val_metrics.json` (farklı cümle sayısı ve checkpoint epoch'u, predict.py). Görüntü sayısı `src.train.train_loader` ile alt küme yeniden kurularak sayıldı; `summary.json`'daki `train_examples` ile eşleştiği betikte kontrol edildi.
- Alt küme kontrolü: her koşuda örnek = görüntü × 5, tüm görüntüler train bölümünden, val/test kesişimi 0; %25 alt kümesi %50 alt kümesinin içinde: evet.
- `Val loss (best epoch)` = checkpoint'in (CIDEr ile seçilen) epoch'undaki val loss; erken durdurma ise en düşük val loss'a göre işler.

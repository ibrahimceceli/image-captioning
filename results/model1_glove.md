# Model 1 — GloVe embedding deneyi (glove.6B.300d)

Model 1 (CNN-LSTM), `configs/model1.yaml` ayarları. Embedding boyutu 256 → 300 değiştiği için iki koşu yapıldı: `configs/model1_emb300.yaml` (300 boyut, rastgele başlatma — kontrol) ve `configs/model1_glove300.yaml` (300 boyut, GloVe ile başlatma, `freeze: false`, yani embedding ince ayarlanır). **Asıl karşılaştırma emb300 ↔ glove300**; aralarındaki tek fark embedding başlatması (GloVe ayrı bir RNG kullanır; modelin geri kalanının başlatılması ve veri sırası aynı). 256 boyutlu `model1_lstm` yalnız referans. Metrikler doğrulama kümesinden (1000 görüntü × 5 referans); test kümesi açılmadı. Checkpoint seçimi greedy val CIDEr, erken durdurma val loss (sabır 5).

## GloVe kapsaması

- Sözlük: 2572 token (2568 kelime + 4 özel token).
- GloVe'da bulunan: **2546 / 2568 (%99.14)**, eksik 22.
- Eksik kelimeler ve `<start>`/`<end>`/`<unk>`: N(0, 0.3706), seed 42 (σ = bulunan GloVe vektörlerinin std'si); `<pad>` sıfır.
- Eksik kelimeler: `tshirt`, `snowcovered`, `rollerblader`, `upsidedown`, `bicycler`, `floaties`, `merrygoround`, `africanamerican`, `dalmation`, `darkhaired`, `outstreached`, `tugofwar`, `waterskier`, `offcamera`, `wakeboarder`, `brownhaired`, `corndogs`, `darkskinned`, `fourwheeler`, `hulahoops`, `surfboarder`, `treelined`
- Eksik 22 kelimenin yarıdan fazlası (ör. `tshirt`, `snowcovered`, `upsidedown`, `merrygoround`, `darkhaired`, `tugofwar`, `treelined`) altyazı temizliğinde tirenin silinmesiyle oluşan birleşik kelimeler; geri kalanı nadir türetmeler (`rollerblader`, `bicycler`, `wakeboarder`, `surfboarder`, `waterskier`, `floaties`) ve yazım hataları (`dalmation`, `outstreached`).
- Sayıların kaynağı: `results/model1_lstm_glove300/embedding_stats.json`.
- Vektörler: GloVe 6B, 300 boyut (Pennington vd., 2014). Resmi sunucu (nlp.stanford.edu) bu ağdan erişilemediği için Kaggle kopyası (`thanakomsn/glove6b300dtxt`) kullanıldı ve doğrulandı: 1,037,962,819 bayt, 400000 satır, bozuk satır 0, ilk satır resmi dosyayla aynı (`the 0.04656 0.21318 -0.0074364 …`), king − man + woman → queen, `dog` komşuları dogs/cat/pet/puppy.

## Eğitim

| Koşu | Parametre | Best epoch | Çalışılan epoch | Süre (dk) | Val loss (best epoch) | En düşük val loss (epoch) | Son epoch train / val loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| emb300 (rastgele, 300) | 5,856,284 | 12 | 12 | 7.0 | 2.9155 | 2.8480 (7) | 2.1711 / 2.9155 |
| glove300 (GloVe, 300) | 5,856,284 | 12 | 13 | 6.8 | 2.8314 | 2.7895 (8) | 2.1815 / 2.8701 |
| referans: model1_lstm (rastgele, 256) | 5,653,004 | 10 | 12 | 5.7 | 2.8761 | 2.8429 (7) | 2.1780 / 2.9139 |

## Doğrulama metrikleri (best.pt)

| Koşu | Çözme | BLEU-1 | BLEU-4 | CIDEr | ROUGE-L | Farklı cümle (/1000) |
|---|---|---:|---:|---:|---:|---:|
| emb300 (rastgele, 300) | greedy | 0.5150 | 0.1260 | 0.2365 | 0.3874 | 154 |
| glove300 (GloVe, 300) | greedy | 0.5173 | 0.1312 | 0.2610 | 0.3865 | 217 |
| referans: model1_lstm (rastgele, 256) | greedy | 0.4691 | 0.1063 | 0.2306 | 0.3718 | 176 |
| emb300 (rastgele, 300) | beam-3 | 0.5251 | 0.1384 | 0.2843 | 0.3872 | 168 |
| glove300 (GloVe, 300) | beam-3 | 0.5431 | 0.1499 | 0.2919 | 0.3912 | 212 |
| referans: model1_lstm (rastgele, 256) | beam-3 | 0.5297 | 0.1389 | 0.2778 | 0.3860 | 164 |

## Yorum

GloVe ile başlatılan koşu, aynı boyuttaki rastgele başlatılmış kontrole (emb300) göre her iki çözmede de BLEU-1, BLEU-4 ve CIDEr'de daha yüksek: CIDEr greedy'de 0.2365 → 0.2610, beam-3'te 0.2843 → 0.2919; ROUGE-L'de fark küçük ve yönü tutarsız (greedy'de biraz düşük, beam-3'te biraz yüksek). Doğrulama kaybı glove300'de ortak 12 epoch'un hepsinde emb300'den düşük; fark epoch 1'de 0.004 iken epoch 12'de 0.084'e çıkıyor ve en düşük val loss 2.8480'den 2.7895'e iniyor. glove300'ün ürettiği farklı cümle sayısı belirgin şekilde daha fazla (greedy 154 → 217, beam-3 168 → 212); yani 1000 görüntüye daha az kalıp cümleyle yanıt veriyor. 256 boyutlu model1_lstm yalnız referans: emb300 ile arasında hem boyut hem koşu farkı olduğundan GloVe etkisi için kullanılmamalı. Epoch'lar arası greedy CIDEr dalgalı olduğu için (ör. emb300'de 0.2375 → 0.2221 → 0.2387) beam-3 CIDEr'deki 0.008'lik fark tek koşuda gürültü sınırında; val loss ve çeşitlilikteki fark daha tutarlı bir sinyal.

Not: Her koşu tek seed (42) ile bir kez eğitildi; sonuçlar varyans ya da güven aralığı tahmini içermez.

## Nasıl üretildi

- Eğitim: `python -m src.train --config configs/model1_{emb300,glove300}.yaml` (log: `results/model1_lstm_{emb300,glove300}_train.log`).
- Tahmin: `python -m src.predict --checkpoint checkpoints/<ad>/best.pt --method greedy` ve `--method beam --beam-size 3`.
- Metrik: `python -m src.evaluate --predictions results/<ad>/val_captions_{greedy,beam3}.json --model-name <ad> [--method beam --beam-size 3] --output results/<ad>/val_metrics_{greedy,beam3}.json`.
- Tablolar bir Python betiğiyle `summary.json`, `history.json`, `val_metrics_{greedy,beam3}.json` (evaluate.py), `val_metrics.json` (farklı cümle sayısı, predict.py) ve `embedding_stats.json` dosyalarından okundu.

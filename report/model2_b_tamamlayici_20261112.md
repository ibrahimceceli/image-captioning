# B — Model 2 tamamlayıcı val deneyleri ve görsel hata incelemesi

Tarih: 12 Kasım 2026. Dal: `b/model2-val-beam5-review-20261112`; temel `origin/main` `a625d644d4e970e5b02bfbc2606666afcf90023a`. Kaynak model/decoder/eğitim kodu değiştirilmedi. Yeniden eğitim veya test kümesi üzerinde tahmin/değerlendirme yapılmadı. Commit, push ve PR yapılmadı.

## Beam-5 deneyi

Aynı yerel epoch-10 checkpoint, aynı 1.000 val görüntüsü, görüntü başına aynı beş referans ve aynı `src.predict`/`src.evaluate` kullanıldı. Val-ID kapsamı, referans ve split SHA-256 eşitliği `audit.json`, `data_provenance.json` ve metrik JSON dosyalarıyla denetlendi. Checkpoint SHA-256: `902073d3b34e4c75b6bee33c16c3ab61214f44c54109df7e8b9b77c913158f42`. Eski koşu checkpoint hash'i kaydetmemiş; tarihsel dosya kimliği yol/epoch/mimari/sözlük ve aşağıdaki yeniden üretim kontrolüne dayanır.

Kaggle arşivi resmi uç noktadan hesap girişi olmadan indirildi; arşivden yalnız 1.000 val JPEG ve özgün captions.txt çıkarıldı. Train/test görüntüleri çıkarılmadı. Yeni özellik önbelleği yalnız val için `[1000,2048,7,7]` float16: ResNet50 IMAGENET1K_V1, Resize(256), CenterCrop(224), ortak ImageNet normalizasyonu; çıkarım CUDA AMP, batch 64. Önbellek hazırlığı 24,6 saniye; decoder zamanlarına dahil değil.

Yeni beam-3 ve beam-5 aynı önbelleği kullanır: CUDA RTX 3050 Ti Laptop, torch 2.6.0+cu118 / torchvision 0.21.0+cu118, batch 1, decoder AMP kapalı, en fazla start dahil 34 token, OMP/MKL thread sayısı 4. Beam sıralaması toplam log olasılığıdır; uzunluk normalizasyonu yok. Tek değiştirilen çözümleme parametresi beam genişliğidir.

| Metrik | Arşiv beam-3 | Yerel beam-3 kontrol | Yerel beam-5 | Δ beam-5 − yerel beam-3 |
|---|---:|---:|---:|---:|
| BLEU-1 | 0.6170 | 0.6169 | 0.6019 | -0.0150 |
| BLEU-2 | 0.4355 | 0.4354 | 0.4289 | -0.0065 |
| BLEU-3 | 0.2988 | 0.2987 | 0.2980 | -0.0007 |
| BLEU-4 | 0.2030 | 0.2029 | 0.2061 | +0.0033 |
| METEOR | 0.1861 | — | — | — |
| ROUGE-L | 0.4396 | 0.4395 | 0.4337 | -0.0057 |
| CIDEr | 0.4697 | 0.4701 | 0.4590 | -0.0110 |
| Farklı tam cümle / 1000 | 813 | 813 | 773 | -40.00 |
| ms/görüntü (yalnız decoder) | 182.65 | 111.99 | 193.55 | +81.56 |

Metrikler ham pycocoevalcap 1.2 ölçeği; yüzde değil. Özel tokenları kaldırma, küçük harfe çevirme, noktalama kaldırma ve boşluk birleştirme aynı. CIDEr belge frekansları tüm 1.000 val görüntüsü üzerinde hesaplandı; boş tahminler dışlanmadı.

Yerel beam-3 kontrolünde arşivdekinden farklı cümle sayısı: **5/1000**. Varsa ID listesi `comparison.json` içinde. Farkın kesin nedeni izole edilmedi; yeniden oluşturulan önbellek ve yazılım ortamı olası etkenlerdir. Beam genişliği karşılaştırmasında aynı yerel önbelleği kullanan yeni beam-3 tabanı esas alındı.

Beam-5, yerel beam-3'e göre BLEU-4'te +0.0033, CIDEr'de -0.0110, ROUGE-L'de -0.0057 değişim verdi. Farklı tam cümle sayısı 813 → 773. Tek zamanlama koşusunda beam-5/beam-3 süre oranı 1.73; daha geniş arama bütün kalite ölçülerini birlikte iyileştirmedi. Bu sonuç tek başına beam-5 ile ana beam-3 seçiminin değiştirilmesini desteklemiyor.

Yaklaşık toplam decoder süresi beam-3 111.99 sn, beam-5 193.55 sn. Bunlar CUDA senkronizasyonlu tek koşu süreleridir; model yükleme, görüntü/önbellek hazırlığı ve metrik hesaplama hariçtir. Arşivdeki 182,65 ms/görüntü farklı ortamdan geldiği için yeni süreye doğrudan hızlanma iddiası yapılmaz.

**METEOR:** Java yürütülebiliri ortamda yok. Yeni beam-5 ve yeni beam-3 kontrol sonuçlarında METEOR eksik; arşiv beam-3 METEOR=0,1861 korunmuştur. METEOR ölçülmeden bu ölçütte beam-5 iyileşti/kötüleşti denemez. Eski 8 Kasım durum raporunun METEOR hesaplanmadı notu güncel `results/val_table.md` ve arşiv metrik JSON dosyaları açısından eskimiştir.

## Rapora eklenebilecek kısa Model 1 / Model 2 yorumu

> Aynı çözümleme yöntemiyle Model 2, Model 1’den daha yüksek referans uyumu ve daha fazla farklı cümle üretmiştir. Greedy’de BLEU-4 0,1063’ten 0,1698’e, CIDEr 0,2306’dan 0,4409’a; beam-3’te BLEU-4 0,1389’dan 0,2030’a, CIDEr 0,2778’den 0,4697’ye yükselmiştir. METEOR da greedy’de 0,1555 → 0,1867, beam-3’te 0,1590 → 0,1861 olmuştur. Farklı cümle sayıları greedy’de 176 → 877, beam-3’te 164 → 813’tür. Buna karşılık parametre sayısı 5.653.004’ten 11.159.565’e ve arşiv decoder süreleri greedy’de 15,05 → 63,20, beam-3’te 45,96 → 182,65 ms/görüntüye çıkmıştır. Ortak eğitim ayarları eşleşse de model kapasitesi ve decoder yapısı da değiştiği için kazancın yalnız attention’dan kaynaklandığı iddia edilemez.

Kaynak: `results/val_table.md` ana tablosu ve model1/model2 `val_metrics_{greedy,beam3}.json` dosyaları. Bu karşılaştırma Model 1 için yeni deney yürütmez.

### Config kontrolü

`configs/model1.yaml` / `configs/model2.yaml` karşılaştırmasında farklı alanlar yalnız `name`, `model.type`, `model.num_layers` (Model 1: 1) ve `model.attention_dim` (Model 2: 512). Embedding 256, hidden 512, dropout 0,5 aynıdır. Şunlar birebir eşleşir:

- seed 42, ortak ResNet50 özellikleri, tam eğitim verisi (`train_fraction=1.0`);
- Adam lr=0.001; batch 32; üst sınır 30 epoch; grad clip 5; teacher forcing 1; AMP açık;
- padding yok sayılan çapraz entropi; val_loss erken durdurma, patience 5;
- checkpoint seçimi val CIDEr; cider_every 2; değerlendirme batch 64.

Ortak `base.yaml` aynı split, sözlük, seed ve ön işlemeyi tanımlar. Yerel checkpoint içindeki config de `model2.yaml` ile birebir eşleşir. Bunlar config eşitliğini doğrular; eğitimlerin her rastlantısal yürütme ayrıntısının aynı olduğunu kanıtlamaz. Gerçekte epoch sayıları farklıdır (Model 1: 12, Model 2: 11); seçilen checkpoint her ikisinde epoch 10’dur. Model 1 nn.LSTM, Model 2 attention + LSTMCell kullanır; parametre sayıları eşit değildir. Bu nedenle saf, parametre eşleştirilmiş bir attention ablation deneyi değildir.

## Nucleus sınırı ve koruma

Mevcut top-p=0.9, seed 42/43/44 sonuçları yeniden çalıştırılmadı ve değiştirilmedi. Decoder her görüntü için yerel CPU rastgele üretecini aynı koşu seed’ine sıfırlar; görüntü başına bağımsız seed akışları yoktur. Üç seed ortalaması ve ddof=1 standart sapması yalnız bu üç tekrarı özetler; güven aralığı, görüntü bazlı belirsizlik veya anlamsal çeşitlilik kanıtı değildir. `audit.json` içindeki nucleus dosya hash’leri deney sonunda yeniden karşılaştırıldı ve değişmedi.

## 40 görüntülük görsel hata incelemesi

Seçim 1.000 val ID üzerinde sabit tuzlu SHA-256 sıralamasının ilk 40 ID’sidir; kaliteye göre eleme yok. Tüm görüntüler, arşiv beam-3 tahminleri ve beşer referans ilk incelemede birlikte değerlendirildi. İşaretlenen kayıtların insan kararları annotations.json içinde korunur. Sayılar güncel review40.csv dosyasının altı kategori sütunundaki 1 değerlerinden hesaplanır; uncertain_labels alanındaki belirsiz etiketler kesin toplamlara katılmaz.

| Kategori | Kesin görüntü sayısı / 40 |
|---|---:|
| yanlış nesne | 11 |
| sayı | 0 |
| renk | 10 |
| eylem | 11 |
| genel ifade | 18 |
| kelime tekrarı | 4 |

Başlangıçta kullanıcı kontrolü bekleyen **18/18 kayıt tamamlandı**; bekleyen kayıt yok. 40 görüntünün **28** tanesinde en az bir kesin kategori etiketi, **12** tanesinde bu altı kategoride kesin etiket yok. **6 görüntüde belirsiz etiket** ayrı tutuldu; bunlar kesin kategori toplamlarına dahil edilmedi. Toplam **54 kategori ataması** var. Bir görüntü birden fazla kategoriye girebildiği için kategori sayıları örtüşür; toplamları farklı hatalı görüntü sayısı değildir. İnsan kontrolüne işaretlenmeyen kayıtların ilk görsel etiketleri korundu; sonuçlar bu 40 görüntüyü betimler, tüm val kümesine genellenmez.

**Kodlama kuralları:** Tekil bir aktörü anlatıp diğerini atlamak tek başına sayı hatası sayılmaz. Ana etkinliği doğru veren kısa açıklamalarda salt ayrıntı eksikliği hata sayılmaz. Belirsiz etiketler ayrı tutulur; kişi niteliği (boy/girl) notu altı kesin kategorinin dışındadır.

### Belirsiz olarak bırakılan kayıtlar

| Kayıt | Görüntü ID | Belirsiz etiket |
|---|---|---|
| 03 | 1865794069_6e3a1e57bb.jpg | belirsiz — sayı |
| 08 | 319847643_df7c2a1d25.jpg | belirsiz — eylem |
| 18 | 3265864834_e0229020dd.jpg | belirsiz — eylem |
| 23 | 3220126881_b0a4f7cccb.jpg | belirsiz — eylem |
| 38 | 3052038928_9f53aa2084.jpg | belirsiz — yanlış nesne |
| 40 | 2599903773_0f724d8f63.jpg | belirsiz — kişi niteliği (boy/girl) |

[Görüntü kimlikleri, tahminler, beş referans, son etiket ve gerekçe](../results/model2_val_study_20261112/review/review40.md) · [Sayım kaynağı CSV](../results/model2_val_study_20261112/review/review40.csv) · [Makine okunabilir özet](../results/model2_val_study_20261112/review/summary.json). CSV’de human_labels ve human_notes alanları insan kararlarını içerir. Fotoğrafları açmak için yerel Flickr8k kurulumu gerekir; veri seti teslim paketinde bulunmaz.

## Attention örnekleri

40 örnek sabit kaldı; aşağıdaki dört örnek hata türlerini ve bir doğru örneği göstermek için inceleme sonrasında seçildi, temsili istatistiksel örneklem değildir. Aynı checkpoint ve val önbelleğiyle beam-3 kelimelerine hizalı `[kelime_sayısı,49]` ağırlıklar çizildi. Her satır sonlu, negatif olmayan ve toplamı ≈1; görsel aynı Resize/CenterCrop RGB kırpımıdır. Attention doğruluk veya nedensel açıklama garantisi değildir.

- [01 — 3168354472_866fe70d36.jpg](../results/model2_val_study_20261112/attention/01_3168354472_866fe70d36_beam3.png): yanlış nesne, renk, eylem. Arşiv tahminiyle eşleşme: True; attention kaynağı: beam3 decoded path.
- [15 — 512045825_1be2083922.jpg](../results/model2_val_study_20261112/attention/15_512045825_1be2083922_beam3.png): ana içerik doğru. Arşiv tahminiyle eşleşme: True; attention kaynağı: beam3 decoded path.
- [29 — 3553225222_f5ebe44af1.jpg](../results/model2_val_study_20261112/attention/29_3553225222_f5ebe44af1_beam3.png): yanlış nesne, renk, genel ifade, kelime tekrarı. Arşiv tahminiyle eşleşme: True; attention kaynağı: beam3 decoded path.
- [33 — 2854959952_3991a385ab.jpg](../results/model2_val_study_20261112/attention/33_2854959952_3991a385ab_beam3.png): eylem, genel ifade, kelime tekrarı. Arşiv tahminiyle eşleşme: True; attention kaynağı: beam3 decoded path.

## Dosyalar ve tekrar çalıştırma

- `results/model2_val_study_20261112/beam5/`: yeni tahminler, üretim ölçümleri ve altı kalite metriği.
- `results/model2_val_study_20261112/beam3_recheck/`: aynı ortam/önbellekle kontrol koşusu.
- `comparison.json`, `comparison.csv`, `audit.json`, `data_provenance.json`: karşılaştırma ve izlenebilirlik.
- `review/`: seçim, ham metinler, ön etiketler, bütünleşik JSON/CSV/Markdown inceleme tablosu.
- `attention/`: dört PNG ve manifest; eski attention/nucleus dosyaları korunur.
- `report/tools/model2_val_study.py`: val hazırlama, seçim, rapor ve attention üretme aracı.

Veri, ZIP, özellik önbelleği ve checkpoint `data/` / `checkpoints/` içinde ve Git tarafından ignore edilir; stage edilmedi. Üretilen rapor/deney dosyaları untracked durumdadır. Mevcut sonuçlar ve `results/val_table.md` üzerine yazılmadı.

PowerShell, repo kökünden; aşağıdaki iki komut gerçekten çalıştırıldı. Mevcut çıktı dosyaları varken tekrar çalıştırmak hata verir; yeni koşu için farklı `--output-dir` kullanın:

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m src.predict --checkpoint checkpoints/model2_attention_lstm_full_20261104_01/best.pt --split val --method beam --beam-size 5 --features-dir data/features/model2_val_20261112 --output-dir results/model2_val_study_20261112/beam5 --device cuda
.\.venv\Scripts\python.exe -m src.evaluate --predictions results/model2_val_study_20261112/beam5/val_captions_beam5.json --model-name model2_attention_lstm --split val --method beam --beam-size 5 --output results/model2_val_study_20261112/beam5/val_metrics_beam5.json
```

Bu değerlendirme komutu Java olmadığı için `--meteor` kullanmadı. Deney tamamlandı; eğitim komutu veya `--allow-test` çalıştırılmadı.

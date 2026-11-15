# Flickr8k ile görüntü altyazılama · Üye C ara raporu

**Melih Demirtaş · 15 Kasım 2026 teslimi**
**İncelenen ekip sürümü** `main@7af69a9` · **Deney kümesi** 1.000 doğrulama görüntüsü, görüntü başına beş insan altyazısı.

## Araştırma sorusu ve yöntem

Bir görüntü için üretilen İngilizce altyazı, insan açıklamalarıyla ne kadar örtüşüyor? Aynı görüntü bölmesini ve değerlendirme kodunu kullanarak model mimarisinin ve kelime seçim yönteminin etkisini inceliyoruz. Greedy her adımda en yüksek puanlı kelimeyi seçer. Beam araması birkaç olası cümleyi birlikte izler. CIDEr, insan açıklamalarında ortak geçen ve görüntüyü ayırt eden kelime gruplarına ağırlık verir. Farklı cümle sayısı, bir modelin aynı ifadeyi kaç görüntüde tekrar kullandığını yorumlamaya yardımcı olur.

## Doğrulama bulguları

| Model ve koşu | BLEU-4 | CIDEr | Farklı cümle / 1.000 |
|---|---:|---:|---:|
| Model 1 · CNN-LSTM, greedy | 0,1063 | 0,2306 | 176 |
| Model 1 · CNN-LSTM, beam-3 | 0,1389 | 0,2778 | 164 |
| Model 2 · Attention LSTM, greedy | 0,1698 | 0,4409 | 877 |
| Model 2 · Attention LSTM, beam-3 | 0,2030 | 0,4697 | 813 |
| Model 2 · beam-3, takip deneyindeki tekrar | 0,2029 | 0,4701 | 813 |
| Model 2 · beam-5, aynı takip deneyi | 0,2061 | 0,4590 | 773 |

Model 2, greedy koşusunda Model 1'e göre daha yüksek CIDEr verdi ve aynı 1.000 görüntü için 701 daha fazla farklı cümle üretti. Sonuç, kelime üretirken ayrı görüntü bölgelerine bakmanın faydalı olabileceğini gösteriyor. İki modelin parametre sayısı da farklı olduğu için kazanımın tamamını attention mekanizmasına bağlayamıyoruz.

Ayça'nın aynı ortamda yaptığı takip deneyinde arama genişliği 3'ten 5'e çıkınca BLEU-4 biraz yükseldi, fakat CIDEr 0,4701'den 0,4590'a indi. Farklı cümle sayısı 40 azaldı ve görüntü başına üretim süresi yaklaşık 112 ms'den 194 ms'ye çıktı. Bu koşullarda beam-3, CIDEr ve süre açısından daha iyi seçim. Ayça ayrıca 40 görüntüyü insan gözüyle inceleyip dört örnek için kelime bazında attention görselleri hazırladı. [Takip deneyi ve hata incelemesi](model2_b_tamamlayici_20261112.md)

Üye C kontrolünde altı koşunun saklanan tahminleri, ortak bölmedeki görüntü kimlikleriyle eşleştirildi. Her koşuda BLEU-1–4, ROUGE-L ve CIDEr orijinal Flickr8k açıklamalarından yeniden hesaplandı. Altı metrik de kaydedilen değerleri yeniden üretti. METEOR bu kontrolde tekrar hesaplanmadı. [Ortak sonuç tablosu](../results/val_table.md) · [Beam karşılaştırması](../results/model2_val_study_20261112/comparison.csv)

## Üye C uygulaması ve deney durumu

**BLIP.** Salesforce'un hazır altyazılama modeli ham RGB görüntüyü kendi işlemcisiyle alır. Hazır ağırlıkları ve Flickr8k üzerinde uyarlanmış ağırlıkları karşılaştırmak için iki ayrı koşu tanımladık. Uyarlamada 6.000 eğitim görüntüsünün beşer açıklaması kullanılır. Görüntü kodlayıcısı ve metin çözücüsü AdamW ile birlikte güncellenir (öğrenme oranı 5×10⁻⁶, batch 8, en fazla 5 epoch). Her iki koşu aynı 1.000 doğrulama görüntüsü için greedy ve beam-3 tahminlerini, metrikleri, parametre sayısını ve görüntü başına süreyi kaydeder. BLIP süresine görüntü işleme de dahildir. Önbellek kullanan modellerin tahmin süresi ise hazır öznitelikten başlar. [BLIP kodu](../src/blip.py) · [Kaggle notebook’u](../notebooks/member_c_blip_kaggle.ipynb)

**Model 3.** Transformer, ortak ResNet50 önbelleğindeki 7×7 bölgeyi kullanır. Her bölgeyi 2.048 boyuttan 256'ya indirip öğrenilen satır ve sütun konumlarını ekler. Üç çözücü katmanının her birinde sekiz attention kafası bulunur. Nedensel maske, bir kelime tahmin edilirken sonraki kelimelerin görülmesini engeller. Son katmanın 49 bölge üzerindeki ağırlıkları, demoda üretilen kelimelerle hizalanabilir. Ortak eğitim, tahmin ve demo kodu aynı model kaydını kullanır. [Model](../src/models/transformer.py) · [Deney ayarları](../configs/model3.yaml)

**Yerel doğrulama.** Ortak sözlükle kurulan Model 3 çözücüsünde 5.016.588 parametre ölçüldü. İlerideki bir kelime değiştirildiğinde önceki tahminlerin sabit kaldığı, attention ağırlıklarının 49 bölge üzerinde bire toplandığı ve kaybın görüntü projeksiyonuna geri yayıldığı sınandı. Küçültülmüş bir BLIP modeli ile yerel checkpoint yükleme, gerçek işlemciyle batch hazırlama, geri yayılım, greedy ve beam-3 üretimi CPU üzerinde çalıştırıldı. Proje testlerinde 53 test geçti. Bu kontroller yazılım davranışını ölçer.

**15 Kasım Kaggle doğrulaması.** Hazır BLIP, ortak 1.000 doğrulama görüntüsü üzerinde greedy ile 0,2427 BLEU-4 ve 0,6337 CIDEr, beam-3 ile 0,2621 BLEU-4 ve 0,6833 CIDEr elde etti. Beam-3 ile CIDEr artışı 0,0496 oldu. Hazır BLIP'in beam-3 CIDEr değeri Model 2'nin aynı bölmedeki 0,4697 değerinden yüksek. BLIP dış verilerle önceden eğitildiği ve farklı bir görüntü kodlayıcısı kullandığı için bu fark yalnızca çözücü mimarisine bağlanamaz. METEOR bu koşuda hesaplanmadı. [Kaggle deney günlüğü, sürüm 2](https://www.kaggle.com/code/peluska/notebooka728ac7736/log?scriptVersionId=369724091)

**Devam eden deney.** BLIP, 6.000 Flickr8k görüntüsüyle beş epoch boyunca eğitildi. Beşinci epoch sonunda 3.750/3.750 eğitim adımı görüldü. Fine-tuning sonrası doğrulama dosyalarının Kaggle çıktısından aktarılması bekleniyor. Model 3 eğitimi hedef kapsamda bekliyor. Test bölmesi final değerlendirmesine ayrıldı.

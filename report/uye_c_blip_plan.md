# Üye C — 9 Kasım durum ve sıra

## Ekipte tamamlananlar
- Main ee2c0c7 kendi transformer-demo dalımıza fast-forward ile alındı.
- Model 1 ve Model 2 doğrulama sonuçları, METEOR ve birleşik tablo mevcut.
- Model 1 veri miktarı ve GloVe deneyleri mevcut.
- Demo iskeleti mevcut; Model 2 checkpoint paylaşımı Üye B'nin işi.

## Öncelik
1. BLIP: fine-tuning öncesi ve sonrası 1000 val görüntüsünde greedy ve beam-3.
2. BLIP'in gerçek ağırlıklarla demo kontrolü.
3. Zaman kalırsa Model 3: 3 katman/8 kafa/d_model 256, ortak eğitim betiğine kayıt.
4. Çalışan ortamın sürümlerini sabitleme, gerçek skorlarla README ve rapor.

## Bu değişiklikte hazırlananlar
- src/blip.py: iki koşu, sabit train/val bölmesi, beş referans, maskelenmiş padding,
  AdamW 5e-6, batch 8, dropout 0.1, AMP, clip 5, en fazla 5 epoch.
- Kayıp iyileşince/her iki epoch/son epoch val greedy CIDEr; best seçimi CIDEr.
- Encoder ve decoder birlikte eğitilir. BLIP kendi processor/tokenizer'ını kullanır.
- Generation: sampling kapalı, greedy beam=1, beam-3 beam=3, max_new_tokens=33,
  length_penalty=1. BLIP native beam kullanır; ortak LSTM beam uygulamasıyla skor
  sıralaması tamamen aynı değildir. Bu fark raporda belirtilmelidir.
- Çıktılar model4_blip_pretrained ve model4_blip_finetuned klasörlerine ayrı yazılır.
- Demo load/generate bağlandı, ham RGB görüntü kullanılır ve attention=None döner.
- Kaggle notebook'u hazırlandı; ortak requirements.txt denenmeden değiştirilmedi.

## Kaggle'da çalıştırma
Notebook dosyasını Kaggle'a import edin. GPU seçin, Internet'i açın.
Flickr8k veri setini Add Input ile ekleyin. Kod ZIP'ini özel bir Kaggle Dataset'e yükleyin
ve onu da Input olarak ekleyin. Notebook'un ilk hücresindeki yolları gerçek Input
konumlarına göre düzenleyin. Kurulumdan sonra hücreleri sırayla çalıştırın.
Kaggle GPU bulunabilirliği ve kota hesap üzerinden kontrol edilmelidir.

Yerel RTX 3060 6 GB ile full fine-tuning batch 8 için bellek yeterliliği denenmedi;
Kaggle kullanımı önerildi. Görüntüler yerel çalışma klasöründe yok.

## Doğrulama sınırı
Yerel testler yalnız veri/ayar/padding sözleşmesi ve mevcut tablo/metrik testleridir.
PyTorch/Transformers gerçek eğitim, generation ve Gradio model çağrısı burada
çalıştırılmadı. Notebook henüz çalıştırılmadı; BLIP için skor, eğitim süresi veya
parametre sayısı uydurulmadı. İlk Kaggle çalıştırması bu kontrolleri tamamlayacak.
Notebook Transformers 4.57.1 kullanır; bu bir deneme ortamı seçimi, bütün projenin
başarıyla denenmiş sürüm kilidi değildir. Çalışan pip freeze her koşuya yazılır.

Checkpoint best/last HF klasörleri her epoch kaydedilir. Optimizer resume bu sürümde
yok; GPU oturumu kesilirse ağırlıkları saklayın, tam tekrar için yeni çıktı klasörü
kullanın. Test kümesi için bu betikte seçenek yoktur. METEOR cümleleri İbrahim'e
iletilebilir. Checkpoint ve veri Git'e eklenmez.

Kaynak: https://huggingface.co/Salesforce/blip-image-captioning-base

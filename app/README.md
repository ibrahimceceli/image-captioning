# Gradio demosu

Kullanıcı bir fotoğraf yükler. Dört modelin altyazısı yan yana çıkar; her modelin yanında
çıkarım süresi (ms) ve parametre sayısı görünür. Attention destekleyen modellerde üretilen her
kelime için attention haritası çizilir. BLIP'in attention haritası yoktur.

## Çalıştırma

Depo kökünden:

```bash
python -m app.app
```

```bash
python -m app.app --device cpu --port 7861
```

- `--device`: `auto` (varsayılan; `app/config.yaml`), `cpu` ya da `cuda`. GPU eğitimle paylaşılıyorsa `cpu` kullanın: ResNet50 + Model 1 CPU'da görüntü başına yaklaşık 150–250 ms sürer.
- Adres: `http://127.0.0.1:7860` (varsayılan port). `--share` geçici bir genel Gradio bağlantısı açar.
- Bağımlılık: `gradio` (requirements.txt'te var). Gradio 6.31 ile denendi.
- Sol altta doğrulama kümesinden 6 örnek görüntü listelenir (`data/flickr8k/Images` varsa). Test kümesi kullanılmaz.

Arayüz:

- **Girdi:** fotoğraf (`gr.Image(type="pil")`), çözümleme seçimi (greedy / beam), beam genişliği (1–5, varsayılan 3; yalnız beam seçiliyken etkin) ve **Altyazı üret** düğmesi.
- **Model kutuları:** Model 1 CNN-LSTM, Model 2 Attention LSTM, Model 3 Transformer ve Model 4 BLIP için birer kutu. Her kutuda altyazı ile `⏱ toplam ms (encoder + decoder) · parametre` satırı bulunur. Yüklenemeyen bir model uygulamayı çökertmez; kutusunda `⚠️ checkpoint bulunamadı: <yol>` ya da `⚠️ henüz bağlanmadı: …` yazar.
- **Attention haritaları:** attention destekleyen modeller arasında seçim yapılır (`attention: true`). Tüm kelimeler tek bir figürde, ortak renk ölçeğiyle gösterilir. Figür `src/visualize_attention.py` ile, normalize edilmemiş 224×224 RGB kırpım (Resize(256) + CenterCrop(224)) üzerine çizilir. Model değiştirildiğinde altyazılar yeniden üretilmez; son sonuçtan çizilir.

Süreler tek görüntü içindir. Önbellek tabanlı modellerde (1, 2, 3) toplam süre, görüntü başına bir kez çalışan ortak ResNet50 encoder süresi ile modelin decoder süresinin toplamıdır. Parametre sayısına paylaşılan ResNet50 (23,508,032) dahil değildir. Bu yüzden Model 1 için 5,653,004 görünür.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `app/config.yaml` | **Tek kayıt yeri:** modellerin sırası, etiketleri, adaptör türleri (`kind`), checkpoint yolları, cihaz |
| `app/captioners.py` | Adaptör katmanı: `CaptionResult`, `PreparedImage`, `SharedEncoder`, `CheckpointCaptioner`, `TransformerCaptioner`, `BlipCaptioner` (yer tutucu), `CaptionerRegistry`, `register_adapter` |
| `app/app.py` | Gradio arayüzü |

## Adaptör arayüzü

```python
captioner.load()                                   # başlangıçta bir kez
captioner.generate(image, method, beam_size) -> CaptionResult
#   image: PIL.Image ya da PreparedImage; method: "greedy" | "beam"; beam_size: 1..5

CaptionResult(
    caption: str,                 # özel token'sız
    attention_words: list[str],   # caption.split()
    attention: Tensor | None,     # [len(words), 49], CPU float; attention yoksa None
    ms: float,                    # toplam çıkarım süresi
    params: int,                  # model parametresi (paylaşılan encoder hariç)
    detail: str = "",             # ör. "encoder 65 ms + decoder 56 ms"
)
```

- `load()` başarısız olursa `NotConnectedError("açık mesaj")` atmalıdır; mesaj olduğu gibi kutuda gösterilir. Beklenmeyen başka hatalar da yakalanır ve `yüklenemedi: …` olarak gösterilir.
- Önbellek tabanlı modeller (1, 2, 3) aynı `SharedEncoder`'ı kullanır. Bu nesne kilitli ön işlemeyi ve dondurulmuş ResNet50'yi `src/data/features.py`'den alır ve cihaz başına bir kez yüklenir. `CaptionerRegistry.run_all` her görüntüyü bir kez `PreparedImage`'a çevirir (`original`, `crop`, `features [1,2048,7,7]`, `encoder_ms`) ve bunu tüm modellere verir.
- `CheckpointCaptioner`, checkpoint'i `torch.load(weights_only=True)` ile açar ve modeli `src.train.build_model` ile kurar. Çözümleme `src.decode.generate_caption(..., return_attention=True)` ile yapılır.

## Üye C: Model 3 ve BLIP'i bağlama

### Model 3 (Transformer): `kind: transformer` → `TransformerCaptioner`

1. Modeli `src/train.py` içindeki `MODELS` sözlüğüne ekleyin (ör. `"transformer": CaptionTransformer`). Checkpoint, Model 1/2 ile aynı state'i taşımalı: `model`, `config` (`config["model"]["type"]` = MODELS anahtarı), `vocab_size`, `epoch`.
2. Forward sözleşmesi Model 1/2 ile aynıdır: `forward(features [B,2048,7,7] | [B,49,2048], tokens [B,T]) -> (logits [B,T,V], attention)`. `attention` şunlardan biri olabilir:
   - `[B, T, 49]`: son katman cross-attention'ı, kafalar ortalanmış;
   - `[B, T, H, 49]`: kafa başına; `TransformerCaptioner` kafaların ortalamasını alır;
   - `None`: harita gösterilmez.
3. `app/config.yaml` içindeki `model3.checkpoint` yolunu gerçek klasörle değiştirin (şu an yer tutucu: `checkpoints/model3_transformer/best.pt`).

Başka değişiklik gerekmez. Durum mesajları şöyle değişir:

- Checkpoint yoksa: `checkpoint bulunamadı`
- Tür MODELS'ta yoksa: `henüz bağlanmadı: model türü '…' src/train.py MODELS kaydında yok`

Farklı bir çözümleme gerekiyorsa `TransformerCaptioner.generate`'i ezin, ama `CaptionResult` döndürmeye devam edin.

### Model 4 (BLIP): `kind: blip` → `BlipCaptioner`

`app/captioners.py` içindeki `BlipCaptioner.load()` ve `generate()` metotlarını doldurun. Docstring'de çalışan bir taslak var.

- `needs_features = False`. BLIP paylaşılan ResNet50'yi ve 224 kırpımı kullanmaz. `generate` içinde `image.original` (ön işlenmemiş RGB PIL) alınır ve BLIP'in kendi `BlipProcessor`'ına verilir.
- Model kimliği `app/config.yaml` → `blip.model_id` alanındadır (şu an `Salesforce/blip-image-captioning-base`; fine-tune edilmiş bir yerel klasör de olabilir).
- Greedy için `num_beams=1`, beam için `num_beams=beam_size` verin. Süre, işlemci + `generate` için ölçülür (CUDA'da `_sync` kullanın). `params = count_params(model)`.
- Dönüş değeri: `CaptionResult(caption, caption.split(), None, ms, params, detail)`. `attention=None` olmalı ve `config.yaml`'da `attention: false` kalmalı.

### Yeni bir adaptör türü

`Captioner`'dan türetin, `@register_adapter("tür")` ile kaydedin ve `config.yaml`'da `kind: tür` yazın.

## Üye B: Model 2 attention formatı ve checkpoint

- Checkpoint'i şu yola koyun: **`checkpoints/model2_attention_lstm_full_20261104_01/best.pt`**. Yol `app/config.yaml` → `model2.checkpoint`'tan değiştirilebilir. Dosya yoksa kutuda `checkpoint bulunamadı: <yol>` yazar.
- Checkpoint `torch.load(weights_only=True)` ile açılabilmeli. `train.py`'nin yazdığı state (model, config, vocab_size, epoch, history…) bu koşulu sağlıyor.
- Demonun beklediği attention biçimi:
  - **kelime listesi** `caption.split()`;
  - **ağırlık** `[len(kelimeler), 49]`: negatif olmayan değerler, satır başına toplam ≈ 1; satır `i`, kelime `i`'ye aittir; bölge indeksi `satır*7 + sütun`, satır 0 üstte.

  `AttentionLSTM.forward`'ın döndürdüğü `[B,T,49]` ve `generate_caption(..., return_attention=True)` bu biçimi zaten üretiyor. Satır sayısı kelime sayısıyla eşleşmezse harita gösterilmez ve `attention döndürmedi` mesajı çıkar.
- `src/models/attention_captioner.py` (`AttentionCaptioner`) demoda kullanılmıyor. Demo aynı ön işlemeyi ve aynı checkpoint'i `CheckpointCaptioner` üzerinden kullanıyor; ek olarak attention'ı da alıyor.

## Doğrulama notları

- Model 1 (CPU, `1000268201_693b08cb0e.jpg`):
  - greedy: "a young girl wearing a pink shirt and blue jeans is walking on a sidewalk"
  - beam-3: "a little girl in a pink dress is walking through a park"
  - 5,653,004 parametre
- Gri tonlamalı (`L`) ve RGBA girdiler RGB'ye çevrilir.
- Attention yolu, rastgele ağırlıklı geçici bir `attention_lstm` checkpoint'iyle denendi (greedy ve beam). `[n,49]` biçimi doğrulandı, haritalar çizildi, Model 3 seçilince "checkpoint bulunamadı" mesajı göründü.

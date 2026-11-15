# Demo ve Ortam Doğrulama Raporu

**Tarih:** 15 Kasım 2026  
**Kapsam:** Python ortamı ve sabitlenmiş `requirements.txt` doğrulaması, Gradio demosunun (`app/app.py`, `app/captioners.py`, `app/config.yaml`) gerçek yerel checkpoint'ler ve yerel HuggingFace önbelleğindeki `Salesforce/blip-image-captioning-base` ağırlıklarıyla 4/4 model için CPU üzerinde uçtan uca sınanması, attention haritası üretiminin kontrolü ve BLIP (`model4`) yerel durum tespiti.  
**Çalıştırma kısıtı:** Tüm doğrulamalar GPU kullanılmadan (`CUDA_VISIBLE_DEVICES="-1"`, `device="cpu"`, `HF_HUB_OFFLINE="1"`, `TRANSFORMERS_OFFLINE="1"`, `PYTHONUTF8="1"`) çalıştırılmıştır (`torch.cuda.is_available() == False`).

---

## 1. Python Ortamı ve Sabitlenen Paket Sürümleri (`requirements.txt`)

Aktif yerel ortamda doğrulanan sürümler:

- **Python:** `3.12.3` (`tags/v3.12.3:f6650f9, Apr 9 2024, 14:05:25 [MSC v.1938 64 bit (AMD64)]`)
- **PyTorch / CUDA derlemesi:** `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124` (`torch.version.cuda == "12.4"`)

[requirements.txt](../requirements.txt) dosyası aşağıdaki sabit (`==`) sürümlerle güncellenmiştir:

| Paket | `requirements.txt` sabiti | Yerel kurulu sürüm | Not |
|---|---|---|---|
| `torch` | `torch==2.6.0` | `2.6.0+cu124` | GPU için `--index-url https://download.pytorch.org/whl/cu124` yorum satırı olarak eklendi; standart PyPI kurulumunu bozmamak için `+cu124` yerel etiketi satır dışı bırakıldı. |
| `torchvision` | `torchvision==0.21.0` | `0.21.0+cu124` | Yukarıdaki CUDA 12.4 index açıklamasıyla aynı. |
| `transformers` | `transformers==5.17.0` | `5.17.0` | — |
| `pillow` | `pillow==10.4.0` | `10.4.0` | — |
| `numpy` | `numpy==1.26.4` | `1.26.4` | — |
| `PyYAML` | `PyYAML==6.0.1` | `6.0.1` | — |
| `tqdm` | `tqdm==4.66.4` | `4.66.4` | — |
| `matplotlib` | `matplotlib==3.9.1` | `3.9.1` | PyPI'de `3.9.1` Windows wheel'leri geri çekildiği (`yanked`) için temiz Windows `.venv` kurulumunda `matplotlib==3.9.1.post1` kullanılabileceği yorum satırında not edildi. |
| `gradio` | `gradio==6.31.0` | `6.31.0` | — |
| `pycocoevalcap` | `pycocoevalcap==1.2` | `1.2` | — |
| `pytest` | `pytest==9.1.1` | `9.1.1` | Doğrulama testleri için eklendi. |

**Ek bulgular:**
- **`timm`:** İlk iskeletteki `requirements.txt` dosyasında yer alan `timm`, aktif yerel ortamda kurulu değildir ve `src/`, `app/`, `tests/`, `report/`, `notebooks/` altındaki Python dosyalarının hiçbirinde import edilmemektedir. Bu nedenle bağımlılık listesinden çıkarılıp dosya başındaki yorumda belirtilmiştir.
- **Bağımlılık tutarlılığı:** `python -m pip check` çıktısı `No broken requirements found.` döndürmüştür. `python -m pip install --dry-run -r requirements.txt` komutu çıkış kodu `0` ile tüm paketlerin (`torch==2.6.0` → `2.6.0+cu124` ve `torchvision==0.21.0` → `0.21.0+cu124` dahil) sağlandığını doğrulamıştır.

---

## 2. Gerçek Checkpoint'lerle CPU Üzerinde Demo Doğrulaması

`CaptionerRegistry(config_path="app/config.yaml", device="cpu")` nesnesi çevrimdışı HuggingFace bayraklarıyla (`HF_HUB_OFFLINE="1"`, `TRANSFORMERS_OFFLINE="1"`, `CUDA_VISIBLE_DEVICES="-1"`) başlatılıp `load_all()` çağrılmıştır. `Salesforce/blip-image-captioning-base` ağırlıklarının yerel HuggingFace önbelleğinde (`~/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-base`) bulunmasıyla birlikte **4/4 model** CPU üzerinde `ready=True` ve `status="hazır"` durumuna geçmiştir.

### 2.1. Kayıt ve Checkpoint Durumu

- **Cihaz:** `cpu`
- **Ortak encoder:** `ResNet50 encoder hazır (23,508,032 parametre, cpu)` (yerel önbellek: `~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth`)

| Anahtar | Etiket | Checkpoint / Kaynak | Epoch / Tür | `ready` | Parametre | `status` |
|---|---|---|---|---|---:|---|
| `model1` | Model 1 · CNN-LSTM | `checkpoints/model1_lstm/best.pt` | epoch 10 (`lstm`, vocab 2572) | `True` | 5,653,004 | `hazır` |
| `model2` | Model 2 · Attention LSTM | `checkpoints/model2_attention_lstm_full_20261104_01/best.pt` | epoch 10 (`attention_lstm`, vocab 2572) | `True` | 11,159,565 | `hazır` |
| `model3` | Model 3 · Transformer | `checkpoints/model3_transformer/best.pt` | epoch 8 (`transformer`, vocab 2572) | `True` | 5,016,588 | `hazır` |
| `blip` | Model 4 · BLIP (fine-tune edilmeden) | `Salesforce/blip-image-captioning-base` (yerel HF önbelleği) | ön eğitimli (`BlipForConditionalGeneration`) | `True` | 223,971,644 | `hazır` |

### 2.2. Tek Doğrulama Görüntüsü Üzerinde `run_all` Çıktıları

Test görüntüsü olarak `app/app.py` içindeki `example_images()` işlevinin döndürdüğü ilk doğrulama görüntüsü `data/flickr8k/Images/1015584366_dfcec3c85a.jpg` (`500×375` RGB → `PreparedImage.crop`: `224×224` RGB, `features`: `[1, 2048, 7, 7]`) kullanılmıştır. (Not: [app/README.md](../app/README.md) içinde anılan `1000268201_693b08cb0e.jpg` ortak bölmede `train` kümesinde yer aldığı için test amacıyla kullanılmamış, doğrulama kümesindeki ilk örnek seçilmiştir.)

`data/flickr8k/captions.txt` içindeki referans açıklamalar:
- *A black dog leaps over a log .*
- *A grey dog is leaping over a fallen tree .*
- *A large black dog leaps a fallen log .*
- *A mottled black and grey dog in a blue collar jumping over a fallen tree .*
- *The black dog jumped the tree stump .*

**Doğrudan `registry.run_all(img, method, 3)` ölçümleri (CPU):**

*(Model 1–3 değerleri ilk doğrulamadan korunmuştur; `blip` satırları ve alt not yerel HF önbelleği sonrası yapılan 4/4 CPU doğrulamasından eklenmiştir.)*

| Model | Çözme | Üretilen Altyazı | Kelime / `attention` Şekli | Süre (`ms`) | Süre Dökümü (`detail`) | Parametre |
|---|---|---|---|---:|---|---:|
| `model1` | `greedy` | `a dog is jumping over a hurdle` | 7 / `None` | 66.22 | `encoder 46 ms + decoder 20 ms` | 5,653,004 |
| `model2` | `greedy` | `a black dog is jumping over a fallen tree` | 9 / `(9, 49)` | 157.68 | `encoder 46 ms + decoder 112 ms` | 11,159,565 |
| `model3` | `greedy` | `a dog is jumping over a log` | 7 / `(7, 49)` | 81.22 | `encoder 46 ms + decoder 35 ms` | 5,016,588 |
| `blip` | `greedy` | `a dog is sitting on a log in the grass` | 10 / `None` | 2623.60 | `BLIP: Salesforce/blip-image-captioning-base` | 223,971,644 |
| `model1` | `beam-3` | `a black and white dog is jumping over a hurdle` | 10 / `None` | 127.73 | `encoder 56 ms + decoder 72 ms` | 5,653,004 |
| `model2` | `beam-3` | `a black and white dog is running through the woods` | 10 / `(10, 49)` | 545.19 | `encoder 56 ms + decoder 489 ms` | 11,159,565 |
| `model3` | `beam-3` | `a dog is jumping over a log` | 7 / `(7, 49)` | 166.28 | `encoder 56 ms + decoder 110 ms` | 5,016,588 |
| `blip` | `beam-3` | `a dog is sitting on a log in the grass` | 10 / `None` | 1807.90 | `BLIP: Salesforce/blip-image-captioning-base` | 223,971,644 |

**Sözleşme ve ek kenar durumu kontrolleri:**
- 4/4 modelin birlikte çalıştırıldığı güncel CPU doğrulama turunda Model 1–3 aynı altyazıları, kelime sayılarını, attention şekillerini ve satır toplamı sapma sınırlarını (`1.19e-07` / `5.96e-08`) üretmiştir (bu turda ölçülen CPU süreleri `greedy` için `model1`: `143.27 ms` (`encoder 91 ms + decoder 52 ms`), `model2`: `300.45 ms` (`encoder 91 ms + decoder 209 ms`), `model3`: `147.13 ms` (`encoder 91 ms + decoder 56 ms`), `blip`: `2623.60 ms`; `beam-3` için `model1`: `272.96 ms` (`encoder 138 ms + decoder 135 ms`), `model2`: `1213.87 ms` (`encoder 138 ms + decoder 1076 ms`), `model3`: `334.72 ms` (`encoder 138 ms + decoder 197 ms`), `blip`: `1807.90 ms`).
- `CaptionResult.attention_words == caption.split()` tüm 4 modelin çıktılarında sağlandı (`blip` için 10 kelime ve `attention=None`).
- `model2` ve `model3` attention tensörlerinin tüm elemanları sonlu ve negatif olmayan (`>= 0`) değerlerden oluşmaktadır; satır toplamlarının `1.0`'dan en büyük mutlak sapması `model2` için `1.19e-07`, `model3` için `5.96e-08` ölçülmüştür.
- `TransformerCaptioner.postprocess_attention` sentetik `[4, 8, 49]` çok-kafalı giriş üzerinde ayrıca sınanmış, `(4, 49)` şekline indirgenip satır toplamlarının `1.0` kaldığı (`max_row_err = 0.0`) doğrulanmıştır.
- Gri tonlamalı (`L`) ve saydam kanallı (`RGBA`) kopyalar `registry.run_all(..., "greedy", 3)` ile çalıştırılmış, RGB'ye çevrilerek 4/4 modelde hatasız biçimde geçerli altyazı üretilmiştir: `model3` için `(7, 49)` attention çıktısı, `blip` için `a dog is sitting on a log in the grass` altyazısı ve `attention=None` (`L`: `1421.87 ms`, `RGBA`: `1643.57 ms`) döndürülmüştür.

---

## 3. Gradio Arayüzü, Geri Çağırma İşlevleri ve Attention Figürü Doğrulaması

Sunucu başlatılmadan (`launch()` çağrılmadan) [app/app.py](../app/app.py) arayüz kurucusu ve tüm olay işleyicileri doğrudan Python içinden test edilmiştir:

- `example_images()`: Doğrulama kümesinden 6 geçerli `.jpg` yolu döndürdü (`1015584366_dfcec3c85a.jpg`, `102455176_5f8ead62d5.jpg`, `1028205764_7e8df9a2ea.jpg`, `1067180831_a59dc64344.jpg`, `10815824_2997e03d76.jpg`, `1082252566_8c79beef93.jpg`).
- `build_demo(registry)` ve `.queue(default_concurrency_limit=1)`: `<class 'gradio.blocks.Blocks'>` nesnesini 4 olay işleyicisiyle (`load_example`, `on_method`, `on_generate`, attention seçimi `<lambda>`) hatasız oluşturdu.
- `idle_meta` ve `render_attention(None, ...)`: Tüm 4 model (`model1`, `model2`, `model3`, `blip`) için `✅ hazır · **...** parametre` durumunu üretti (`blip` için `✅ hazır · **223,971,644** parametre`; başlangıç `render_attention(None, ...)` mesajı `"Önce bir fotoğraf yükleyip **Altyazı üret**'e basın."` ve `None`).
- `on_method`: `"beam"` seçildiğinde `Beam genişliği` kaydırıcısını `interactive=True`, `"greedy"` seçildiğinde `interactive=False` döndürdü.
- `on_generate(None, "greedy", 3, "model2")`: Boş görüntüde `UserWarning: Önce bir fotoğraf yükleyin.` uyarısı verip tüm 11 çıkış bileşeni için `gr.skip()` (`{'__type__': 'update'}`) döndürdü.
- `on_generate(img, "greedy", 3, "model2")` ve `on_generate(img, "beam", 3, "model2")`:
  - **4/4 model kutusunun tamamı metin ve süre/parametre özeti üretti** (ilk doğrulamada Model 1–3 CPU süreleri `model1`: `49 ms` / `82 ms`, `model2`: `108 ms` / `385 ms`, `model3`: `56 ms` / `120 ms` ölçülmüştü; güncel 4/4 doğrulamada tüm modellerin çıktıları aşağıdadır):
    - `greedy`:
      - `model1`: `a dog is jumping over a hurdle` (`⏱ **102 ms** (encoder 81 ms + decoder 21 ms) · **5,653,004** parametre`)
      - `model2`: `a black dog is jumping over a fallen tree` (`⏱ **340 ms** (encoder 81 ms + decoder 260 ms) · **11,159,565** parametre`)
      - `model3`: `a dog is jumping over a log` (`⏱ **155 ms** (encoder 81 ms + decoder 74 ms) · **5,016,588** parametre`)
      - `blip`: `a dog is sitting on a log in the grass` (`⏱ **1440 ms** (BLIP: Salesforce/blip-image-captioning-base) · **223,971,644** parametre`)
    - `beam` (`beam_size=3`):
      - `model1`: `a black and white dog is jumping over a hurdle` (`⏱ **215 ms** (encoder 99 ms + decoder 116 ms) · **5,653,004** parametre`)
      - `model2`: `a black and white dog is running through the woods` (`⏱ **1300 ms** (encoder 99 ms + decoder 1202 ms) · **11,159,565** parametre`)
      - `model3`: `a dog is jumping over a log` (`⏱ **399 ms** (encoder 99 ms + decoder 301 ms) · **5,016,588** parametre`)
      - `blip`: `a dog is sitting on a log in the grass` (`⏱ **1979 ms** (BLIP: Salesforce/blip-image-captioning-base) · **223,971,644** parametre`)
  - `src/visualize_attention.py` (`visualize_attention`) ve `figure_to_array` üzerinden üretilen `model2` attention figürleri `(1139, 1400, 3)` boyutunda ve `uint8` RGB NumPy dizisi olarak döndü.
- Attention seçim işleyicisi (`<lambda>(state, key)`):
  - `model3` seçildiğinde yeniden çıkarım yapmadan mevcut `state` üzerinden `(760, 1400, 3)` boyutunda `uint8` RGB attention figürü ve `**Model 3 · Transformer** — *a dog is jumping over a log* · 7 kelime · greedy · 224×224 kırpım üzerinde, ortak renk ölçeği` başlığını üretti.
  - `model1` seçildiğinde `**Model 1 · CNN-LSTM**: bu model attention ağırlığı döndürmedi.` mesajını ve `None` görseli döndürdü.

---

## 4. BLIP (`model4`) Yerel Demo Durumu ve Birim Testleri

1. **Kaggle fine-tune ağırlıkları hâlâ yerel diskte yok:** Kaggle'da eğitilen Model 4 (`model4_blip_finetuned`) checkpoint klasörü `checkpoints/model4_blip/best` yerel `checkpoints/` dizinine aktarılmamıştır (`checkpoints/` altında yalnızca `model1_*`, `model2_attention_lstm_full_20261104_01` ve `model3_transformer` klasörleri bulunmaktadır). Bu nedenle demo yapılandırmasındaki `blip` girişi fine-tune edilmemiş `Model 4 · BLIP (fine-tune edilmeden)` etiketini korumaktadır.
2. **Önceden eğitilmiş HF modeli yerel önbellekte mevcut (4/4 model çalışır durumda):** [app/config.yaml](../app/config.yaml) içinde `blip.model_id: Salesforce/blip-image-captioning-base` tanımlıdır ve yerel HuggingFace önbelleğinde (`~/.cache/huggingface/hub`) artık `models--Salesforce--blip-image-captioning-base` (yanında `models--Qwen--Qwen2.5-Coder-1.5B-Instruct`) yer almaktadır. Çevrimdışı ortamda (`HF_HUB_OFFLINE="1"`, `TRANSFORMERS_OFFLINE="1"`) `CaptionerRegistry.load_all()` bu ağırlıkları ağ erişimi olmadan yükleyerek `blip` modelini CPU üzerinde `ready=True`, `223,971,644` parametre ve `status="hazır"` durumuna geçirmiş; böylece 4/4 modelin Gradio demosunda uçtan uca metin döndürmesi doğrulanmıştır.
3. **ResNet50 paylaşımı dışı mimari ve sözleşme:** `BlipCaptioner.needs_features = False` olduğu için paylaşılan ResNet50 özniteliklerini kullanmaz; `PreparedImage.original` (ön işlenmemiş RGB) üzerinden kendi `BlipProcessor` ön işlemesini uygular, `attention=None` (`app/config.yaml` içinde `attention: false`), `attention_words == caption.split()` ve `detail="BLIP: Salesforce/blip-image-captioning-base"` döndürür.
4. **`BlipCaptioner` adaptör birim doğrulaması:**
   - Eksik yerel klasör (`checkpoints/model4_blip/best`) verildiğinde `NotConnectedError("BLIP checkpoint bulunamadı: checkpoints/model4_blip/best")` yükseltildiği önceki testte doğrulanmıştır.
   - `TemporaryDirectory` içinde oluşturulan küçük sentetik `save_pretrained` BLIP ağırlığıyla (`64,006` parametre) `BlipCaptioner.load()` ve `generate(..., "greedy")` / `generate(..., "beam", 3)` çağrılarının CPU üzerinde `attention=None` ile çalıştığı doğrulanmıştır.

### `pytest` Test Sonuçları (CPU)

- Hedefli çalıştırma (`python -m pytest tests/test_blip.py tests/test_blip_runtime.py tests/test_transformer.py -v`, `CUDA_VISIBLE_DEVICES="-1"`): **6 passed in 17.08s**
- Tam paket (`python -m pytest tests -v`, `CUDA_VISIBLE_DEVICES="-1"`): **53 passed, 14 subtests passed in 25.72s**

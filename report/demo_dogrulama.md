# Demo ve Ortam Doğrulama Raporu

**Tarih:** 15 Kasım 2026  
**Kapsam:** Python ortamı ve sabitlenmiş `requirements.txt` doğrulaması, Gradio demosunun (`app/app.py`, `app/captioners.py`, `app/config.yaml`) gerçek yerel checkpoint'ler ile CPU üzerinde uçtan uca sınanması, attention haritası üretiminin kontrolü ve BLIP (`model4`) yerel durum tespiti.  
**Çalıştırma kısıtı:** Tüm doğrulamalar GPU kullanılmadan (`CUDA_VISIBLE_DEVICES=""`, `device="cpu"`) çalıştırılmıştır.

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

`CaptionerRegistry(config_path="app/config.yaml", device="cpu")` nesnesi çevrimdışı HuggingFace bayraklarıyla (`HF_HUB_OFFLINE="1"`, `TRANSFORMERS_OFFLINE="1"`, `CUDA_VISIBLE_DEVICES=""`) başlatılıp `load_all()` çağrılmıştır.

### 2.1. Kayıt ve Checkpoint Durumu

- **Cihaz:** `cpu`
- **Ortak encoder:** `ResNet50 encoder hazır (23,508,032 parametre, cpu)` (yerel önbellek: `~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth`)

| Anahtar | Etiket | Checkpoint / Kaynak | Epoch / Tür | `ready` | Parametre | `status` |
|---|---|---|---|---|---:|---|
| `model1` | Model 1 · CNN-LSTM | `checkpoints/model1_lstm/best.pt` | epoch 10 (`lstm`, vocab 2572) | `True` | 5,653,004 | `hazır` |
| `model2` | Model 2 · Attention LSTM | `checkpoints/model2_attention_lstm_full_20261104_01/best.pt` | epoch 10 (`attention_lstm`, vocab 2572) | `True` | 11,159,565 | `hazır` |
| `model3` | Model 3 · Transformer | `checkpoints/model3_transformer/best.pt` | epoch 8 (`transformer`, vocab 2572) | `True` | 5,016,588 | `hazır` |
| `blip` | Model 4 · BLIP (fine-tune edilmeden) | `Salesforce/blip-image-captioning-base` | — | `False` | `None` | `yüklenemedi: OSError: Can't load image processor for 'Salesforce/blip-image-captioning-base'...` |

### 2.2. Tek Doğrulama Görüntüsü Üzerinde `run_all` Çıktıları

Test görüntüsü olarak `app/app.py` içindeki `example_images()` işlevinin döndürdüğü ilk doğrulama görüntüsü `data/flickr8k/Images/1015584366_dfcec3c85a.jpg` (`500×375` RGB → `PreparedImage.crop`: `224×224` RGB, `features`: `[1, 2048, 7, 7]`) kullanılmıştır. (Not: [app/README.md](../app/README.md) içinde anılan `1000268201_693b08cb0e.jpg` ortak bölmede `train` kümesinde yer aldığı için test amacıyla kullanılmamış, doğrulama kümesindeki ilk örnek seçilmiştir.)

`data/flickr8k/captions.txt` içindeki referans açıklamalar:
- *A black dog leaps over a log .*
- *A grey dog is leaping over a fallen tree .*
- *A large black dog leaps a fallen log .*
- *A mottled black and grey dog in a blue collar jumping over a fallen tree .*
- *The black dog jumped the tree stump .*

**Doğrudan `registry.run_all(img, method, 3)` ölçümleri (CPU):**

| Model | Çözme | Üretilen Altyazı | Kelime / `attention` Şekli | Süre (`ms`) | Süre Dökümü (`detail`) | Parametre |
|---|---|---|---|---:|---|---:|
| `model1` | `greedy` | `a dog is jumping over a hurdle` | 7 / `None` | 66.22 | `encoder 46 ms + decoder 20 ms` | 5,653,004 |
| `model2` | `greedy` | `a black dog is jumping over a fallen tree` | 9 / `(9, 49)` | 157.68 | `encoder 46 ms + decoder 112 ms` | 11,159,565 |
| `model3` | `greedy` | `a dog is jumping over a log` | 7 / `(7, 49)` | 81.22 | `encoder 46 ms + decoder 35 ms` | 5,016,588 |
| `model1` | `beam-3` | `a black and white dog is jumping over a hurdle` | 10 / `None` | 127.73 | `encoder 56 ms + decoder 72 ms` | 5,653,004 |
| `model2` | `beam-3` | `a black and white dog is running through the woods` | 10 / `(10, 49)` | 545.19 | `encoder 56 ms + decoder 489 ms` | 11,159,565 |
| `model3` | `beam-3` | `a dog is jumping over a log` | 7 / `(7, 49)` | 166.28 | `encoder 56 ms + decoder 110 ms` | 5,016,588 |

**Sözleşme ve ek kenar durumu kontrolleri:**
- `CaptionResult.attention_words == caption.split()` tüm sonuçlarda sağlandı.
- `model2` ve `model3` attention tensörlerinin tüm elemanları sonlu ve negatif olmayan (`>= 0`) değerlerden oluşmaktadır; satır toplamlarının `1.0`'dan en büyük mutlak sapması `model2` için `1.19e-07`, `model3` için `5.96e-08` ölçülmüştür.
- `TransformerCaptioner.postprocess_attention` sentetik `[4, 8, 49]` çok-kafalı giriş üzerinde ayrıca sınanmış, `(4, 49)` şekline indirgenip satır toplamlarının `1.0` kaldığı doğrulanmıştır.
- Gri tonlamalı (`L`) ve saydam kanallı (`RGBA`) kopyalar `registry.run_all(..., "greedy", 3)` ile çalıştırılmış, RGB'ye çevrilerek hatasız biçimde geçerli altyazı ve `(7, 49)` attention çıktısı vermiştir (`model3`).

---

## 3. Gradio Arayüzü, Geri Çağırma İşlevleri ve Attention Figürü Doğrulaması

Sunucu başlatılmadan (`launch()` çağrılmadan) [app/app.py](../app/app.py) arayüz kurucusu ve tüm olay işleyicileri doğrudan Python içinden test edilmiştir:

- `example_images()`: Doğrulama kümesinden 6 geçerli `.jpg` yolu döndürdü (`1015584366_dfcec3c85a.jpg`, `102455176_5f8ead62d5.jpg`, `1028205764_7e8df9a2ea.jpg`, `1067180831_a59dc64344.jpg`, `10815824_2997e03d76.jpg`, `1082252566_8c79beef93.jpg`).
- `build_demo(registry)` ve `.queue(default_concurrency_limit=1)`: `<class 'gradio.blocks.Blocks'>` nesnesini 4 olay işleyicisiyle (`load_example`, `on_method`, `on_generate`, attention seçimi `<lambda>`) hatasız oluşturdu.
- `idle_meta` ve `render_attention(None, ...)`: Model 1–3 için `✅ hazır`, BLIP için `⚠️ yüklenemedi: OSError: ...` durumunu üretti.
- `on_method`: `"beam"` seçildiğinde `Beam genişliği` kaydırıcısını `interactive=True`, `"greedy"` seçildiğinde `interactive=False` döndürdü.
- `on_generate(None, "greedy", 3, "model2")`: Boş görüntüde `UserWarning: Önce bir fotoğraf yükleyin.` uyarısı verip tüm 11 çıkış bileşeni için `gr.skip()` döndürdü.
- `on_generate(img, "greedy", 3, "model2")` ve `on_generate(img, "beam", 3, "model2")`:
  - Dört model kutusunun metin ve süre/parametre özetlerini üretti (ikinci çağrı turunda CPU süreleri `model1`: `49 ms` / `82 ms`, `model2`: `108 ms` / `385 ms`, `model3`: `56 ms` / `120 ms`; `blip`: metin `'—'` ve `⚠️ yüklenemedi: ...`).
  - `src/visualize_attention.py` (`visualize_attention`) ve `figure_to_array` üzerinden üretilen `model2` attention figürleri `(1139, 1400, 3)` boyutunda ve `uint8` RGB NumPy dizisi olarak döndü.
- Attention seçim işleyicisi (`<lambda>(state, key)`):
  - `model3` seçildiğinde yeniden çıkarım yapmadan mevcut `state` üzerinden `(760, 1400, 3)` boyutunda `uint8` RGB attention figürü ve `**Model 3 · Transformer** — *a dog is jumping over a log* · 7 kelime · ...` başlığını üretti.
  - `model1` seçildiğinde `**Model 1 · CNN-LSTM**: bu model attention ağırlığı döndürmedi.` mesajını ve `None` görseli döndürdü.

---

## 4. BLIP (`model4`) Yerel Demo Durumu ve Birim Testleri

1. **Kaggle fine-tune ağırlıkları yerel diskte yok:** Kaggle'da eğitilen Model 4 (`model4_blip_finetuned`) checkpoint klasörü `checkpoints/model4_blip/best` yerel `checkpoints/` dizinine aktarılmamıştır (`checkpoints/` altında yalnızca `model1_*`, `model2_attention_lstm_full_20261104_01` ve `model3_transformer` bulunmaktadır).
2. **Önceden eğitilmiş HF modeli yerel önbellekte yok:** [app/config.yaml](../app/config.yaml) içinde `blip.model_id: Salesforce/blip-image-captioning-base` tanımlıdır; ancak yerel HuggingFace önbelleğinde (`~/.cache/huggingface/hub`) yalnızca `models--Qwen--Qwen2.5-Coder-1.5B-Instruct` bulunmakta, `Salesforce/blip-image-captioning-base` yer almamaktadır.
3. **Çökmesiz hata yönetimi:** Ağdan ~1 GB ağırlık indirilmeyen çevrimdışı ortamda `CaptionerRegistry.load_all()`, `blip` kutusunu `ready=False` ve `yüklenemedi: OSError: ...` durumuna alarak uygulamanın Model 1–3 ile normal çalışmasını sürdürmesini sağlar.
4. **`BlipCaptioner` adaptör doğrulaması:**
   - Eksik yerel klasör (`checkpoints/model4_blip/best`) verildiğinde `NotConnectedError("BLIP checkpoint bulunamadı: checkpoints/model4_blip/best")` yükseltildiği doğrulandı.
   - `TemporaryDirectory` içinde oluşturulan küçük sentetik `save_pretrained` BLIP ağırlığıyla (`64,006` parametre) `BlipCaptioner.load()` ve `generate(..., "greedy")` / `generate(..., "beam", 3)` çağrılarının CPU üzerinde `attention=None` ile çalıştığı doğrulandı.

### `pytest` Test Sonuçları (CPU)

- Hedefli çalıştırma (`python -m pytest tests/test_blip.py tests/test_blip_runtime.py tests/test_transformer.py -v`): **6 passed in 10.55s**
- Tam paket (`python -m pytest tests -v`, `CUDA_VISIBLE_DEVICES=""`): **53 passed, 14 subtests passed in 17.26s**

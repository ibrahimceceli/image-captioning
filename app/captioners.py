"""Demo adaptör katmanı: modeller bir kez yüklenir, her istek aynı nesnelerle çalışır.

Ortak arayüz (tüm adaptörler):

    captioner.load()                                   # bir kez; hata -> kutuda mesaj
    captioner.generate(image, method, beam_size) -> CaptionResult

    image      : PIL.Image (her boyut/mod) ya da CaptionerRegistry.prepare() çıktısı PreparedImage
    method     : "greedy" | "beam"
    beam_size  : 1..5 (greedy'de yok sayılır; beam 1 greedy ile aynıdır)

CaptionResult:
    caption          : str, özel token'sız
    attention_words  : list[str] == caption.split()
    attention        : torch.Tensor [len(words), 49] CPU float ya da None (attention yoksa)
                       satır i, words[i] üretilirken 7x7 bölgeler üzerindeki ağırlık;
                       bölge indeksi satır*7+sütun, satır 0 üstte (src/visualize_attention.py)
    ms               : toplam çıkarım süresi (önbellek tabanlı modellerde encoder + decoder)
    params           : modelin parametre sayısı (paylaşılan ResNet50 encoder hariç)
    detail           : süre dökümü gibi kısa açıklama

Önbellek tabanlı modeller (1, 2, 3) dondurulmuş ResNet50'yi paylaşır: SharedEncoder cihaz
başına bir kez yüklenir, görüntü başına bir kez çalışır (PreparedImage). BLIP kendi işlemcisini
kullanır ve PreparedImage.original'ı (ön işlenmemiş RGB) alır.

Yeni bir adaptör türü: Captioner'dan türet, @register_adapter("tür") ile kaydet,
app/config.yaml'da kind: tür yaz. Yalnız güvenilir checkpoint'leri yükleyin.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch import Tensor, nn
from torchvision import transforms

from src.data.config import load_config, repo_root

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.yaml"
METHODS = ("greedy", "beam")
MAX_BEAM = 5


class NotConnectedError(RuntimeError):
    """Model henüz bağlanmadı ya da dosyası yok. Mesaj olduğu gibi arayüzde gösterilir."""


@dataclass
class CaptionResult:
    caption: str
    attention_words: list[str]
    attention: Tensor | None
    ms: float
    params: int
    detail: str = ""


@dataclass
class PreparedImage:
    """Bir görüntünün tüm modeller için ortak hazırlığı.

    original: RGB PIL, ön işlemesiz (BLIP için).
    crop:     Resize(256)+CenterCrop(224) RGB PIL, normalize öncesi; attention bunun üzerine çizilir.
    features: [1, 2048, 7, 7] float32, encoder cihazında (encoder yüklenmediyse None).
    """

    original: Image.Image
    crop: Image.Image
    features: Tensor | None
    encoder_ms: float


def count_params(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def resolve_device(name: str | None) -> torch.device:
    if name in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root() / path


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


@lru_cache(maxsize=1)
def get_vocab():
    from src.data.vocab import Vocabulary

    return Vocabulary.load(repo_path(load_config()["vocab_path"]))


class SharedEncoder:
    """Kilitli ön işleme (256/224/ImageNet) + dondurulmuş ResNet50, src/data/features.py'den."""

    def __init__(self, device: torch.device):
        from src.data.features import build_transform, load_encoder

        self.device = device
        self.transform = build_transform(load_config())
        # Normalize öncesi kırpım: aynı Compose'un ilk iki adımı (Resize, CenterCrop).
        self.crop_transform = transforms.Compose(self.transform.transforms[:2])
        self.encoder = load_encoder(device)
        self.params = count_params(self.encoder)

    @torch.no_grad()
    def prepare(self, image: Image.Image) -> PreparedImage:
        from src.data.features import extract_image

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image (Gradio type='pil')")
        rgb = image.convert("RGB")
        crop = self.crop_transform(rgb)
        _sync(self.device)
        started = time.perf_counter()
        features = extract_image(rgb, self.encoder, self.transform, self.device)
        _sync(self.device)
        return PreparedImage(rgb, crop, features, 1000 * (time.perf_counter() - started))


_ENCODERS: dict[str, SharedEncoder] = {}
_ENCODER_LOCK = threading.Lock()


def get_shared_encoder(device: torch.device) -> SharedEncoder:
    """Cihaz başına tek ResNet50; tüm önbellek tabanlı adaptörler bunu kullanır."""
    with _ENCODER_LOCK:
        if str(device) not in _ENCODERS:
            _ENCODERS[str(device)] = SharedEncoder(device)
        return _ENCODERS[str(device)]


def check_decoding(method: str, beam_size: int) -> int:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method!r}")
    beam_size = int(beam_size)
    if not 1 <= beam_size <= MAX_BEAM:
        raise ValueError(f"beam_size must be in 1..{MAX_BEAM}")
    return beam_size


# --------------------------------------------------------------------------- kayıt noktası

ADAPTERS: dict[str, type["Captioner"]] = {}


def register_adapter(kind: str):
    """app/config.yaml'daki `kind` adını bir Captioner sınıfına bağlar."""

    def decorator(cls):
        ADAPTERS[kind] = cls
        return cls

    return decorator


class Captioner:
    """Temel sınıf. Alt sınıflar load() ve generate() yazar.

    load() başarısızsa NotConnectedError (açık mesajla) atmalı; kayıt bu mesajı kutuya yazar.
    load() başarılıysa self.params doldurulmalı.
    """

    needs_features = True  # True: PreparedImage.features (paylaşılan ResNet50) kullanır

    def __init__(self, key: str, label: str, device: torch.device, attention: bool = False, **options):
        self.key, self.label, self.device = key, label, device
        self.attention_capable = bool(attention)
        self.options = options
        self.params: int | None = None
        self.ready = False
        self.status = "yüklenmedi"

    def load(self) -> None:
        raise NotImplementedError

    def generate(self, image, method: str = "greedy", beam_size: int = 3) -> CaptionResult:
        raise NotImplementedError

    def _prepare(self, image) -> PreparedImage:
        if isinstance(image, PreparedImage):
            return image
        return get_shared_encoder(self.device).prepare(image)


@register_adapter("checkpoint")
class CheckpointCaptioner(Captioner):
    """src.train.MODELS'taki herhangi bir önbellek tabanlı model (Model 1, 2 ve kayda girince 3).

    options: checkpoint (depo köküne göre yol). Checkpoint state: model, config
    (config["model"]["type"] MODELS anahtarı), vocab_size, epoch. Model forward sözleşmesi:
    forward(features [B,2048,7,7] | [B,49,2048], tokens [B,T]) -> (logits [B,T,V], attention [B,T,49] | None).
    Çözümleme src.decode.generate_caption(..., return_attention=True) ile yapılır.
    """

    def load(self) -> None:
        path = repo_path(self.options["checkpoint"])
        if not path.is_file():
            raise NotConnectedError(f"checkpoint bulunamadı: {display_path(path)}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        config = state["config"]
        model_type = config["model"]["type"]
        from src.train import MODELS, build_model

        if model_type not in MODELS:
            raise NotConnectedError(
                f"henüz bağlanmadı: model türü '{model_type}' src/train.py MODELS kaydında yok"
            )
        vocab = get_vocab()
        if state.get("vocab_size", len(vocab)) != len(vocab):
            raise ValueError(f"checkpoint vocab_size {state['vocab_size']} != sözlük {len(vocab)}")
        model = build_model(config, vocab)
        model.load_state_dict(state["model"])
        self.model = model.to(self.device).eval()
        self.vocab = vocab
        self.id_to_word = dict(enumerate(vocab.itos))
        self.params = count_params(self.model)
        self.epoch = state.get("epoch")
        self.checkpoint = display_path(path)

    def postprocess_attention(self, weights: Tensor | None, words: list[str]) -> Tensor | None:
        """generate_caption çıktısını [len(words), 49]'a getirir; uymuyorsa None."""
        if weights is None:
            return None
        weights = weights.float()
        if tuple(weights.shape) != (len(words), 49):
            return None
        return weights

    @torch.no_grad()
    def generate(self, image, method: str = "greedy", beam_size: int = 3) -> CaptionResult:
        from src.decode import generate_caption

        beam_size = check_decoding(method, beam_size)
        prepared = self._prepare(image)
        features = prepared.features.to(self.device)
        _sync(self.device)
        started = time.perf_counter()
        caption, weights = generate_caption(
            self.model, features, self.id_to_word,
            start_id=self.vocab.start_idx, end_id=self.vocab.end_idx, pad_id=self.vocab.pad_idx,
            method=method, beam_size=beam_size, return_attention=True,
        )
        _sync(self.device)
        decoder_ms = 1000 * (time.perf_counter() - started)
        words = caption.split()
        return CaptionResult(
            caption=caption,
            attention_words=words,
            attention=self.postprocess_attention(weights, words),
            ms=prepared.encoder_ms + decoder_ms,
            params=self.params,
            detail=f"encoder {prepared.encoder_ms:.0f} ms + decoder {decoder_ms:.0f} ms",
        )


@register_adapter("transformer")
class TransformerCaptioner(CheckpointCaptioner):
    """Model 3 (Transformer) — Üye C'nin bağlantı noktası.

    Model 3 `src/train.py` MODELS'a (ör. "transformer") eklenip checkpoint'i app/config.yaml'daki
    yola konunca bu sınıf değişmeden çalışır; o zamana kadar kutuda "checkpoint bulunamadı" ya da
    "henüz bağlanmadı" yazar.

    Attention: forward ikinci çıktı olarak son katman cross-attention'ını döndürmeli:
      [B, T, 49] (kafalar ortalanmış) ya da [B, T, H, 49] (kafa başına) — ikincisinde burada
      kafalar üzerinden ortalanır. Attention vermiyorsa None döndürsün; demo haritayı gizler.
    Farklı bir çözümleme gerekiyorsa generate()'i ezin ama CaptionResult sözleşmesini koruyun.
    """

    def postprocess_attention(self, weights: Tensor | None, words: list[str]) -> Tensor | None:
        if weights is not None and weights.ndim > 2 and weights.shape[0] == len(words) and weights.shape[-1] == 49:
            weights = weights.float().reshape(len(words), -1, 49).mean(dim=1)
        return super().postprocess_attention(weights, words)


@register_adapter("blip")
class BlipCaptioner(Captioner):
    """Model 4 (BLIP) — Üye C dolduracak yer tutucu. Attention haritası yok.

    BLIP kendi işlemcisini kullanır: paylaşılan ResNet50 ve 224 kırpım kullanılmaz,
    generate() PreparedImage.original'ı (ön işlenmemiş RGB PIL) almalı.
    options: model_id (app/config.yaml, ör. Salesforce/blip-image-captioning-base).

    Doldurulacaklar (taslak):

        def load(self):
            from transformers import BlipForConditionalGeneration, BlipProcessor
            self.processor = BlipProcessor.from_pretrained(self.options["model_id"])
            self.model = BlipForConditionalGeneration.from_pretrained(self.options["model_id"])
            self.model.to(self.device).eval()
            self.params = count_params(self.model)

        @torch.no_grad()
        def generate(self, image, method="greedy", beam_size=3):
            beam_size = check_decoding(method, beam_size)
            rgb = image.original if isinstance(image, PreparedImage) else image.convert("RGB")
            inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
            # süreyi işlemci + generate için ölçün (_sync ile)
            out = self.model.generate(**inputs, num_beams=1 if method == "greedy" else beam_size,
                                      max_new_tokens=30)
            caption = self.processor.decode(out[0], skip_special_tokens=True).strip().lower()
            return CaptionResult(caption, caption.split(), None, ms, self.params, detail="...")
    """

    needs_features = False

    def load(self) -> None:
        raise NotConnectedError(
            "henüz bağlanmadı: BLIP adaptörü (app/captioners.py · BlipCaptioner) Üye C tarafından doldurulacak"
        )

    def generate(self, image, method: str = "greedy", beam_size: int = 3) -> CaptionResult:
        raise NotConnectedError(self.status)


# --------------------------------------------------------------------------- kayıt


class CaptionerRegistry:
    """app/config.yaml'daki modelleri sırayla kurar, yükler ve bir görüntüde hepsini çalıştırır."""

    def __init__(self, config_path: str | Path = CONFIG_PATH, device: str | None = None):
        with Path(config_path).open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.device = resolve_device(device or config.get("device", "auto"))
        self.captioners: list[Captioner] = []
        for entry in config["models"]:
            entry = dict(entry)
            kind = entry.pop("kind")
            if kind not in ADAPTERS:
                raise ValueError(f"Bilinmeyen adaptör türü {kind!r}; seçenekler: {', '.join(ADAPTERS)}")
            self.captioners.append(ADAPTERS[kind](device=self.device, **entry))
        self.encoder: SharedEncoder | None = None
        self.encoder_status = "yüklenmedi"
        self._lock = threading.Lock()  # tek GPU/CPU: istekler sırayla

    def __getitem__(self, key: str) -> Captioner:
        for captioner in self.captioners:
            if captioner.key == key:
                return captioner
        raise KeyError(key)

    def load_all(self) -> None:
        for captioner in self.captioners:
            try:
                captioner.load()
                captioner.ready = True
                captioner.status = "hazır"
            except NotConnectedError as error:
                captioner.ready, captioner.status = False, str(error)
            except Exception as error:  # bozuk checkpoint vb. uygulamayı düşürmesin
                captioner.ready, captioner.status = False, f"yüklenemedi: {type(error).__name__}: {error}"
        if any(c.ready and c.needs_features for c in self.captioners):
            try:
                self.encoder = get_shared_encoder(self.device)
                self.encoder.prepare(Image.new("RGB", (256, 256), (128, 128, 128)))  # ısınma: ilk süre şişmesin
                self.encoder_status = f"ResNet50 encoder hazır ({self.encoder.params:,} parametre, {self.device})"
            except Exception as error:
                self.encoder_status = f"ResNet50 yüklenemedi: {type(error).__name__}: {error}"
                for captioner in self.captioners:
                    if captioner.ready and captioner.needs_features:
                        captioner.ready, captioner.status = False, self.encoder_status
        else:
            self.encoder_status = "ResNet50 encoder gerekmedi (önbellek tabanlı model hazır değil)"

    def prepare(self, image: Image.Image) -> PreparedImage:
        if self.encoder is None:
            rgb = image.convert("RGB")
            crop = transforms.CenterCrop(224)(transforms.Resize(256)(rgb))
            return PreparedImage(rgb, crop, None, 0.0)
        return self.encoder.prepare(image)

    def run_all(self, image: Image.Image, method: str, beam_size: int):
        """-> (PreparedImage, {key: CaptionResult | str hata/durum mesajı})"""
        beam_size = check_decoding(method, beam_size)
        with self._lock:
            prepared = self.prepare(image)
            results: dict[str, CaptionResult | str] = {}
            for captioner in self.captioners:
                if not captioner.ready:
                    results[captioner.key] = captioner.status
                    continue
                try:
                    results[captioner.key] = captioner.generate(prepared, method, beam_size)
                except Exception as error:
                    results[captioner.key] = f"çalıştırılamadı: {type(error).__name__}: {error}"
        return prepared, results

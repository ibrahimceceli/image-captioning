"""ResNet50 layer4 öznitelikleri. Önbellek betiği ve demo aynı dönüşümü buradan alır."""

from pathlib import Path

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

from src.data.cache import FEATURE_SHAPE

# IMAGENET1K_V1 seçildi: klasik ResNet50 ağırlığı (top-1 %76.1). V2'nin eğitim
# tarifi farklı ön işleme (232 resize) varsayar; kilitli 256/224 hattıyla V1 uyumludur.
WEIGHTS = ResNet50_Weights.IMAGENET1K_V1
WEIGHTS_NAME = "ResNet50_Weights.IMAGENET1K_V1"


def build_transform(config: dict) -> transforms.Compose:
    """Kısa kenarı 256'ya getirir, merkezden 224 kırpar, ImageNet ile normalize eder."""
    return transforms.Compose(
        [
            transforms.Resize(int(config["resize_short_edge"])),
            transforms.CenterCrop(int(config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(config["imagenet_mean"], config["imagenet_std"]),
        ]
    )


def load_encoder(device: str | torch.device = "cpu") -> nn.Module:
    """avgpool ve fc atılmış, dondurulmuş ResNet50. Çıktı [B, 2048, 7, 7]."""
    backbone = resnet50(weights=WEIGHTS)
    encoder = nn.Sequential(*list(backbone.children())[:-2])
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder.to(device).eval()


def open_image(path: Path | str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


@torch.no_grad()
def extract_batch(
    images: Tensor,
    encoder: nn.Module,
    device: str | torch.device,
    amp: bool = False,
) -> Tensor:
    """Dönüştürülmüş [B, 3, 224, 224] tensörden float32 [B, 2048, 7, 7] üretir."""
    images = images.to(device)
    use_amp = amp and torch.device(device).type == "cuda"
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        features = encoder(images)
    if tuple(features.shape[1:]) != FEATURE_SHAPE:
        raise ValueError(f"Beklenmeyen öznitelik şekli: {tuple(features.shape)}")
    return features.float()


def extract_image(
    pil_image: Image.Image,
    encoder: nn.Module,
    transform: transforms.Compose,
    device: str | torch.device,
) -> Tensor:
    """Demo için tek görüntü. Çıktı float32 [1, 2048, 7, 7], encoder'ın cihazında."""
    image = transform(pil_image.convert("RGB")).unsqueeze(0)
    return extract_batch(image, encoder, device)

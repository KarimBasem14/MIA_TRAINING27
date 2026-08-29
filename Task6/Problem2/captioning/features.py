"""Pretrained CNN feature extraction and on-disk caching."""

from pathlib import Path
import hashlib

import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50


class ImageFeatureExtractor:
    def __init__(self, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        weights = ResNet50_Weights.DEFAULT
        backbone = resnet50(weights=weights)
        self.model = torch.nn.Sequential(*list(backbone.children())[:-2]).to(self.device).eval()
        self.transform = weights.transforms()

    @torch.no_grad()
    def extract(self, image_path: str | Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        features = self.model(tensor).flatten(2).transpose(1, 2)
        return features.squeeze(0).cpu()

    def extract_cached(self, image_path: str | Path, cache_dir: str | Path) -> torch.Tensor:
        path = Path(image_path)
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        cache_file = cache / f"{hashlib.sha1(str(path.resolve()).encode()).hexdigest()}.pt"
        if cache_file.exists():
            return torch.load(cache_file, map_location="cpu", weights_only=True)
        features = self.extract(path)
        torch.save(features, cache_file)
        return features

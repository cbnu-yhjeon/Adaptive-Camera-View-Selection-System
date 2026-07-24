"""백본 특징 사전 추출·캐시 (CLAUDE.md M3 / B.5).

MobileNetV3-small의 GAP 특징(D_f)을 keyframe·카메라별로 미리 뽑아
cache/{version}/feat_{camera}.npy (memmap)에 저장한다. 학습 시 무거운 백본을
재실행하지 않으므로 1 epoch가 매우 빨라진다(원본 이미지 학습 아님).

인덱스(index.json)에 sample_token 순서와 feat_dim을 기록해 재현·로드에 사용.
재실행 시 캐시가 있으면 스킵.
"""

from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

from src.utils.constants import CAMERAS

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _build_model(device: str):
    import timm
    import torch
    model = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=0)
    model.eval().to(device)
    # forward_features → (B, C, H, W) 후 GAP 하여 C=576 특징을 쓴다(conv_head 이전).
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 64, 64, device=device)
        feat_dim = int(model.forward_features(dummy).shape[1])
    return model, feat_dim


def _preprocess(path: str, img_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return arr.transpose(2, 0, 1)  # (3,H,W)


def cache_paths(cache_dir: str) -> dict:
    return {
        "index": os.path.join(cache_dir, "index.json"),
        **{cam: os.path.join(cache_dir, f"feat_{cam}.npy") for cam in CAMERAS},
    }


def is_cached(cache_dir: str) -> bool:
    paths = cache_paths(cache_dir)
    return all(os.path.isfile(p) for p in paths.values())


def build_feature_cache(wrapper, cache_dir: str, img_size: int = 224,
                        device: str = "cuda", batch_size: int = 64,
                        force: bool = False) -> dict:
    """특징 캐시 생성. 이미 있으면 스킵(force=True면 재생성)."""
    import torch

    if is_cached(cache_dir) and not force:
        with open(cache_paths(cache_dir)["index"], encoding="utf-8") as f:
            return json.load(f)

    os.makedirs(cache_dir, exist_ok=True)
    tokens = wrapper.all_sample_tokens()
    n = len(tokens)
    model, feat_dim = _build_model(device)
    print(f"[특징캐시] {n} keyframe × {len(CAMERAS)}채널, feat_dim={feat_dim}, "
          f"img_size={img_size}, device={device}")

    memmaps = {cam: np.lib.format.open_memmap(
        cache_paths(cache_dir)[cam], mode="w+", dtype=np.float32, shape=(n, feat_dim))
        for cam in CAMERAS}

    for cam in CAMERAS:
        buf, rows = [], []
        for i, tok in enumerate(tokens):
            path = wrapper.get_image_paths(tok)[cam]
            buf.append(_preprocess(path, img_size))
            rows.append(i)
            if len(buf) == batch_size or i == n - 1:
                x = torch.from_numpy(np.stack(buf)).to(device)
                with torch.no_grad():
                    feats = model.forward_features(x)          # (B,C,H,W)
                    feats = feats.mean(dim=(2, 3))              # GAP → (B,C)
                memmaps[cam][rows] = feats.cpu().numpy()
                buf, rows = [], []
        memmaps[cam].flush()
        print(f"  {cam} 완료")

    index = {"version": wrapper.version, "feat_dim": feat_dim,
             "img_size": img_size, "n": n, "tokens": tokens,
             "backbone": "mobilenetv3_small_100"}
    with open(cache_paths(cache_dir)["index"], "w", encoding="utf-8") as f:
        json.dump(index, f)
    print(f"[특징캐시] 저장 완료 → {cache_dir}")
    return index


class FeatureCache:
    """캐시된 특징 로더. get(token) → (6, feat_dim)."""

    def __init__(self, cache_dir: str):
        with open(cache_paths(cache_dir)["index"], encoding="utf-8") as f:
            self.index = json.load(f)
        self.feat_dim = self.index["feat_dim"]
        self.row = {t: i for i, t in enumerate(self.index["tokens"])}
        self.mm = {cam: np.load(cache_paths(cache_dir)[cam], mmap_mode="r")
                   for cam in CAMERAS}

    def get(self, sample_token: str) -> np.ndarray:
        r = self.row[sample_token]
        return np.stack([np.asarray(self.mm[cam][r]) for cam in CAMERAS])  # (6,D)

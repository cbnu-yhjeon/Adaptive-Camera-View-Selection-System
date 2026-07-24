"""라벨 시각화 (CLAUDE.md M2 — 사용자 검수용).

무작위 N개 keyframe에 대해 한 장의 PNG를 만든다:
  - 6채널 이미지 격자(surround view 배치)
  - 각 채널에 투영된 GT 3D 박스 (위험도=TTC 기반 색상)
  - 채널 점수 s(c) 막대 + 선택된 main_view 강조
  - 기동 상태 / min_TTC 텍스트

→ results/label_viz/*.png. 사용자가 이 그림들을 보고 라벨 타당성을 승인한다.

사용:
    python scripts/03_visualize_labels.py --n 20
    python scripts/03_visualize_labels.py --n 20 --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.nusc_wrapper import NuscWrapper  # noqa: E402
from src.labeling.label_function import LabelConfig  # noqa: E402
from src.utils.constants import CAMERAS  # noqa: E402
from src.utils.geometry import global_to_camera, project_to_image  # noqa: E402

# surround view 배치 (행,열) → 채널
GRID = [["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
        ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]]
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def load_configs():
    d = os.path.join(os.path.dirname(__file__), "..", "configs")
    with open(os.path.join(d, "base.yaml"), encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with open(os.path.join(d, "label_v1.yaml"), encoding="utf-8") as f:
        lab = yaml.safe_load(f)
    return base, lab


def _box_corners_global(center, size, rotation):
    """global 좌표 8코너 (3,8). devkit Box와 동일 규약."""
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion
    return Box(center.tolist(), size.tolist(), Quaternion(rotation)).corners()


def _draw_boxes_on_cam(ax, wrapper, sample_token, cam, boxes, ego):
    """한 카메라 축에 이미지 + 투영 박스."""
    paths = wrapper.get_image_paths(sample_token)
    img = np.array(Image.open(paths[cam]))
    ax.imshow(img)
    ax.set_title(cam, fontsize=8)
    ax.axis("off")
    H, W = img.shape[:2]
    cal = wrapper.get_camera_calib(sample_token, cam)
    from src.labeling.ttc import compute_ttc, horizontal_range
    ego_vel = wrapper.get_ego_velocity(sample_token)
    for b in boxes:
        corners_g = _box_corners_global(b["global_center"], b["size"], b["rotation"])
        cc = global_to_camera(corners_g.T, ego["translation"], ego["rotation"],
                              cal["translation"], cal["rotation"])  # (8,3)
        uv, depth = project_to_image(cc, cal["intrinsic"])
        if np.sum(depth > 0) < 8:      # 일부라도 뒤에 있으면 스킵(간단화)
            continue
        if np.all((uv[:, 0] < 0) | (uv[:, 0] > W) | (uv[:, 1] < 0) | (uv[:, 1] > H)):
            continue
        # 위험도 색상 (TTC 짧을수록 빨강)
        v_obj = b["global_velocity"] if np.all(np.isfinite(b["global_velocity"])) else np.zeros(2)
        ttc = compute_ttc(b["global_center"][:2], ego["translation"][:2], v_obj, ego_vel)
        color = (1.0, 0.1, 0.1) if ttc < 3 else (1.0, 0.6, 0.0) if ttc < 8 else (0.2, 0.7, 1.0)
        for i, j in EDGES:
            ax.plot([uv[i, 0], uv[j, 0]], [uv[i, 1], uv[j, 1]], color=color, lw=1.0)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)


def visualize_one(wrapper, row, cfg, out_path):
    st = row["sample_token"]
    ego = wrapper.get_ego_pose(st)
    boxes = wrapper.get_boxes(st, cfg.categories, cfg.max_range)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(3, 3, height_ratios=[3, 3, 1.4])
    for r in range(2):
        for c in range(3):
            ax = fig.add_subplot(gs[r, c])
            _draw_boxes_on_cam(ax, wrapper, st, GRID[r][c], boxes, ego)

    # 점수 막대
    axb = fig.add_subplot(gs[2, :])
    scores = [row[f"s_{i}"] for i in range(6)]
    main = int(row["main_view"])
    colors = ["crimson" if i == main else "steelblue" for i in range(6)]
    axb.bar(range(6), scores, color=colors)
    axb.set_xticks(range(6))
    axb.set_xticklabels([c.replace("CAM_", "") for c in CAMERAS], fontsize=8)
    axb.set_ylabel("s(c)")
    ttc_txt = f"{row['min_ttc']:.1f}s" if row["min_ttc"] >= 0 else "inf"
    axb.set_title(f"main_view = [{main}] {CAMERAS[main]}  |  "
                  f"maneuver = {row['maneuver']}  |  layout = {int(row['layout'])}  |  "
                  f"n_obj = {int(row['n_objects'])}  |  min_TTC = {ttc_txt}",
                  fontsize=10)
    fig.suptitle(f"{wrapper.scene_name(row['scene_token'])}  "
                 f"t={row['timestamp']:.1f}  (red=TTC<3s, orange<8s, blue=far)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    base, lab = load_configs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default=base["dataroot"])
    ap.add_argument("--version", default=base["version"])
    ap.add_argument("--labels", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=base.get("seed", 42))
    ap.add_argument("--outdir", default="results/label_viz")
    args = ap.parse_args()

    labels = args.labels or f"data/labels_{args.version}.parquet"
    df = pd.read_parquet(labels)
    os.makedirs(args.outdir, exist_ok=True)
    cfg = LabelConfig.from_dict(lab)
    w = NuscWrapper(args.dataroot, args.version,
                    require_image_files=base.get("require_image_files", True))

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(df), size=min(args.n, len(df)), replace=False)
    print(f"[시각화] {len(idx)}장 생성 → {args.outdir}/")
    for k, i in enumerate(sorted(idx)):
        row = df.iloc[i]
        out = os.path.join(args.outdir, f"viz_{k:02d}_{row['maneuver']}.png")
        visualize_one(w, row, cfg, out)
        print(f"  [{k+1}/{len(idx)}] {os.path.basename(out)}  "
              f"main={int(row['main_view'])} {row['maneuver']}")
    print(f"\n✅ 완료 — {args.outdir}/ 의 PNG를 검수하세요 (M2 게이트)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

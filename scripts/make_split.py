"""scene 단위 분할 생성 (CLAUDE.md 원칙 3) — **최초 1회만 실행하는 일회성 스크립트**.

무작위 70/15/15 (seed 42). 생성물 `splits/scene_split_v1.json`은 git에 커밋하고
**이후 절대 변경하지 않는다.** 이미 파일이 있으면 --force 없이는 덮어쓰지 않는다.

분할은 nuScenes 메타데이터의 scene 이름 전체(trainval 850개)를 대상으로 한다.
이미지 blob을 일부만 받은 경우에도 분할 자체는 바뀌지 않고, 학습 시
`splits.split_for_names()`가 실제 존재하는 scene만 걸러낸다.

사용법:
    python scripts/make_split.py                       # v1.0-trainval 기준 생성
    python scripts/make_split.py --version v1.0-mini --out splits/scene_split_mini.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.splits import make_scene_split, save_split  # noqa: E402

CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def main():
    with open(os.path.join(CFG_DIR, "base.yaml"), encoding="utf-8") as f:
        base = yaml.safe_load(f)

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default=base["dataroot"])
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--out", default=base.get("splits_file", "splits/scene_split_v1.json"))
    ap.add_argument("--seed", type=int, default=base.get("seed", 42))
    ap.add_argument("--force", action="store_true", help="기존 분할 파일 덮어쓰기")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        print(f"[중단] {args.out} 이 이미 있다. 원칙 3에 따라 분할은 변경하지 않는다.\n"
              f"       의도적으로 재생성하려면 --force.")
        return 1

    root = os.path.abspath(os.path.expanduser(args.dataroot))
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=root, verbose=False)

    names = [s["name"] for s in nusc.scene]
    loc_of = {s["name"]: nusc.get("log", s["log_token"])["location"] for s in nusc.scene}
    print(f"[분할] {args.version}: scene {len(names)}개")

    split = make_scene_split(names, seed=args.seed, ratios=(0.70, 0.15, 0.15))
    save_split(split, args.out, meta={
        "version_generated_on": args.version,
        "seed": args.seed,
        "n_scenes": len(names),
        "note": "무작위 scene 단위 분할. 생성 후 변경 금지(CLAUDE.md 원칙 3).",
    })

    # 구성 리포트 (논문 3.1 데이터 절에 그대로 인용 가능)
    print(f"\n{'분할':6s} {'scene':>6s} {'비율':>7s}   지역 분포")
    by_loc: dict[str, Counter] = defaultdict(Counter)
    for k in ("train", "val", "test"):
        for n in split[k]:
            by_loc[k][loc_of[n]] += 1
        pct = 100 * len(split[k]) / len(names)
        locs = ", ".join(f"{l.replace('singapore-', 'sg-')} {c}"
                         for l, c in sorted(by_loc[k].items()))
        print(f"{k:6s} {len(split[k]):6d} {pct:6.1f}%   {locs}")

    print(f"\n[저장] {args.out}")
    with open(args.out, encoding="utf-8") as f:
        print(json.dumps({k: v for k, v in json.load(f).items()
                          if k not in ("train", "val", "test")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

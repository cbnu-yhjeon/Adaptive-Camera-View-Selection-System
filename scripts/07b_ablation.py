"""Ablation 표 (CLAUDE.md B.9, M7) — 구성 요소를 하나씩 뺀 변형을 같은 입력·같은 표시량에서 비교.

변형마다 설정 파일의 exp_name으로 가장 최근 trainval 체크포인트를 찾는다. 시퀀스 길이가 다른
변형(A3=B4는 T=1, 보조 실험 T=10)은 각자의 길이로 데이터셋을 만들되, 평가 프레임 집합은
모두 같다(데이터셋은 길이와 무관하게 모든 keyframe을 한 번씩 돈다 — 실행 중 확인한다).

지표
  val AP      학습 중 모델 선택에 쓴 val macro-AP (metrics.json)
  test AP     평가 분할에서 다시 잰 macro-AP — 임계값과 무관한 순위 품질
  포착률      같은 표시량(평가 분할 라벨의 평균 보조 뷰 수)에서의 위험 사건 포착률 — τ 스윕 곡선을
              선형 보간한다(B.8). 고정 임계값 비교는 표시량 차이에 좌우되어 성립하지 않는다.
  정밀도·flicker  같은 표시량에서
입력 조건은 clean(gt)과 noisy(medium) — 후자가 표 1의 공정 비교 조건이다(B.8).
규칙(B2·B2c)은 참고 행으로 함께 싣는다.

출력: results/ablation.md · .json (test), 다른 분할은 results/ablation_{split}.*

사용
    python scripts/07b_ablation.py                 # test
    python scripts/07b_ablation.py --split val
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.baselines.rule_based import RuleBased  # noqa: E402
from src.eval.metrics import macro_ap  # noqa: E402
from src.models.proposed import ModelConfig, ViewSelector  # noqa: E402
from src.utils.config import load_cfgs  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ev", os.path.join(os.path.dirname(__file__), "06_evaluate.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

# (태그, 설정 파일, 설명) — 첫 행이 기준(제안 모델)
VARIANTS = [
    ("제안", "configs/train_proposed.yaml", "전체 (T=4)"),
    ("A1", "configs/ablation_a1_no_can.yaml", "− CAN 토큰 (지시등 포함)"),
    ("A2", "configs/ablation_a2_no_risk.yaml", "− 위험도 토큰"),
    ("A3", "configs/train_b4_single_frame.yaml", "− GRU, T=1 (= B4)"),
    ("A4", "configs/ablation_a4_no_xview.yaml", "− cross-view attention"),
    ("A5", "configs/ablation_a5_no_temporal.yaml", "− temporal consistency (β=0)"),
    ("B3", "configs/train_b3_img_only.yaml", "영상만 (− CAN − 위험도)"),
    ("T10", "configs/exp_t10.yaml", "보조 실험: 이력 5초 (T=10)"),
]
CONDS = [("gt", "clean"), ("noisy", "medium")]
TAU = [round(x, 3) for x in np.linspace(0.02, 0.98, 49)]


def cond_name(rs, noise):
    return f"{rs}-{noise}" if rs == "noisy" else rs


def interp(pts, key, load):
    xs = np.array([p["mean_aux"] for p in pts]); ys = np.array([p[key] for p in pts])
    o = np.argsort(xs); xs, ys = xs[o], ys[o]
    if load < xs[0] or load > xs[-1]:
        return None
    return float(np.interp(load, xs, ys))


def at_load(pts, load):
    return {k: interp(pts, k, load) for k in ("recall", "precision", "flicker")}


def val_ap(ckpt):
    with open(os.path.join(os.path.dirname(ckpt), "metrics.json"), encoding="utf-8") as f:
        m = json.load(f)
    return m.get("best_val_ap", m.get("best_f1")), m.get("best_epoch")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--plot_only", action="store_true", help="저장된 json으로 표만 다시 만든다")
    ap.add_argument("--version", default=None, help="mini 스모크: v1.0-mini")
    ap.add_argument("--splits", default=None, help="mini 스모크: splits/scene_split_mini.json")
    ap.add_argument("--out", default=None, help="출력 경로 앞부분 (기본 results/ablation[_split])")
    args = ap.parse_args()
    stem = args.out or ("results/ablation" + ("" if args.split == "test" else f"_{args.split}"))
    if args.plot_only:
        with open(stem + ".json", encoding="utf-8") as f:
            write_md(json.load(f), stem)
        return 0

    base, lab, tc0 = load_cfgs(VARIANTS[0][1])
    version = args.version or base["version"]
    env = ev.Env(base, lab, tc0, version, args.splits)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds0 = env.dataset(args.split, "gt", "clean")
    ref_tokens = [ds0.scene_tokens[sc][pos] for sc, pos in ds0.flat]
    load = float(env.df.set_index("sample_token").loc[ref_tokens, "n_aux"].mean())
    print(f"[ablation] {version} · {args.split} · 기준 표시량 {load:.3f}장")

    out = dict(split=args.split, load=load, conds=[cond_name(*c) for c in CONDS], rows=[])
    for tag, cfg_path, desc in VARIANTS:
        _, _, tc = load_cfgs(cfg_path)
        ckpt = ev.latest_ckpt(tc.get("exp_name", "proposed"), version)
        if ckpt is None:
            print(f"  {tag}: {version} 체크포인트 없음 — 건너뜀")
            continue
        seq_len = tc["train"].get("seq_len", 4)
        st = torch.load(ckpt, map_location=device, weights_only=False)
        model = ViewSelector(ModelConfig(**st["mcfg"])).to(device)
        model.load_state_dict(st["model"])
        model.eval()
        vap, vep = val_ap(ckpt)
        row = dict(tag=tag, desc=desc, ckpt=ckpt, seq_len=seq_len, val_ap=vap, best_epoch=vep, res={})
        for rs, noise in CONDS:
            ds = env.dataset(args.split, rs, noise, seq_len=seq_len)
            P, att, meta = ev.run_model(model, ds, device)
            if meta["token"] != ref_tokens:
                raise RuntimeError(f"{tag}: 평가 프레임 집합/순서가 기준과 다르다 (T={seq_len})")
            pts = []
            for tau in TAU:
                sel = ev.select_from_probs(P, meta["primary"].astype(int), tau, env.cfg.aux_max)
                r = ev.metrics(env, sel, att, meta)
                pts.append(dict(param=tau, **{k: r[k] for k in ("mean_aux", "recall", "precision", "flicker")}))
            row["res"][cond_name(rs, noise)] = dict(
                ap=macro_ap(P, meta["aux"] > 0.5, meta["mask"] > 0.5), at_load=at_load(pts, load), curve=pts)
        out["rows"].append(row)
        r = row["res"]
        print(f"  {tag:4s} valAP {vap:.3f}  " + "  ".join(
            f"{c}: AP {r[c]['ap']:.3f} 포착 {r[c]['at_load']['recall']:.1f}%" for c in r))

    rules = {"B2": RuleBased(copy.deepcopy(env.cfg), "oracle"),
             "B2c": RuleBased(copy.deepcopy(env.cfg), "causal", wrapper=env.w)}
    out["rules"] = []
    for name, rule in rules.items():
        row = dict(tag=name, res={})
        for rs, noise in CONDS:
            ds = env.dataset(args.split, rs, noise)
            pts = []
            for th in ev.CURVE_TH:
                rule.cfg.aux_threshold = th
                sel, att, meta = ev.run_rule(rule, ds)
                r = ev.metrics(env, sel, att, meta)
                pts.append(dict(param=th, **{k: r[k] for k in ("mean_aux", "recall", "precision", "flicker")}))
            row["res"][cond_name(rs, noise)] = dict(at_load=at_load(pts, load), curve=pts)
        out["rules"].append(row)
        print(f"  {name:4s} " + "  ".join(f"{c}: 포착 {v['at_load']['recall']:.1f}%" for c, v in row["res"].items()))

    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    write_md(out, stem)
    return 0


def write_md(out, stem):
    load = out["load"]
    c_clean, c_noisy = out["conds"]
    f1 = lambda v: "—" if v is None else f"{v:.1f}%"
    f3 = lambda v: "—" if v is None else f"{v:.3f}"
    ref = out["rows"][0]["res"][c_noisy]["at_load"]["recall"]
    lines = [f"# Ablation — {out['split']} 분할, 같은 표시량({load:.2f}장)에서 비교\n",
             f"포착률·정밀도·flicker는 τ 스윕 곡선을 표시량 {load:.2f}장에서 보간한 값(B.8). "
             f"입력 조건: clean = {c_clean}, noisy = {c_noisy}(표 1의 공정 비교 조건). "
             "Δ는 noisy 포착률의 제안 모델 대비 차이.\n",
             f"| 태그 | 구성 | val AP | {out['split']} AP (noisy) | 포착률 clean | 포착률 noisy | Δ | 정밀도 noisy | flicker noisy |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in out["rows"]:
        n = r["res"][c_noisy]; c = r["res"][c_clean]
        rec = n["at_load"]["recall"]
        d = "—" if (rec is None or ref is None or r is out["rows"][0]) else f"{rec - ref:+.1f}%p"
        lines.append(f"| {r['tag']} | {r['desc']} | {f3(r['val_ap'])} | {f3(n['ap'])} | "
                     f"{f1(c['at_load']['recall'])} | {f1(rec)} | {d} | "
                     f"{f1(n['at_load']['precision'])} | {f3(n['at_load']['flicker'])} |")
    lines += ["", "**참고 — 규칙** (학습 없음, AP 해당 없음)", "",
              "| 방법 | 포착률 clean | 포착률 noisy | 정밀도 noisy | flicker noisy |", "|---|---|---|---|---|"]
    for r in out.get("rules", []):
        n = r["res"][c_noisy]["at_load"]; c = r["res"][c_clean]["at_load"]
        lines.append(f"| {r['tag']} | {f1(c['recall'])} | {f1(n['recall'])} | {f1(n['precision'])} | {f3(n['flicker'])} |")
    lines += ["", "체크포인트:", ""] + [f"- {r['tag']}: `{r['ckpt']}` (best ep{r['best_epoch']}, T={r['seq_len']})"
                                      for r in out["rows"]]
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n✅ {stem}.json · .md")


if __name__ == "__main__":
    raise SystemExit(main())

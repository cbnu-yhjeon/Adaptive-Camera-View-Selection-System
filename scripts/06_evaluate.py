"""공통 평가 엔트리 (CLAUDE.md M5·M6) — 모든 방법을 같은 입력·같은 지표로 잰다.

방법
  b1     고정 — 보조 뷰 없음
  b2     규칙 — 라벨 뷰 정책 + 라벨 기동(오라클, 미래 궤적)
  b2c    규칙 — 라벨 뷰 정책 + 과거 궤적만으로 추정한 기동
  model  학습 모델 (--ckpt)

모든 방법은 같은 Dataset 아이템을 본다. 평가 노이즈 시드는 (seed, index)뿐이라
방법이 달라도 같은 프레임에는 같은 손상 입력이 들어간다(B.6) — 열화 곡선 비교의 전제다.

지표 (B.8)
  pseudo_f1     의사 라벨 보조 뷰 집합과의 채널별 F1 — 보조 지표(순환 평가 주의, 원칙 1)
  mean_aux      평균 보조 뷰 수 — 인지 부하의 대리 지표
  recall        '실제 위험' 채널이 선택 뷰(primary ∪ aux)에 든 비율 — GT 기하로 직접 계산
  precision     보조 뷰로 켠 채널 중 실제 위험이 있던 비율 — 반드시 recall과 병기
  flicker       보조 뷰 집합이 바뀐 뒤 3 keyframe 안에 되돌아간 비율 (scene 시간순)
  top1          단일 선택(attention_view)과의 일치 — 베이스라인 비교용

모델의 임계값은 **val에서** 라벨과 같은 표시량이 되도록 정하고 평가 분할에 그대로 적용한다.
평가 분할에서 고르면 그 분할을 본 셈이 된다.

사용
    python scripts/06_evaluate.py --method b2 --split test --risk_source noisy --noise medium
    python scripts/06_evaluate.py --method model --ckpt results/proposed/<run>/best.ckpt --split test
    python scripts/06_evaluate.py --all --split test        # b1·b2·b2c·model 한 번에
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_cfgs  # noqa: E402

from src.baselines.rule_based import FixedFront, RuleBased, as_arrays  # noqa: E402
from src.data import contrib_cache  # noqa: E402
from src.data.dataset import ViewSelectionDataset, collate  # noqa: E402
from src.data.feature_cache import FeatureCache  # noqa: E402
from src.data.nusc_wrapper import NuscWrapper  # noqa: E402
from src.data.splits import load_split, split_for_names  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    flicker_rate, mean_aux, multilabel_f1, threshold_for_load, top1,
)
from src.labeling.label_function import LabelConfig, is_hazard  # noqa: E402
from src.models.proposed import ModelConfig, ViewSelector  # noqa: E402
from src.utils.constants import NUM_CAMERAS  # noqa: E402

TENSORS = ("img_feat", "can", "risk", "pad_mask", "soft_label", "aux_target", "aux_mask",
           "attention_view", "primary_view", "layout")


class Env:
    """평가에 공통으로 쓰는 데이터 — 방법마다 다시 만들지 않는다."""

    def __init__(self, base, lab, tc, version, splits_file=None):
        self.base, self.lab, self.tc = base, lab, tc
        self.cfg = LabelConfig.from_dict(lab)
        self.w = NuscWrapper(base["dataroot"], version,
                             require_image_files=base.get("require_image_files", True))
        cache_dir = os.path.join(base.get("cache_dir", "cache"), version)
        self.fc = FeatureCache(cache_dir)
        self.df = pd.read_parquet(f"data/labels_{version}.parquet")
        self.split = split_for_names(load_split(splits_file or base["splits_file"]),
                                     {self.w.scene_name(t) for t in self.w.scene_tokens()})
        self.cc = contrib_cache.load_or_build(self.w, self.w.all_sample_tokens(), self.cfg, cache_dir)
        self._haz: dict[str, set] = {}

    def dataset(self, split, risk_source, noise):
        return ViewSelectionDataset(
            self.w, self.fc, self.df, self.cfg, self.split[split],
            seq_len=self.tc["train"].get("seq_len", 4),
            risk_source=risk_source, noise_preset=noise, train_risk_noise=False,
            seed=self.base.get("seed", 42), contrib_cache=self.cc)

    def hazard(self, token) -> set:
        """'실제 위험' 채널 — GT 기하로 판정. 라벨·평가·골드셋이 같은 is_hazard를 쓴다."""
        if token not in self._haz:
            ch = set()
            for c in self.cc.get(token):
                if is_hazard(c, self.cfg):
                    ch.update(c["vis"])
            self._haz[token] = ch
        return self._haz[token]


@torch.no_grad()
def run_model(model, ds, device, batch_size=128):
    """확률(6) · 단일 선택 argmax · 메타를 모은다."""
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    probs, att, meta = [], [], []
    for batch in dl:
        for k in TENSORS:
            batch[k] = batch[k].to(device)
        logits = model(batch)["attn_logits"]
        probs.append(torch.sigmoid(logits).cpu().numpy())
        masked = logits.masked_fill(
            torch.nn.functional.one_hot(batch["primary_view"], NUM_CAMERAS).bool(), float("-inf"))
        att.append(masked.argmax(1).cpu().numpy())
        meta.append(_meta(batch))
    return np.concatenate(probs), np.concatenate(att), _cat_meta(meta)


def run_rule(method, ds):
    """규칙은 이미지 특징이 필요 없으므로 DataLoader 없이 scene 시간순으로 돈다."""
    tokens = [ds.scene_tokens[sc][pos] for sc, pos in ds.flat]     # 데이터셋 순서
    sel, prim, att = as_arrays(method.run(ds), tokens)
    lab = ds.labels.loc[tokens]
    aux = np.stack([lab[f"aux_{c}"].to_numpy() for c in range(NUM_CAMERAS)], 1).astype(float)
    mask = np.ones_like(aux)
    mask[np.arange(len(tokens)), lab["primary_view"].to_numpy().astype(int)] = 0.0
    meta = dict(token=tokens, scene=[sc for sc, _ in ds.flat], aux=aux, mask=mask,
                primary=prim, attention=lab["attention_view"].to_numpy().astype(int))
    return sel, (att if (att >= 0).all() else np.full(len(tokens), -1)), meta


def _meta(batch):
    t = lambda k: batch[k].cpu().numpy() if torch.is_tensor(batch[k]) else np.asarray(batch[k])
    return dict(token=list(batch["sample_token"]), scene=list(batch["scene_token"]),
                aux=t("aux_target"), mask=t("aux_mask"), primary=t("primary_view"),
                attention=t("attention_view"))


def _cat_meta(ms):
    out = {}
    for k in ms[0]:
        out[k] = sum((m[k] for m in ms), []) if isinstance(ms[0][k], list) else np.concatenate([m[k] for m in ms])
    return out


def select_from_probs(P, primary, th, aux_max):
    keep = P >= th
    keep[np.arange(len(primary)), primary] = False
    order = np.argsort(-np.where(keep, P, -1.0), axis=1)
    np.put_along_axis(keep, order[:, aux_max:], False, axis=1)
    return keep


def metrics(env, sel, att, meta) -> dict:
    sel = np.asarray(sel, dtype=bool)
    n_h = caught = on = hit = 0
    for i, tok in enumerate(meta["token"]):
        haz = env.hazard(tok)
        chosen = set(np.flatnonzero(sel[i]).tolist()) | {int(meta["primary"][i])}
        n_h += len(haz)
        caught += len(haz & chosen)
        aux_ch = np.flatnonzero(sel[i])
        on += len(aux_ch)
        hit += sum(1 for c in aux_ch if c in haz)
    seqs: dict[str, list] = {}
    for i, sc in enumerate(meta["scene"]):                 # 데이터셋 순서 = scene 시간순
        seqs.setdefault(sc, []).append(tuple(sel[i].tolist()))
    out = dict(
        n=int(len(sel)),
        pseudo_f1=round(multilabel_f1(sel, meta["aux"] > 0.5, meta["mask"] > 0.5), 4),
        mean_aux=round(mean_aux(sel), 4),
        recall=round(100 * caught / max(n_h, 1), 2),
        precision=round(100 * hit / max(on, 1), 2),
        flicker=round(flicker_rate(seqs, window=3), 4),
        n_hazard=int(n_h))
    if (np.asarray(att) >= 0).all():
        out["top1"] = round(top1(np.asarray(att), meta["attention"]), 4)
    return out


def evaluate(env, method, split, risk_source, noise, ckpt=None, device="cpu"):
    aux_max = env.cfg.aux_max
    if method == "model":
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model = ViewSelector(ModelConfig(**state["mcfg"])).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        # 임계값은 val에서 라벨 표시량에 맞춘다 — 평가 분할을 보지 않기 위해.
        Pv, _, mv = run_model(model, env.dataset("val", risk_source, noise), device)
        th = threshold_for_load(Pv, float((mv["aux"] > 0.5).sum(1).mean()),
                                mv["mask"] > 0.5, aux_max)
        P, att, meta = run_model(model, env.dataset(split, risk_source, noise), device)
        sel = select_from_probs(P, meta["primary"].astype(int), th, aux_max)
        res = metrics(env, sel, att, meta)
        res["threshold"] = round(float(th), 3)
        return res
    if method == "b1":
        rule = FixedFront()
    elif method == "b2":
        rule = RuleBased(env.cfg, "oracle")
    elif method == "b2c":
        rule = RuleBased(env.cfg, "causal", wrapper=env.w)
    else:
        raise ValueError(method)
    sel, att, meta = run_rule(rule, env.dataset(split, risk_source, noise))
    return metrics(env, sel, att, meta)


CURVE_TH = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.55]
CURVE_TAU = [round(x, 2) for x in np.linspace(0.05, 0.95, 19)]


def curve(env, method, split, risk_source, noise, ckpt=None, device="cpu"):
    """운영 곡선 — 규칙은 보조 뷰 임계값 θ, 모델은 확률 임계값 τ를 쓴다.

    포착률·정밀도는 표시량에 좌우되므로(B.8) 방법 간 비교는 같은 표시량에서만 성립한다.
    고정 운영점 하나로 비교하면 표시량을 더 많이 쓴 쪽이 포착률에서 이긴다.
    """
    ds = env.dataset(split, risk_source, noise)
    pts = []
    if method == "model":
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model = ViewSelector(ModelConfig(**state["mcfg"])).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        P, att, meta = run_model(model, ds, device)
        for tau in CURVE_TAU:
            sel = select_from_probs(P, meta["primary"].astype(int), tau, env.cfg.aux_max)
            r = metrics(env, sel, att, meta)
            pts.append(dict(param=tau, **{k: r[k] for k in ("mean_aux", "recall", "precision", "flicker")}))
        return pts
    import copy
    cfg = copy.deepcopy(env.cfg)
    rule = RuleBased(cfg, "oracle" if method == "b2" else "causal", wrapper=env.w)
    for th in CURVE_TH:
        cfg.aux_threshold = th
        sel, att, meta = run_rule(rule, ds)
        r = metrics(env, sel, att, meta)
        pts.append(dict(param=th, **{k: r[k] for k in ("mean_aux", "recall", "precision", "flicker")}))
    return pts


def recall_at_load(pts, load):
    """곡선에서 주어진 평균 표시량의 포착률(선형 보간). 범위 밖이면 None."""
    xs = np.array([p["mean_aux"] for p in pts]); ys = np.array([p["recall"] for p in pts])
    o = np.argsort(xs); xs, ys = xs[o], ys[o]
    if load < xs[0] or load > xs[-1]:
        return None
    return float(np.interp(load, xs, ys))


def latest_ckpt(exp="proposed", version=None):
    """해당 데이터 버전으로 학습한 가장 최근 체크포인트.

    버전으로 거르지 않으면 mini 스모크 실행이 trainval 모델보다 최근이라 그쪽을 집는다.
    """
    for ck in sorted(glob.glob(f"results/{exp}/20*/best.ckpt"), reverse=True):
        cfg_path = os.path.join(os.path.dirname(ck), "config.yaml")
        try:
            with open(cfg_path, encoding="utf-8") as f:
                ran = yaml.safe_load(f)["base"]["version"]
        except (OSError, KeyError, TypeError):
            continue
        if version is None or ran == version:
            return ck
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["b1", "b2", "b2c", "model"])
    ap.add_argument("--all", action="store_true", help="b1·b2·b2c·model 모두")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--risk_source", default="noisy", choices=["gt", "noisy", "none"])
    ap.add_argument("--noise", default="medium", choices=["clean", "light", "medium", "heavy"])
    ap.add_argument("--config", default="configs/train_proposed.yaml")
    ap.add_argument("--version", default=None)
    ap.add_argument("--splits", default=None, help="mini 스모크: splits/scene_split_mini.json")
    ap.add_argument("--out_dir", default="results/eval")
    ap.add_argument("--curve", action="store_true", help="운영 곡선(임계값 스윕)으로 비교")
    args = ap.parse_args()

    base, lab, tc = load_cfgs(args.config)
    version = args.version or base["version"]
    env = Env(base, lab, tc, version, args.splits)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    methods = ["b1", "b2", "b2c", "model"] if args.all else [args.method]
    ckpt = args.ckpt or latest_ckpt(tc.get("exp_name", "proposed"), version)

    os.makedirs(args.out_dir, exist_ok=True)
    cond = f"{args.risk_source}-{args.noise}" if args.risk_source == "noisy" else args.risk_source

    if args.curve:
        curves = {}
        for m in methods:
            if m == "b1":
                continue                                   # 표시 0장 한 점뿐
            if m == "model" and not ckpt:
                continue
            curves[m] = curve(env, m, args.split, args.risk_source, args.noise, ckpt, device)
        with open(os.path.join(args.out_dir, f"curve_{args.split}_{cond}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(dict(split=args.split, condition=cond, ckpt=ckpt, curves=curves),
                      f, ensure_ascii=False, indent=2)
        print(f"\n[운영 곡선] {version} · {args.split} · 입력 {cond}")
        for m, pts in curves.items():
            print(f"\n  {m}")
            for p in pts:
                print(f"    {p['param']:5.2f}  표시 {p['mean_aux']:5.2f}장  포착 {p['recall']:5.1f}%"
                      f"  정밀 {p['precision']:5.1f}%  flicker {p['flicker']:.3f}")
        loads = [0.3, 0.46, 0.6, 0.8]
        print(f"\n  같은 표시량에서의 포착률 (선형 보간)")
        print("    " + "".join(f"{l:>9.2f}장" for l in loads))
        for m, pts in curves.items():
            vals = [recall_at_load(pts, l) for l in loads]
            print(f"  {m:6s}" + "".join(f"{v:9.1f}% " if v is not None else "       — " for v in vals))
        return 0
    rows = []
    for m in methods:
        if m == "model" and not ckpt:
            print("  model: 체크포인트 없음 — 건너뜀")
            continue
        res = evaluate(env, m, args.split, args.risk_source, args.noise, ckpt, device)
        res.update(method=m, split=args.split, condition=cond,
                   ckpt=(ckpt if m == "model" else None))
        rows.append(res)
        with open(os.path.join(args.out_dir, f"{m}_{args.split}_{cond}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"\n[평가] {version} · {args.split} · 입력 {cond}\n")
    print(f"{'방법':6s} {'평균표시':>8s} {'포착률':>8s} {'정밀도':>8s} {'flicker':>8s} "
          f"{'pseudoF1':>9s} {'top1':>6s}")
    for r in rows:
        print(f"{r['method']:6s} {r['mean_aux']:7.2f}장 {r['recall']:7.1f}% {r['precision']:7.1f}% "
              f"{r['flicker']:8.3f} {r['pseudo_f1']:9.3f} {r.get('top1', float('nan')):6.3f}"
              + (f"   (τ={r['threshold']})" if "threshold" in r else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

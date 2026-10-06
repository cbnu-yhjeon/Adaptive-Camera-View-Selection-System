"""공통 평가 엔트리 (CLAUDE.md M5·M6) — 모든 방법을 같은 입력·같은 지표로 잰다.

방법
  b1     고정 — 보조 뷰 없음
  b2     규칙 — 라벨 뷰 정책 + 라벨 기동(오라클, 미래 궤적)
  b2c    규칙 — 라벨 뷰 정책 + 과거 궤적만으로 추정한 기동
  b3     영상 단일 — 학습 모델에서 CAN·위험도 토큰 제거 (configs/train_b3_img_only.yaml)
  b4     단일 프레임 — GRU 제거, T=1 (configs/train_b4_single_frame.yaml)
  model  제안 모델 (--ckpt, 없으면 --config의 최신 trainval 실행)

모든 방법은 같은 Dataset 아이템을 본다. 평가 노이즈 시드는 (seed, index)뿐이라
방법이 달라도 같은 프레임에는 같은 손상 입력이 들어간다(B.6) — 열화 곡선 비교의 전제다.

지표 (B.8)
  pseudo_f1     의사 라벨 보조 뷰 집합과의 채널별 F1 — 보조 지표(순환 평가 주의, 원칙 1)
  mean_aux      평균 보조 뷰 수 — 인지 부하의 대리 지표
  recall        '실제 위험' 채널이 선택 뷰(primary ∪ aux)에 든 비율 — GT 기하로 직접 계산
  precision     보조 뷰로 켠 채널 중 실제 위험이 있던 비율 — 반드시 recall과 병기
  flicker       보조 뷰 집합이 바뀐 뒤 3 keyframe 안에 되돌아간 비율 (scene 시간순)
  top1          단일 선택(attention_view)과의 일치 — 베이스라인 비교용
  gold          골드셋(--gold)이 있으면 Top-1·Top-2·Macro-F1·정숙률 — 역확률 가중(주)·비가중·층별.
                사람이 고른 정답이라 순환이 없다 — 표 1의 주 지표(원칙 1)

모델의 임계값은 **val에서** 라벨과 같은 표시량이 되도록 정하고 평가 분할에 그대로 적용한다.
평가 분할에서 고르면 그 분할을 본 셈이 된다.

사용
    python scripts/06_evaluate.py --method b2 --split test --risk_source noisy --noise medium
    python scripts/06_evaluate.py --method model --ckpt results/proposed/<run>/best.ckpt --split test
    python scripts/06_evaluate.py --all --split test        # 6개 방법 + results/table1.md
    python scripts/06_evaluate.py --all --split test --risk_source gt   # 부록: results/table1_gt.md
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

from src.baselines.rule_based import FixedFront, RuleBased, as_arrays, score_array  # noqa: E402
from src.data import contrib_cache  # noqa: E402
from src.data.dataset import ViewSelectionDataset, collate  # noqa: E402
from src.data.feature_cache import FeatureCache  # noqa: E402
from src.data.nusc_wrapper import NuscWrapper  # noqa: E402
from src.data.splits import load_split, split_for_names  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    flicker_rate, gold_scores, mean_aux, multilabel_f1, threshold_for_load, top1,
)
from src.labeling.label_function import LabelConfig, is_hazard  # noqa: E402
from src.models.proposed import ModelConfig, ViewSelector  # noqa: E402
from src.utils.constants import NUM_CAMERAS  # noqa: E402

TENSORS = ("img_feat", "can", "risk", "pad_mask", "soft_label", "aux_target", "aux_mask",
           "attention_view", "primary_view", "layout")

# 학습 모델 방법 → 설정 파일 ('model'은 --config). 시퀀스 길이·exp_name을 각 설정에서 읽는다.
MODEL_CONFIGS = {"b3": "configs/train_b3_img_only.yaml", "b4": "configs/train_b4_single_frame.yaml"}
METHOD_KO = {"b1": "B1 고정", "b2": "B2 규칙(오라클 기동)", "b2c": "B2c 규칙(과거만)",
             "b3": "B3 영상 단일", "b4": "B4 단일 프레임", "model": "제안",
             "label": "의사 라벨 (참고)"}


def load_gold(path, cand_path="data/goldset/candidates_v1.json"):
    """골드셋 → {token: {set, weight, stratum, hmm}}. 없으면 None.

    검수 화면이 내보내는 {meta, gold, item_meta} 형식과 명세 형식 {token: [채널]}을 모두 받는다.
    가중치·층은 item_meta에 없으면 후보 목록에서 가져온다.
    """
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    gold, im = (d["gold"], d.get("item_meta", {})) if "gold" in d else (d, {})
    cand = {}
    if os.path.exists(cand_path):
        with open(cand_path, encoding="utf-8") as f:
            cand = {it["token"]: it for it in json.load(f)["items"]}
    out = {}
    for t, chs in gold.items():
        m = im.get(t) or cand.get(t, {})
        out[t] = dict(set=set(int(c) for c in chs), weight=float(m.get("weight", 1.0)),
                      stratum=m.get("stratum", "?"), hmm=bool(m.get("hmm", False)))
    return out


class Env:
    """평가에 공통으로 쓰는 데이터 — 방법마다 다시 만들지 않는다."""

    def __init__(self, base, lab, tc, version, splits_file=None, gold_path=None):
        self.base, self.lab, self.tc = base, lab, tc
        self.gold = load_gold(gold_path)
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

    def dataset(self, split, risk_source, noise, seq_len=None):
        """seq_len을 주면 설정 파일 대신 그 길이로 — 시퀀스 길이가 다른 변형(B4 T=1 등)을 함께 잴 때."""
        return ViewSelectionDataset(
            self.w, self.fc, self.df, self.cfg, self.split[split],
            seq_len=seq_len or self.tc["train"].get("seq_len", 4),
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
    P = np.concatenate(probs)
    meta = _cat_meta(meta)
    meta["rank"] = P                                       # 골드셋 Top-k 순위
    return P, np.concatenate(att), meta


def run_rule(method, ds):
    """규칙은 이미지 특징이 필요 없으므로 DataLoader 없이 scene 시간순으로 돈다."""
    tokens = [ds.scene_tokens[sc][pos] for sc, pos in ds.flat]     # 데이터셋 순서
    pred = method.run(ds)
    sel, prim, att = as_arrays(pred, tokens)
    lab = ds.labels.loc[tokens]
    aux = np.stack([lab[f"aux_{c}"].to_numpy() for c in range(NUM_CAMERAS)], 1).astype(float)
    mask = np.ones_like(aux)
    mask[np.arange(len(tokens)), lab["primary_view"].to_numpy().astype(int)] = 0.0
    meta = dict(token=tokens, scene=[sc for sc, _ in ds.flat], aux=aux, mask=mask,
                primary=prim, attention=lab["attention_view"].to_numpy().astype(int),
                rank=score_array(pred, tokens))
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
    if env.gold:
        idx = [i for i, t in enumerate(meta["token"]) if t in env.gold]
        if idx:
            out["gold"] = gold_block(env.gold, sel, meta, idx)
    return out


def gold_block(gold, sel, meta, idx):
    """골드셋 지표 — 역확률 가중(전체 분포 추정, 주 수치)·비가중·층별 (B.8 골드셋 절차)."""
    toks = [meta["token"][i] for i in idx]
    S = np.asarray(sel, dtype=bool)[idx]
    rank = None if meta.get("rank") is None else np.asarray(meta["rank"])[idx]
    G = [gold[t]["set"] for t in toks]
    prim = np.asarray(meta["primary"], dtype=int)[idx]
    w = np.array([gold[t]["weight"] for t in toks])
    res = dict(n=len(toks), n_hmm=int(sum(gold[t]["hmm"] for t in toks)),
               weighted=gold_scores(S, rank, G, prim, w), unweighted=gold_scores(S, rank, G, prim),
               by_stratum={})
    if rank is not None:                                    # 짝비교(같은 장면에서 누가 맞혔나)용
        r = rank.astype(float).copy()
        r[np.arange(len(toks)), prim] = -np.inf
        top = r.argmax(1)
        res["top1_hit"] = {t: bool(top[i] in G[i]) for i, t in enumerate(toks) if G[i] - {prim[i]}}
    strata = np.array([gold[t]["stratum"] for t in toks])
    for st in sorted(set(strata)):
        m = strata == st
        res["by_stratum"][st] = gold_scores(S[m], None if rank is None else rank[m],
                                            [g for g, k in zip(G, m) if k], prim[m])
    return res


def run_label(env, split):
    """의사 라벨 자체를 한 '방법'으로 — 골드셋과의 일치도가 라벨 생성 방법의 검증이다(주장 1).

    라벨은 GT·미래 궤적으로 만든 값이라 입력 조건과 무관하다. 순위는 라벨 점수 s(c).
    """
    ds = env.dataset(split, "gt", "clean")
    tokens = [ds.scene_tokens[sc][pos] for sc, pos in ds.flat]
    L = env.df.set_index("sample_token").loc[tokens]
    aux = np.stack([L[f"aux_{c}"].to_numpy() for c in range(NUM_CAMERAS)], 1).astype(float)
    prim = L["primary_view"].to_numpy().astype(int)
    mask = np.ones_like(aux)
    mask[np.arange(len(tokens)), prim] = 0.0
    meta = dict(token=tokens, scene=[sc for sc, _ in ds.flat], aux=aux, mask=mask, primary=prim,
                attention=L["attention_view"].to_numpy().astype(int),
                rank=np.stack([L[f"s_{c}"].to_numpy() for c in range(NUM_CAMERAS)], 1))
    return aux > 0.5, meta["attention"], meta


def evaluate(env, method, split, risk_source, noise, ckpt=None, device="cpu", seq_len=None):
    aux_max = env.cfg.aux_max
    if method == "label":
        sel, att, meta = run_label(env, split)
        return metrics(env, sel, att, meta)
    if method in ("model", *MODEL_CONFIGS):
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model = ViewSelector(ModelConfig(**state["mcfg"])).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        # 임계값은 val에서 라벨 표시량에 맞춘다 — 평가 분할을 보지 않기 위해.
        Pv, _, mv = run_model(model, env.dataset("val", risk_source, noise, seq_len), device)
        th = threshold_for_load(Pv, float((mv["aux"] > 0.5).sum(1).mean()),
                                mv["mask"] > 0.5, aux_max)
        P, att, meta = run_model(model, env.dataset(split, risk_source, noise, seq_len), device)
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


def curve(env, method, split, risk_source, noise, ckpt=None, device="cpu", seq_len=None):
    """운영 곡선 — 규칙은 보조 뷰 임계값 θ, 모델은 확률 임계값 τ를 쓴다.

    포착률·정밀도는 표시량에 좌우되므로(B.8) 방법 간 비교는 같은 표시량에서만 성립한다.
    고정 운영점 하나로 비교하면 표시량을 더 많이 쓴 쪽이 포착률에서 이긴다.
    """
    pts = []
    if method in ("model", *MODEL_CONFIGS):
        ds = env.dataset(split, risk_source, noise, seq_len)
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
    ds = env.dataset(split, risk_source, noise)
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
    metrics.json이 없는 실행은 도중에 끊긴 것이라 건너뛴다 — best.ckpt는 학습 중에도 쓰인다.
    """
    for ck in sorted(glob.glob(f"results/{exp}/20*/best.ckpt"), reverse=True):
        if not os.path.exists(os.path.join(os.path.dirname(ck), "metrics.json")):
            continue
        cfg_path = os.path.join(os.path.dirname(ck), "config.yaml")
        try:
            with open(cfg_path, encoding="utf-8") as f:
                ran = yaml.safe_load(f)["base"]["version"]
        except (OSError, KeyError, TypeError):
            continue
        if version is None or ran == version:
            return ck
    return None


def resolve_model(m, args, version):
    """학습 모델 방법 → (체크포인트, 시퀀스 길이). 'model'은 --ckpt/--config, B3·B4는 각자의 설정."""
    cfg_path = args.config if m == "model" else MODEL_CONFIGS[m]
    _, _, tc = load_cfgs(cfg_path)
    ckpt = args.ckpt if (m == "model" and args.ckpt) else latest_ckpt(tc.get("exp_name", "proposed"), version)
    return ckpt, tc["train"].get("seq_len", 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["b1", "b2", "b2c", "b3", "b4", "model", "label"])
    ap.add_argument("--all", action="store_true", help="b1·b2·b2c·b3·b4·model + 의사 라벨(참고) + 표 1")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--risk_source", default="noisy", choices=["gt", "noisy", "none"])
    ap.add_argument("--noise", default="medium", choices=["clean", "light", "medium", "heavy"])
    ap.add_argument("--config", default="configs/train_proposed.yaml")
    ap.add_argument("--version", default=None)
    ap.add_argument("--splits", default=None, help="mini 스모크: splits/scene_split_mini.json")
    ap.add_argument("--gold", default="data/goldset_v1.json", help="골드셋 (없으면 골드 열은 비운다)")
    ap.add_argument("--out_dir", default="results/eval")
    ap.add_argument("--table", default=None, help="표 1 경로 (기본 results/table1[_조건].md)")
    ap.add_argument("--curve", action="store_true", help="운영 곡선(임계값 스윕)으로 비교")
    args = ap.parse_args()

    base, lab, tc = load_cfgs(args.config)
    version = args.version or base["version"]
    env = Env(base, lab, tc, version, args.splits, gold_path=args.gold)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    methods = ["b1", "b2", "b2c", "b3", "b4", "model", "label"] if args.all else [args.method]
    models = {m: resolve_model(m, args, version) for m in methods if m in ("model", *MODEL_CONFIGS)}
    for m, (ck, _) in models.items():
        if not ck:
            print(f"  {m}: {version} 체크포인트 없음 — 건너뜀")
    methods = [m for m in methods if m not in models or models[m][0]]

    os.makedirs(args.out_dir, exist_ok=True)
    cond = f"{args.risk_source}-{args.noise}" if args.risk_source == "noisy" else args.risk_source
    ds0 = env.dataset(args.split, "gt", "clean")
    toks = [ds0.scene_tokens[sc][pos] for sc, pos in ds0.flat]
    load = float(env.df.set_index("sample_token").loc[toks, "n_aux"].mean())   # 라벨 표시량

    curves = {}
    if args.curve or args.all:
        for m in methods:
            if m in ("b1", "label"):
                continue                                   # 표시량이 고정된 한 점뿐
            ck, sl = models.get(m, (None, None))
            curves[m] = curve(env, m, args.split, args.risk_source, args.noise, ck, device, sl)
        with open(os.path.join(args.out_dir, f"curve_{args.split}_{cond}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(dict(split=args.split, condition=cond, load=load,
                           ckpt={m: v[0] for m, v in models.items()}, curves=curves),
                      f, ensure_ascii=False, indent=2)
    if args.curve:
        print(f"\n[운영 곡선] {version} · {args.split} · 입력 {cond}")
        for m, pts in curves.items():
            print(f"\n  {m}")
            for p in pts:
                print(f"    {p['param']:5.2f}  표시 {p['mean_aux']:5.2f}장  포착 {p['recall']:5.1f}%"
                      f"  정밀 {p['precision']:5.1f}%  flicker {p['flicker']:.3f}")
        loads = [0.3, round(load, 2), 0.6, 0.8]
        print(f"\n  같은 표시량에서의 포착률 (선형 보간)")
        print("    " + "".join(f"{l:>9.2f}장" for l in loads))
        for m, pts in curves.items():
            vals = [recall_at_load(pts, l) for l in loads]
            print(f"  {m:6s}" + "".join(f"{v:9.1f}% " if v is not None else "       — " for v in vals))
        return 0

    rows = []
    for m in methods:
        ck, sl = models.get(m, (None, None))
        res = evaluate(env, m, args.split, args.risk_source, args.noise, ck, device, sl)
        res.update(method=m, split=args.split, condition=cond, ckpt=ck,
                   recall_at_load=(recall_at_load(curves[m], load) if m in curves else None))
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
    if args.all:
        path = args.table or ("results/table1.md" if cond == "noisy-medium" else f"results/table1_{cond}.md")
        write_table1(rows, path, version, args.split, cond, load, env.gold)
    return 0


def wilson(p, n, z=1.96):
    """비율의 95% Wilson 신뢰구간 — 표본이 작을 때 정규근사보다 안정적이다."""
    if p is None or not n:
        return None
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def mcnemar_exact(b, c):
    """짝지은 이진 결과의 McNemar 정확검정(양측) — 엇갈린 장면 수 b, c만 본다."""
    from scipy.stats import binom
    n = b + c
    return 1.0 if n == 0 else float(min(1.0, 2 * binom.cdf(min(b, c), n, 0.5)))


def write_table1(rows, path, version, split, cond, load, gold):
    """표 1 (B.8) — 골드 열은 역확률 가중 추정치. 골드셋이 없으면 비운다."""
    pct = lambda v: "—" if v is None else f"{100 * v:.1f}%"
    f3 = lambda v: "—" if v is None else f"{v:.3f}"
    p1 = lambda v: "—" if v is None else f"{v:.1f}%"
    g_n = rows[0].get("gold", {}).get("n") if rows else None
    lines = [f"# 표 1 — {split} 분할, 입력 {cond}\n",
             (f"골드셋 {g_n}장(역확률 가중 — 전체 분포 추정). " if g_n else
              "**골드셋 라벨링 대기 — 골드 열은 비어 있다.** ")
             + "Top-k는 정답이 있는 프레임 기준, '정숙률'은 정답이 '강조 불필요'인 프레임에서 보조 뷰를 "
               "켜지 않은 비율. 평균 표시·포착률·정밀도는 각 방법의 운영점(규칙 θ=0.12, 모델은 val에서 "
               f"라벨 표시량에 맞춘 τ) 값이고, 마지막 열은 같은 표시량({load:.2f}장)에서 보간한 포착률이다(B.8).\n",
             "| 방법 | Top-1 (gold) | Top-2 (gold) | Macro-F1 (gold) | 정숙률 (gold) | Macro-F1 (pseudo) "
             "| 평균 표시 | 포착률 | 정밀도 | Flicker | 포착률 @같은 표시량 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        g = r.get("gold", {}).get("weighted", {})
        lines.append(f"| {METHOD_KO[r['method']]} | {pct(g.get('top1'))} | {pct(g.get('top2'))} | "
                     f"{f3(g.get('f1'))} | {pct(g.get('quiet'))} | {f3(r['pseudo_f1'])} | "
                     f"{r['mean_aux']:.2f}장 | {p1(r['recall'])} | {p1(r['precision'])} | "
                     f"{f3(r['flicker'])} | {p1(r.get('recall_at_load'))} |")
    if gold and rows and "gold" in rows[0]:
        strata = sorted(rows[0]["gold"]["by_stratum"])
        lines += ["", "**골드셋 층별** (비가중) — Top-1 / Macro-F1", "",
                  "| 방법 | " + " | ".join(strata) + " |", "|---|" + "---|" * len(strata)]
        for r in rows:
            bs = r["gold"]["by_stratum"]
            lines.append(f"| {METHOD_KO[r['method']]} | " + " | ".join(
                f"{pct(bs[s]['top1'])} / {f3(bs[s]['f1'])}" for s in strata) + " |")
        lines.append(f"\n애매 표시 {rows[0]['gold']['n_hmm']}장 포함.")
        lines += ["", "**골드셋 비가중 전체** — Top-1의 95% 신뢰구간(Wilson)과 제안 모델과의 짝비교. "
                  "짝비교는 정답이 있는 같은 장면에서 '제안만 맞힘 / 상대만 맞힘' 수와 McNemar 정확검정 p.", "",
                  "| 방법 | n | Top-1 [95% CI] | Top-2 | Macro-F1 | 평균 표시 | 제안만 / 상대만 | p |",
                  "|---|---|---|---|---|---|---|---|"]
        ref = next((r for r in rows if r["method"] == "model"), None)
        for r in rows:
            u = r["gold"]["unweighted"]
            ci = wilson(u["top1"], u["n_nonempty"])
            t1 = "—" if ci is None else f"{100 * u['top1']:.1f}% [{100 * ci[0]:.1f}–{100 * ci[1]:.1f}]"
            pair, pv = "—", "—"
            if ref is not None and r is not ref and "top1_hit" in r["gold"] and "top1_hit" in ref["gold"]:
                a, b = ref["gold"]["top1_hit"], r["gold"]["top1_hit"]
                only_m = sum(a[t] and not b[t] for t in a)
                only_o = sum(b[t] and not a[t] for t in a)
                pair, pv = f"{only_m} / {only_o}", f"{mcnemar_exact(only_m, only_o):.3f}"
            lines.append(f"| {METHOD_KO[r['method']]} | {u['n_nonempty']} | {t1} | {pct(u['top2'])} | "
                         f"{f3(u['f1'])} | {u['mean_aux']:.2f}장 | {pair} | {pv} |")
    lines += ["", "체크포인트:", ""] + [f"- {METHOD_KO[r['method']]}: `{r['ckpt']}`" for r in rows if r.get("ckpt")]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n✅ {path}")


if __name__ == "__main__":
    raise SystemExit(main())

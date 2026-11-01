"""테스트 프로그램 공통 추론 — 내보내기(export.py)와 업로드 처리(server.py)가 같은 코드를 쓴다.

모델 입력은 학습과 같다 (CLAUDE.md B.5):
  img_feat (T,6,576)  MobileNetV3-small GAP — feature_cache와 같은 전처리(224×224 BILINEAR, ImageNet 정규화)
  can      (T,8)      정규화 CAN 7개 + 결측 마스크(1=있음). 업로드에 CAN이 없으면 0 + 마스크 0
  risk     (T,6)      채널 위험도. 검출기가 없으면 0
과거 T=4 keyframe(2초)만 보며, 장면 시작 이전은 0으로 채운다(Dataset과 같은 패딩).

보조 뷰는 val에서 라벨 표시량에 맞춘 임계값 τ로 고른다(06_evaluate.py와 같은 방식) —
results/demo/thresholds.json. 진행 방향(primary)은 상시 표시라 고를 대상이 아니다.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_yaml  # noqa: E402
from src.utils.constants import (  # noqa: E402
    CAMERA_INDEX, CAMERAS, CAN_FEATURES, CAN_NORM, D_CAN, NUM_CAMERAS, SIGNAL_HOLD_S,
)

# 화면 배치 (골드셋 검수 화면과 같다) — 숫자키 1~6이 이 순서다.
GRID = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
FRONT, BACK = CAMERA_INDEX["CAM_FRONT"], CAMERA_INDEX["CAM_BACK"]


def load_cfg(path: str = "configs/demo.yaml") -> dict:
    """데모 설정 + base.yaml(dataroot·seed) + label_v1.yaml."""
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    cfg = load_yaml(p)
    cfg["base"] = load_yaml(os.path.join(ROOT, "configs", "base.yaml"))
    cfg["label"] = load_yaml(os.path.join(ROOT, "configs", "label_v1.yaml"))
    cfg["dataroot"] = os.path.abspath(os.path.expanduser(cfg["base"]["dataroot"]))
    for k in ("out_dir", "thumb_dir"):
        cfg[k] = os.path.join(ROOT, cfg[k])
    return cfg


def label_cfg(cfg: dict):
    from src.labeling.label_function import LabelConfig
    return LabelConfig.from_dict(cfg["label"])


_EV = None


def ev():
    """scripts/06_evaluate.py — 평가와 같은 선택·임계값 함수를 쓴다."""
    global _EV
    if _EV is None:
        spec = importlib.util.spec_from_file_location("ev", os.path.join(ROOT, "scripts", "06_evaluate.py"))
        _EV = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_EV)
    return _EV


def device_name() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------
_MODELS: dict[str, object] = {}


def load_model(ckpt: str, device: str):
    import torch
    from src.models.proposed import ModelConfig, ViewSelector
    key = f"{ckpt}@{device}"
    if key not in _MODELS:
        path = ckpt if os.path.isabs(ckpt) else os.path.join(ROOT, ckpt)
        state = torch.load(path, map_location=device, weights_only=False)
        model = ViewSelector(ModelConfig(**state["mcfg"])).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        _MODELS[key] = model
    return _MODELS[key]


def windows(arr: np.ndarray, T: int) -> np.ndarray:
    """프레임 배열 (N,...) → 과거 T 창 (N,T,...). 장면 시작 이전은 0 (Dataset 패딩과 같다)."""
    N = arr.shape[0]
    out = np.zeros((N, T) + arr.shape[1:], dtype=np.float32)
    for k in range(T):
        lag = T - 1 - k
        out[lag:, k] = arr[:N - lag] if lag else arr
    return out


def predict(model, img: np.ndarray, can: np.ndarray, risk: np.ndarray, T: int,
            device: str, batch: int = 256) -> np.ndarray:
    """한 장면(시간순)의 keyframe별 강조 확률 (N,6)."""
    import torch
    W = {"img_feat": windows(img, T), "can": windows(can, T), "risk": windows(risk, T)}
    out = []
    with torch.no_grad():
        for s in range(0, img.shape[0], batch):
            b = {k: torch.from_numpy(v[s:s + batch]).to(device) for k, v in W.items()}
            out.append(torch.sigmoid(model(b)["attn_logits"]).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, NUM_CAMERAS), np.float32)


def select(P: np.ndarray, primary, tau: float, aux_max: int) -> list[list[int]]:
    """확률 → 보조 뷰 목록 (진행 방향 제외, τ 이상, 상위 aux_max장)."""
    if len(P) == 0:
        return []
    keep = ev().select_from_probs(np.asarray(P), np.asarray(primary, dtype=int), tau, aux_max)
    order = np.argsort(-np.asarray(P), axis=1)
    return [[int(c) for c in order[i] if keep[i, c]] for i in range(len(P))]


def primary_from_speed(speeds, stop_speed: float = 0.5) -> list[int]:
    """진행 방향 뷰 — 후진이면 후방, 정지 중이면 직전 방향 유지(라벨의 stop_hold와 같은 취지)."""
    out, prev = [], FRONT
    for v in speeds:
        if v is None or not np.isfinite(v) or abs(v) < stop_speed:
            cur = prev
        else:
            cur = BACK if v < 0 else FRONT
        out.append(cur)
        prev = cur
    return out


def load_thresholds(cfg: dict) -> dict:
    p = os.path.join(cfg["out_dir"], "thresholds.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def r3(x) -> list:
    return [round(float(v), 3) for v in x]


# ---------------------------------------------------------------------
# 영상 → 특징
# ---------------------------------------------------------------------
class FeatureExtractor:
    """feature_cache와 같은 백본·전처리를 메모리의 RGB 프레임에 적용한다."""

    def __init__(self, device: str, img_size: int = 224):
        from src.data.feature_cache import _IMAGENET_MEAN, _IMAGENET_STD, _build_model
        self.model, self.feat_dim = _build_model(device)
        self.device, self.size = device, img_size
        self.mean, self.std = _IMAGENET_MEAN, _IMAGENET_STD

    def __call__(self, frames_rgb: list[np.ndarray]) -> np.ndarray:
        import torch
        from PIL import Image
        if not frames_rgb:
            return np.zeros((0, self.feat_dim), np.float32)
        xs = []
        for fr in frames_rgb:
            img = Image.fromarray(fr).resize((self.size, self.size), Image.BILINEAR)
            a = (np.asarray(img, dtype=np.float32) / 255.0 - self.mean) / self.std
            xs.append(a.transpose(2, 0, 1))
        with torch.no_grad():
            x = torch.from_numpy(np.stack(xs)).to(self.device)
            return self.model.forward_features(x).mean(dim=(2, 3)).cpu().numpy()


def read_can_csv(path: str, times: list[float]):
    """업로드용 CAN CSV → (can (N,8), 속도 목록, 표시용 값 목록).

    열: t(초, 필수) + CAN_FEATURES 중 아무거나(speed m/s, steering, accel_x, accel_y, yaw_rate,
    left_signal, right_signal). 각 keyframe 시점 **이하**의 마지막 행을 쓴다(과거만 — 모델과 같은 조건).
    지시등은 램프 점멸이라 직전 SIGNAL_HOLD_S초 안에 1이 있으면 켜짐으로 본다(nusc_wrapper.held_signal).
    """
    with open(path, encoding="utf-8-sig") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows or "t" not in rows[0]:
        raise ValueError("CAN CSV에는 t 열(초)이 있어야 합니다")
    rows.sort(key=lambda r: float(r["t"]))
    ts = np.array([float(r["t"]) for r in rows])

    def val(r, k):
        try:
            return float(r.get(k, "") or 0.0)
        except ValueError:
            return 0.0

    can = np.zeros((len(times), D_CAN), np.float32)
    speeds, shown = [], []
    for i, t in enumerate(times):
        j = int(np.searchsorted(ts, t + 1e-6)) - 1
        if j < 0:
            speeds.append(None)
            shown.append(None)
            continue
        raw = {k: val(rows[j], k) for k in CAN_FEATURES}
        lo = int(np.searchsorted(ts, t - SIGNAL_HOLD_S + 1e-6))
        for s in ("left_signal", "right_signal"):
            raw[s] = float(any(val(rows[q], s) > 0.5 for q in range(lo, j + 1)))
        can[i, :len(CAN_FEATURES)] = [raw[k] / CAN_NORM[k] for k in CAN_FEATURES]
        can[i, -1] = 1.0
        speeds.append(raw["speed"])
        shown.append(dict(speed=round(raw["speed"], 2), steer=round(raw["steering"], 1),
                          left=int(raw["left_signal"]), right=int(raw["right_signal"])))
    return can, speeds, shown


class Job:
    """업로드 처리 진행 상황 — 서버가 폴링 응답으로 돌려준다."""

    def __init__(self, job_id: str):
        self.id, self.state, self.msg, self.file, self.error = job_id, "queued", "대기 중", None, None
        self.t0 = time.time()

    def set(self, msg: str):
        self.state, self.msg = "running", msg

    def as_dict(self) -> dict:
        return dict(id=self.id, state=self.state, msg=self.msg, file=self.file, error=self.error,
                    elapsed=round(time.time() - self.t0, 1))


def _open(path: str):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {os.path.basename(path)}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 1e-3:
        fps = 30.0
    return cap, fps, n


def process_upload(job: Job, job_dir: str, spec: dict, cfg: dict, fx: FeatureExtractor,
                   device: str) -> str:
    """업로드 영상 → 장면 JSON (export.py와 같은 형식). 반환: job_dir 기준 JSON 경로.

    spec: {mode: "six"|"mosaic", videos: {카메라 인덱스|"mosaic": 경로}, flip_back: bool,
           can_csv: 경로|None, name: 표시 이름}
    6개 파일 모드에서 빠진 카메라는 특징 0 — 학습 때의 카메라 고장(블랙아웃)과 같은 입력이다.
    """
    import cv2

    up = cfg["upload"]
    key_hz, disp = float(up["key_hz"]), int(up["display_fps"])
    step = int(round(disp / key_hz))
    if step < 1 or abs(step * key_hz - disp) > 1e-6:
        raise ValueError("display_fps는 key_hz의 정수배여야 합니다")
    tw = int(cfg["thumb_width"])

    # (영상 경로, [(타일 → 카메라 인덱스, 잘라낼 영역 함수)])
    sources = []
    if spec["mode"] == "mosaic":
        flip = bool(spec.get("flip_back"))

        def tiles(fr):
            h, w = fr.shape[0] // 2, fr.shape[1] // 3
            out = {}
            for k, cam in enumerate(GRID):
                r, c = divmod(k, 3)
                t = fr[r * h:(r + 1) * h, c * w:(c + 1) * w]
                out[CAMERA_INDEX[cam]] = t[:, ::-1] if (flip and r == 1) else t
            return out
        sources.append((spec["videos"]["mosaic"], tiles))
    else:
        for ci, path in spec["videos"].items():
            sources.append((path, (lambda ci_: (lambda fr: {ci_: fr}))(int(ci))))
    if not sources:
        raise ValueError("영상이 없습니다")

    # 길이 — 가장 긴 영상 기준, 상한 max_seconds
    dur = 0.0
    for path, _ in sources:
        cap, fps, n = _open(path)
        cap.release()
        dur = max(dur, (n / fps) if n > 0 else float(up["max_seconds"]))
    dur = min(dur, float(up["max_seconds"]))
    n_disp = max(1, int(np.floor(dur * disp + 1e-6)))
    n_key = (n_disp - 1) // step + 1
    present = set()

    feats = np.zeros((n_key, NUM_CAMERAS, fx.feat_dim), np.float32)
    img_name = [[None] * NUM_CAMERAS for _ in range(n_disp)]
    os.makedirs(os.path.join(job_dir, "f"), exist_ok=True)

    for si, (path, tiles) in enumerate(sources):
        cap, fps, n = _open(path)
        need: dict[int, list[int]] = {}                  # 원본 프레임 번호 → 표시 프레임 번호들
        for j in range(n_disp):
            idx = int(round(j / disp * fps))
            if n > 0 and idx > n - 1:                    # 이 영상은 끝났다 — 그 뒤는 비워 둔다
                break
            need.setdefault(idx, []).append(j)
        if not need:
            cap.release()
            continue
        top = max(need)
        pend: list[tuple[int, int, np.ndarray]] = []     # (keyframe, 카메라, RGB)

        def flush():
            if not pend:
                return
            F = fx([p[2] for p in pend])
            for (k, ci, _), f in zip(pend, F):
                feats[k, ci] = f
            pend.clear()

        i = 0
        while i <= top:
            if i not in need:
                if not cap.grab():
                    break
                i += 1
                continue
            ok, fr = cap.read()
            if not ok:
                break
            for ci, tile in tiles(fr).items():
                tile = np.ascontiguousarray(tile)
                present.add(ci)
                h, w = tile.shape[:2]
                th = cv2.resize(tile, (tw, max(1, int(round(h * tw / w)))), interpolation=cv2.INTER_AREA)
                name = f"f/{ci}_{i:06d}.jpg"
                cv2.imwrite(os.path.join(job_dir, name), th, [cv2.IMWRITE_JPEG_QUALITY, 85])
                rgb = None
                for j in need[i]:
                    img_name[j][ci] = name
                    if j % step == 0:
                        if rgb is None:
                            rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                        pend.append((j // step, ci, rgb))
            if len(pend) >= 32:
                flush()
            if i % 30 == 0:
                job.set(f"영상 {si + 1}/{len(sources)} 읽는 중 — {i / fps:.1f}/{dur:.1f}초")
            i += 1
        flush()
        cap.release()

    # 영상이 일찍 끝난 카메라는 그 뒤 프레임이 비어 있다 — 특징 0, 학습 때의 카메라 고장과 같은 입력
    times = [k / key_hz for k in range(n_key)]
    if spec.get("can_csv"):
        can, speeds, shown = read_can_csv(spec["can_csv"], times)
    else:
        can, speeds, shown = np.zeros((n_key, D_CAN), np.float32), [None] * n_key, [None] * n_key
    primary = primary_from_speed(speeds)
    risk = np.zeros((n_key, NUM_CAMERAS), np.float32)

    job.set("모델 추론 중")
    th = load_thresholds(cfg)
    lc = label_cfg(cfg)
    src = cfg["sources"]["upload"]
    methods, per = {}, {}
    for m in src["methods"]:
        mc = cfg["methods"][m]
        if m not in th:
            raise ValueError(f"{m}의 임계값이 없습니다 — 먼저 python tools/demo/export.py --thresholds")
        model = load_model(mc["ckpt"], device)
        P = predict(model, feats, can, risk, int(mc.get("seq_len", 4)), device)
        per[m] = (P, select(P, primary, th[m]["tau"], lc.aux_max))
        methods[m] = dict(name=mc["name"], tau=th[m]["tau"], rule=False, risk=mc["risk_source"])

    kfs = []
    for k in range(n_key):
        kfs.append(dict(t=round(times[k], 3), primary=int(primary[k]), can=shown[k],
                        m={m: dict(p=r3(per[m][0][k]), sel=per[m][1][k]) for m in per}))
    rel = os.path.relpath(job_dir, cfg["out_dir"]).replace(os.sep, "/")
    frames = [dict(t=round(j / disp, 3), k=j // step,
                   img=[(f"data/{rel}/{n}" if n else None) for n in img_name[j]]) for j in range(n_disp)]
    missing = [c for c in range(NUM_CAMERAS) if c not in present]
    out = dict(source="upload", scene=spec.get("name") or os.path.basename(job_dir), id=os.path.basename(job_dir),
               has_gt=False, methods=methods, default=src["default"], key_hz=key_hz,
               missing_cams=missing, can_given=bool(spec.get("can_csv")),
               note=("CAN 없음 — 진행 방향은 전방으로 둡니다" if not spec.get("can_csv") else ""),
               keyframes=kfs, frames=frames)
    path = os.path.join(job_dir, "scene.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    # 원본 영상은 지운다 — 재생은 추출한 프레임으로 한다
    shutil.rmtree(os.path.join(job_dir, "src"), ignore_errors=True)
    return path

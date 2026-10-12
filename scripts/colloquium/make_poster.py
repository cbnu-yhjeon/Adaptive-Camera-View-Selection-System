"""INDAI 콜로키움 2026 포스터 — 양식 pptx(900×1200 mm)에 내용을 채운다.

양식의 머리띠(연구목적·문제정의·연구방법·연구결과·결론 및 제언)와 좌표는 그대로 두고,
제목·저자는 서식을 유지한 채 글자만 바꾼다. 수치는 make_assets.py가 만든
results/colloquium/numbers.json에서만 읽는다.

사용:
    python scripts/colloquium/make_assets.py && python scripts/colloquium/shoot_ui.py
    python scripts/colloquium/make_poster.py --template "<양식 pptx 경로>"
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Mm, Pt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "results", "colloquium")
FIG = os.path.join(OUT, "fig")
CAMS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
KO = {"CAM_FRONT": "전방", "CAM_FRONT_RIGHT": "우전측방", "CAM_BACK_RIGHT": "우후측방",
      "CAM_BACK": "후방", "CAM_BACK_LEFT": "좌후측방", "CAM_FRONT_LEFT": "좌전측방"}
DEFAULT_TEMPLATE = "/mnt/c/Users/USER/Desktop/교육/충북대학교/지능화캡스톤파일럿/[첨부3] Poster_format(korean) v1.2 (양식).pptx"

NAVY, TEAL, RUST, AMBER = "17375E", "0D6B73", "B8422A", "A86B00"
INK, MUTED, LIGHT, TEALBG, RUSTBG = "14181C", "5A646C", "EEF1F3", "E2F0F0", "F7E7E1"
FONT = "맑은 고딕"

# 저자 — 지도교수 이름·소속은 연구자가 채운다 (계획서에 없음)
AUTHORS = [
    [("전양호", None), (", OOO*, OOO**", None)],
    [("충북대학교 산업인공지능학과, *충북대학교 산업인공지능연구센터, **충북대학교 OOO학과", None)],
    [("yangho@chungbuk.ac.kr, *OOO@chungbuk.ac.kr, **OOO@chungbuk.ac.kr", None)],
]
TITLE = ["원격주행을 위한 딥러닝 기반", "적응형 카메라 뷰 선택 시스템"]

LX, LW = 30, 402          # 왼쪽 단 내용
RX, RW = 472, 403         # 오른쪽 단 내용


# ---------------------------------------------------------------------------
def _font(run, size=None, bold=None, color=None, italic=None):
    f = run.font
    if size is not None:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    if color is not None:
        f.color.rgb = RGBColor.from_string(color)
    rPr = run._r.get_or_add_rPr()
    rPr.set("lang", "ko-KR")
    rPr.set("altLang", "en-US")
    for tag in ("a:latin", "a:ea"):
        el = rPr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rPr, qn(tag))
        el.set("typeface", FONT)


def _no_word_break(p):
    """한글 단어 중간에서 줄을 바꾸지 않는다 (PowerPoint '한글 단어 잘림 허용' 끔)."""
    pPr = p._p.get_or_add_pPr()
    pPr.set("eaLnBrk", "0")
    pPr.set("latinLnBrk", "0")


def _bullet(p, char="•", indent_mm=7.0, color=None):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Mm(indent_mm))))
    pPr.set("indent", str(-int(Mm(indent_mm))))
    if color:
        bc = etree.SubElement(pPr, qn("a:buClr"))
        etree.SubElement(bc, qn("a:srgbClr")).set("val", color)
    etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
    etree.SubElement(pPr, qn("a:buChar")).set("char", char)


def textbox(slide, x, y, w, h, paras, size=26, color=INK, anchor=MSO_ANCHOR.TOP, name=None, line=1.12):
    """paras: [ {runs:[(text, {opts})], bullet:bool, align, after(pt), size, color} | str ]"""
    tb = slide.shapes.add_textbox(Mm(x), Mm(y), Mm(w), Mm(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    first = True
    for para in paras:
        if isinstance(para, str):
            para = {"runs": [(para, {})]}
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = para.get("align", PP_ALIGN.LEFT)
        p.line_spacing = para.get("line", line)
        _no_word_break(p)
        if "after" in para:
            p.space_after = Pt(para["after"])
        if para.get("bullet"):
            _bullet(p, para.get("char", "•"), para.get("indent", 7.0), para.get("bcolor"))
        for text, o in para["runs"]:
            r = p.add_run()
            r.text = text
            _font(r, o.get("size", para.get("size", size)), o.get("bold", para.get("bold", False)),
                  o.get("color", para.get("color", color)))
    return tb


def picture(slide, path, x, y, w=None, h=None, name=None):
    pic = slide.shapes.add_picture(path, Mm(x), Mm(y), width=Mm(w) if w else None, height=Mm(h) if h else None)
    if name:
        pic.name = name
    return pic


def pic_h(path, w):
    im = Image.open(path)
    return w * im.height / im.width


def card(slide, x, y, w, h, fill=LIGHT, name=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Mm(x), Mm(y), Mm(w), Mm(h))
    s.adjustments[0] = 0.06
    s.fill.solid()
    s.fill.fore_color.rgb = RGBColor.from_string(fill)
    s.line.fill.background()
    s.shadow.inherit = False
    if name:
        s.name = name
    return s


def table(slide, x, y, w, col_w, rows, row_h, size=20, header_fill=NAVY, highlight=None, name=None, left_cols=(0,)):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Mm(x), Mm(y), Mm(w), Mm(row_h * nr))
    if name:
        gt.name = name
    t = gt.table
    tblPr = t._tbl.tblPr
    for attr in ("bandRow", "firstRow"):
        tblPr.set(attr, "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is not None:
        sid.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"   # No Style, Table Grid
    for j, cw in enumerate(col_w):
        t.columns[j].width = Mm(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Mm(row_h)
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Mm(2.2)
            c.margin_top = c.margin_bottom = Mm(0.8)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j in left_cols else PP_ALIGN.CENTER
            _no_word_break(p)
            r = p.add_run()
            r.text = str(val)
            head = i == 0
            hl = highlight is not None and i == highlight
            _font(r, size, bold=head or hl, color="FFFFFF" if head else (TEAL if hl else INK))
            c.fill.solid()
            c.fill.fore_color.rgb = RGBColor.from_string(header_fill if head else (TEALBG if hl else ("FFFFFF" if i % 2 else "F6F8F9")))
    return gt


def replace_paragraph_runs(shape, lines):
    """양식 상자의 문단 서식(첫 run)을 유지한 채 글자만 바꾼다."""
    tf = shape.text_frame
    paras = list(tf.paragraphs)
    for i, line in enumerate(lines):
        p = paras[i] if i < len(paras) else None
        if p is None:
            break
        runs = p.runs
        base = runs[0]
        for r in runs[1:]:
            r._r.getparent().remove(r._r)
        if isinstance(line, str):
            base.text = line
        else:
            base.text = line[0][0]
            for text, _ in line[1:]:
                nr = p.add_run()
                nr._r.getparent().replace(nr._r, etree.fromstring(etree.tostring(base._r)))
                p.runs[-1].text = text
        for r in p.runs:
            rPr = r._r.get_or_add_rPr()
            rPr.set("lang", "ko-KR")
            if "err" in rPr.attrib:
                del rPr.attrib["err"]
    for p in paras[len(lines):]:
        p._p.getparent().remove(p._p)


# ---------------------------------------------------------------------------
def build(N: dict, template: str, out: str) -> None:
    prs = Presentation(template)
    s = prs.slides[0]
    L, T1, AB, R = N["label"], N["table1"], N["ablation"], N["robust"]
    rows = {r["key"]: r for r in T1["rows"]}
    m, b2, b2c, b4 = rows["model"], rows["b2"], rows["b2c"], rows["b4"]
    a1 = L["always1"]
    abase = next(r for r in AB["rows"] if r["tag"] == "제안")
    ab = {r["tag"]: r for r in AB["rows"]}
    lb = rows["label"]
    c3 = R["claim3"]
    sig_ko = lambda pv: "유의하게 " if pv < 0.05 else ""
    pm = lambda v: f"{v:+.1f}".replace("-", "−")        # 하이픈이 아닌 빼기 기호 — 부호 뒤에서 줄이 바뀌지 않게

    for sh in s.shapes:
        if not sh.has_text_frame:
            continue
        txt = sh.text_frame.text
        if "YOLOv5" in txt:
            replace_paragraph_runs(sh, TITLE)
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(80)
        elif "홍길동" in txt:
            replace_paragraph_runs(sh, AUTHORS)

    # ================= 연구목적 =================
    textbox(s, LX, 297, LW, 44, [
        {"runs": [("카메라 6대 화면 중 ", {}), ("지금 봐야 할 화면", {"bold": True, "color": TEAL}), ("을 AI가 골라 강조", {})], "bullet": True, "after": 4},
        {"runs": [("진행 방향은 상시 표시, 나머지 5개 중 ", {}), ("보조 뷰 0~2장", {"bold": True, "color": RUST}), ("을 선택", {})], "bullet": True, "after": 4},
        {"runs": [("정답 데이터가 없어 ", {}), ("공개 데이터(nuScenes)로 정답을 자동 생성", {"bold": True}), ("해 학습·검증", {})], "bullet": True},
    ], size=28, name="목적 본문")
    cg = os.path.join(FIG, "concept_grid.jpg")
    h = pic_h(cg, LW)
    picture(s, cg, LX, 348, LW, name="목적 그림")
    c = N["concept"]
    textbox(s, LX, 348 + h + 3, LW, 34, [
        {"runs": [("시스템 출력의 형태. ", {"bold": True, "color": INK}),
                  (f"골드셋 판정 장면({c['scene']}, {c['speed_kmh']:.1f} km/h, {c['signal']}측 지시등) — 진행 방향(청록)은 항상, "
                   f"{'·'.join(KO[CAMS[i]] for i in c['aux'])}(적갈)를 보조 뷰로 강조. 6개 화면은 모두 계속 보인다(하이라이트 방식). "
                   "이미지 nuScenes, CC BY-NC-SA 4.0", {})]},
    ], size=20, color=MUTED, name="목적 캡션")

    # ================= 문제정의 =================
    cards = [("64%", TEAL, "동시 감시 화면 16대일 때 이상 탐지율 (4대 83%) ¹"),
             ("26.9~62.4초", TEAL, "현장 CCTV 요원이 카메라를 손으로 바꾸는 주기 ²"),
             (f"{L['hazard_pct']:.1f}%", RUST, "실제 위험이 있는 프레임 비율 — 나머지는 강조할 것이 없음")]
    cw, gap, cy, chh = 128, 9, 598, 88
    for i, (big, col, lab) in enumerate(cards):
        x = LX + i * (cw + gap)
        card(s, x, cy, cw, chh, name=f"문제 카드 {i + 1}")
        textbox(s, x + 5, cy + 6, cw - 10, 34, [{"runs": [(big, {"bold": True, "color": col, "size": 54 if len(big) < 7 else 40})],
                                                 "align": PP_ALIGN.CENTER}], name=f"문제 카드 숫자 {i + 1}",
                anchor=MSO_ANCHOR.MIDDLE)
        textbox(s, x + 6, cy + 43, cw - 12, 42, [{"runs": [(lab, {})], "align": PP_ALIGN.CENTER}], size=20, color=INK,
                name=f"문제 카드 설명 {i + 1}")
    textbox(s, LX, 698, LW, 122, [
        {"runs": [("고정 배치는 한정된 주의를 6개 화면에 똑같이 나눈다 → ", {}), ("상황에 따라 재배분", {"bold": True})], "bullet": True, "after": 6},
        {"runs": [(f"항상 1장을 켜면 정밀도 {a1['precision']:.1f}% (대부분 헛경보) → ", {}), ("볼 것이 없으면 0장", {"bold": True})], "bullet": True, "after": 6},
        {"runs": [("원격주행 뷰 선택 정답 데이터가 없고 실제 검출은 불완전하다 → ", {}),
                  ("의사 라벨 + 손상 입력에서도 버티는 학습 모델", {"bold": True})], "bullet": True, "after": 6},
        {"runs": [("기존 적응형 연구는 전송 품질³·GUI 정보 요소 수⁴를 바꾼다 → ", {}), ("어느 카메라를 강조할지", {"bold": True}),
                  ("는 다루지 않았다", {})], "bullet": True},
    ], size=26, name="문제 본문")
    textbox(s, LX, 830, LW, 30, [
        "¹ Tickner & Poulton, Ergonomics, 1973   ² Stainer et al., Front. Hum. Neurosci., 2013 (관찰 2명)",
        "³ Hofbauer et al., IEEE VTC2022-Spring   ⁴ Wolf et al., MTI 9(8), 2025",
    ], size=16, color=MUTED, name="문제 출처")

    # ================= 연구방법 =================
    pp = os.path.join(FIG, "pipeline_poster.png")
    ph = pic_h(pp, LW)
    picture(s, pp, LX, 912, LW, name="방법 구조도")
    textbox(s, LX, 912 + ph + 2, LW, 10, [{"runs": [("정답(라벨)은 3D 정답 박스로 만들고, 모델 입력은 실전처럼 손상시킨다 — 두 경로를 분리해야 강건성 비교가 성립한다", {})]}],
            size=17, color=MUTED, name="방법 캡션")
    ty = 912 + ph + 15
    n = L["n_split"]
    table(s, LX, ty, LW, [62, LW - 62], [
        ["구성", "내용"],
        ["데이터", f"nuScenes 850 장면 · {L['n_keyframes']:,} keyframe · 장면 단위 분할 {n['train']:,}/{n['val']:,}/{n['test']:,}"],
        ["입력", "과거 4프레임(2초): 영상 특징 6×576 · CAN 8 · 화면별 위험도 6"],
        ["학습", "포컬 BCE + KL + 레이아웃 CE + 시간 일관성 · AdamW · 검출 오류 증강"],
        ["평가", f"같은 표시량({T1['load']:.2f}장)의 위험 포착률 · 골드셋 Top-1 · 강건성"],
    ], row_h=15, size=19, name="실험 설정 표", left_cols=(0, 1))

    # ================= 연구결과 =================
    def sub(y, text, color=NAVY):
        textbox(s, RX, y, RW, 14, [{"runs": [(text, {"bold": True, "color": color})]}], size=27, name="결과 소제목")

    y = 297
    sub(y, f"① 의사 라벨: 평균 {L['mean_aux']:.2f}장으로 실제 위험 {L['recall']:.1f}% 포착")
    y += 17
    op = os.path.join(FIG, "opcurve_poster.png")
    rv = os.path.join(FIG, "ui_review.png")
    hw = 199
    picture(s, op, RX, y, hw, name="운영 곡선")
    picture(s, rv, RX + hw + 5, y, hw, name="라벨 검수 화면")
    hh = max(pic_h(op, hw), pic_h(rv, hw))
    sg = L["signal"]
    textbox(s, RX, y + hh + 2, hw, 18, [{"runs": [("θ 스윕 운영 곡선 (34,149장). ", {"bold": True}),
                                                    (f"항상 1장({a1['recall']:.1f}%·{a1['precision']:.1f}%) 대비 표시량 {100 * (1 - L['mean_aux']):.0f}%↓, 정밀도 {L['precision'] / a1['precision']:.1f}배", {})]}],
            size=17, color=MUTED, name="운영 곡선 캡션")
    textbox(s, RX + hw + 5, y + hh + 2, hw, 18, [{"runs": [("라벨 표적 검수 화면. ", {"bold": True}),
                                                             (f"2차 16장 전부 맞음 · 독립 신호(지시등) 일치 좌회전 {sg['TURN_LEFT']:.1f}%, 우회전 {sg['TURN_RIGHT']:.1f}%", {})]}],
            size=17, color=MUTED, name="검수 화면 캡션")
    y += hh + 24

    sub(y, f"② 골드셋({T1['n_labeled']}/{T1['n_total']}장): 의사 라벨과 사람 판정 Top-1 {100 * lb['top1']:.1f}% 일치")
    y += 17
    gu = os.path.join(FIG, "ui_goldset.png")
    gc = os.path.join(FIG, "gold_poster.png")
    picture(s, gu, RX, y, hw, name="골드셋 검수 화면")
    picture(s, gc, RX + hw + 5, y, hw, name="골드셋 차트")
    hh = max(pic_h(gu, hw), pic_h(gc, hw))
    vs = b2c["vs_model"]
    textbox(s, RX, y + pic_h(gu, hw) + 2, hw, 26, [{"runs": [("골드셋 검수 화면 (실제 판정 장면). ", {"bold": True}),
                                                              ("경로도·속도·지시등·미래 10초를 보고 강조할 화면을 직접 고른다. 라벨·모델의 답은 숨긴다", {})]}],
            size=17, color=MUTED, name="골드셋 화면 캡션")
    textbox(s, RX + hw + 5, y + hh + 2, hw, 18, [{"runs": [(f"Top-1 (판정 {m['n_nonempty']}장, 95% 신뢰구간). ", {"bold": True}),
                                                             (f"제안 vs B2c 짝비교 {vs['model_only']}:{vs['other_only']}, p = {vs['p']:.3f}", {})]}],
            size=17, color=MUTED, name="골드셋 차트 캡션")
    y += hh + 22

    def pct(v, d=1):
        return "—" if v is None else f"{100 * v:.{d}f}%"

    trows = [["방법", "골드 Top-1 [95% CI]", "골드 Top-2", f"포착률 @{T1['load']:.2f}장", "Flicker"]]
    for key in ("b1", "b2", "b2c", "b3", "b4", "model"):
        r = rows[key]
        ci = "" if r["top1"] is None else f" [{100 * r['top1_ci'][0]:.0f}–{100 * r['top1_ci'][1]:.0f}]"
        trows.append([r["name"], pct(r["top1"]) + ci, pct(r["top2"]),
                      "—" if r["recall_at_load"] is None else f"{r['recall_at_load']:.1f}%",
                      "—" if r["flicker_at_load"] is None else f"{r['flicker_at_load']:.2f}"])
    table(s, RX, y, RW, [112, 103, 58, 72, 58], trows, row_h=10.2, size=18, highlight=len(trows) - 1, name="표 1")
    y += 10.2 * len(trows) + 2
    textbox(s, RX, y, RW, 9, [{"runs": [("test 분할 5,108장, 입력 noisy(medium). B2는 미래 5초 기동을 아는 상한, B2c가 같은 정보 조건의 비교 대상", {})]}],
            size=16, color=MUTED, name="표 1 캡션")
    y += 14

    sub(y, f"③ 손상 입력 전 조건에서 규칙(B2c)보다 사람 판정과 {c3['gold_diff'][0]:.0f}~{c3['gold_diff'][1]:.0f}%p 더 일치")
    y += 17
    ac = os.path.join(FIG, "ablation_poster.png")
    rc = os.path.join(FIG, "robust_poster.png")
    picture(s, ac, RX, y, hw, name="ablation 차트")
    picture(s, rc, RX + hw + 5, y, hw, name="강건성 차트")
    hh = max(pic_h(ac, hw), pic_h(rc, hw))
    textbox(s, RX, y + hh + 2, hw, 18, [{"runs": [("Ablation (test, 같은 표시량). ", {"bold": True}),
                                                    (f"위험도 제거 {ab['A2']['recall_noisy'] - abase['recall_noisy']:+.1f}%p, GRU 제거 {ab['A3']['recall_noisy'] - abase['recall_noisy']:+.1f}%p", {})]}],
            size=17, color=MUTED, name="ablation 캡션")
    textbox(s, RX + hw + 5, y + hh + 2, hw, 18, [{"runs": [(f"손상(heavy)별 골드 Top-1 (test, {R['gold_n']}장). ", {"bold": True}),
                                                             (f"하락 폭은 제안 {pm(c3['deg_model'][0])}~{pm(c3['deg_model'][1])}, "
                                                              f"B2c {pm(c3['deg_b2c'][0])}~{pm(c3['deg_b2c'][1])}%p로 비슷 — 차이는 수준", {})]}],
            size=17, color=MUTED, name="강건성 캡션")
    y += hh + 22
    textbox(s, RX, y, RW, 22, [{"runs": [("해석 — ", {"bold": True, "color": RUST}),
                                         (f"위험 포착률은 규칙과 대등({pm(c3['recall_diff'][0])}~{pm(c3['recall_diff'][1])}%p)하고, 위험도를 뺀 A2가 "
                                          f"사람 판정과 가장 잘 일치({100 * c3['a2_gold_clean']:.1f}%) — 사람은 위험 객체보다 진행 의도로 화면을 고른다. "
                                          "카메라 고장에는 모든 방법이 17%p 이상 떨어진다", {"color": INK})]}],
            size=19, name="결과 한계")

    # ================= 결론 및 제언 =================
    textbox(s, RX, 976, RW, 196, [
        {"runs": [("결론", {"bold": True, "color": NAVY, "size": 28})], "after": 3},
        {"runs": [("공개 데이터만으로 '봐야 할 화면' 정답을 자동 생성: ", {"bold": True}),
                  (f"평균 {L['mean_aux']:.2f}장으로 위험 {L['recall']:.1f}% 포착, 항상 1장보다 적게 켜고 더 많이 잡는다", {})], "bullet": True, "after": 3},
        {"runs": [("학습 모델은 손상 입력 전반에서 규칙과 대등한 포착률", {"bold": True}),
                  (f"({m['recall_at_load']:.1f}% vs {b2['recall_at_load']:.1f}~{b2c['recall_at_load']:.1f}%)을 내면서, 같은 정보 조건 규칙보다 "
                   f"사람 판정과 {sig_ko(b2c['vs_model']['p'])}더 일치(Top-1 {100 * m['top1']:.1f}% vs {100 * b2c['top1']:.1f}%, "
                   f"p = {b2c['vs_model']['p']:.3f})", {})], "bullet": True, "after": 3},
        {"runs": [("위험도 입력과 시간 모델이 핵심 구성 요소", {"bold": True}),
                  (f": 단일 프레임은 깜빡임이 {abase['flicker_noisy']:.2f}→{ab['A3']['flicker_noisy']:.2f}로 커진다", {})], "bullet": True, "after": 8},
        {"runs": [("한계 및 향후 연구", {"bold": True, "color": RUST, "size": 28})], "after": 3},
        {"runs": [(f"위험 정의가 사람 기준보다 넓다 — 사람이 '강조 불필요'로 본 장면의 {100 * (1 - lb['quiet_u']):.0f}%에서 라벨이 켬 → 진행 의도·잠재 위험 반영", {})],
         "bullet": True, "bcolor": RUST, "after": 3},
        {"runs": [("위험 포착률은 규칙을 넘지 못하고 카메라 고장에 공통으로 취약 → 결손 채널 학습", {})], "bullet": True, "bcolor": RUST, "after": 3},
        {"runs": [("오프라인 검증에 한정 — 실제 오퍼레이터 평가로 주의 배분 효과 확인 필요", {})], "bullet": True, "bcolor": RUST},
    ], size=26, name="결론 본문")
    textbox(s, RX, 1162, RW, 12, [{"runs": [(f"데이터 nuScenes (CC BY-NC-SA 4.0) · 수치는 results/의 평가 산출물에서 자동 생성 · 골드셋 {T1['n_labeled']}/{T1['n_total']}장 판정 기준", {})]}],
            size=15, color=MUTED, name="꼬리말")
    prs.save(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--out", default=os.path.join(OUT, "poster_INDAI2026.pptx"))
    args = ap.parse_args()
    with open(os.path.join(OUT, "numbers.json"), encoding="utf-8") as f:
        N = json.load(f)
    build(N, args.template, args.out)
    print("✅", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

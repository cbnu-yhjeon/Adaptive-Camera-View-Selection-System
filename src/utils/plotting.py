"""matplotlib 공용 설정 — 한글 글꼴.

WSL의 리눅스 쪽에는 한글 글꼴이 없어 그림의 한글이 네모로 깨진다. 윈도우 글꼴 폴더의
맑은 고딕을 직접 등록한다. 없으면 시스템에서 찾을 수 있는 한글 글꼴을 순서대로 시도한다.
"""

from __future__ import annotations

import os

_CANDIDATES = [
    "/mnt/c/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def use_korean_font() -> str | None:
    """사용 가능한 한글 글꼴을 등록하고 기본 글꼴로 지정한다. 등록한 글꼴 이름을 돌려준다."""
    import matplotlib
    from matplotlib import font_manager
    for path in _CANDIDATES:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False   # 한글 글꼴의 '−' 깨짐 방지
            return name
    return None

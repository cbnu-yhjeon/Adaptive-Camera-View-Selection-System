"""콜로키움 논문 hwp 변환 — Word 문서(make_paper.py 산출)를 한컴오피스 한글로 열어 hwp로 저장한다.

WSL에서 Windows의 한글 COM 자동화(HWPFrame.HwpObject)를 powershell로 부른다. 한글이 직접 뽑은
PDF(paper_INDAI2026_hwp.pdf)도 함께 저장해 변환 뒤 배치를 확인할 수 있게 한다.

주의
  · 표 너비가 '자동'이면 한글이 쪽 전체 폭으로 펼쳐 단을 넘는다 — make_paper.py가 고정 너비로 쓴다.
  · 자동화 보안 모듈이 없는 PC에서는 파일 접근 허용 창이 뜰 수 있다(이 PC에서는 뜨지 않았다).
    창이 뜨면 '허용'을 누른다. 응답이 없으면 timeout으로 끝난다.

사용:
    python scripts/colloquium/make_hwp.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "results", "colloquium")


def winpath(p: str) -> str:
    return subprocess.check_output(["wslpath", "-w", p], text=True).strip()


def main() -> int:
    src = os.path.join(OUT, "paper_INDAI2026.docx")
    hwp = os.path.join(OUT, "paper_INDAI2026.hwp")
    pdf = os.path.join(OUT, "paper_INDAI2026_hwp.pdf")
    ps = f"""$ErrorActionPreference = 'Stop'
$hwp = New-Object -ComObject HWPFrame.HwpObject
try {{ $hwp.XHwpWindows.Item(0).Visible = $false }} catch {{}}
$null = $hwp.SetMessageBoxMode(0x2FFF1)
if (-not $hwp.Open('{winpath(src)}', '', 'forceopen:true')) {{ throw 'open failed' }}
if (-not $hwp.SaveAs('{winpath(OUT)}\\{os.path.basename(hwp)}', 'HWP', '')) {{ throw 'save hwp failed' }}
if (-not $hwp.SaveAs('{winpath(OUT)}\\{os.path.basename(pdf)}', 'PDF', '')) {{ throw 'save pdf failed' }}
Write-Output ("pages=" + $hwp.PageCount)
$hwp.Clear(1)
$hwp.Quit()
"""
    # powershell.exe가 읽을 수 있도록 Windows 쪽 임시 폴더에 둔다
    tmp_dir = "/mnt/c/Users/USER/AppData/Local/Temp"
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", dir=tmp_dir, delete=False, encoding="utf-8-sig") as f:
        f.write(ps)
        script = f.name
    try:
        r = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", winpath(script)],
                           capture_output=True, text=True, timeout=180)
    finally:
        os.remove(script)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr.strip(), file=sys.stderr)
        return r.returncode
    print("✅", hwp)
    return 0


if __name__ == "__main__":
    sys.exit(main())

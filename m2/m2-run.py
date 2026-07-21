"""
短啟動器：避免 VS Code↔Isaac UDP 傳輸整份腳本時觸發 WinError 10040。

Send to Isaac 時程式碼以字串執行，__file__ / cwd 都不可靠
（cwd 常是 isaac-sim 安裝目錄）。因此用「相對路徑候選」在各磁碟搜尋。
"""
import os
import runpy
import string
from pathlib import Path

import carb

name = "m2-urdf.py"
rel = Path("Desktop") / "Experiment" / "DT_learning" / name

candidates = [Path.cwd() / name]
candidates += [Path(f"{d}:/") / rel for d in string.ascii_uppercase if Path(f"{d}:/").exists()]

target = next((p.resolve() for p in candidates if p.is_file()), None)
if target is None:
    raise FileNotFoundError(f"Cannot find {name}")

carb.log_warn(f"=== [m2-run] loading: {target} ===")
runpy.run_path(str(target), run_name="__main__")

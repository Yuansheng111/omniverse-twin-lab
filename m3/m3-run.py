"""
短啟動器：避免 VS Code↔Isaac UDP 傳輸整份腳本時觸發 WinError 10040。

Send to Isaac 時程式碼以字串執行，__file__ / cwd 都不可靠
（cwd 常是 isaac-sim 安裝目錄）。因此用「相對路徑候選」在各磁碟搜尋。
"""
import runpy
import string
from pathlib import Path

import carb

name = "m3-obstacle+Lula_RRT.py"
project_rel = Path("Desktop") / "Experiment" / "DT_learning"
rel = project_rel / "m3" / name

candidates = [Path.cwd() / name]
candidates += [Path(f"{d}:/") / rel for d in string.ascii_uppercase if Path(f"{d}:/").exists()]

target = next((p.resolve() for p in candidates if p.is_file()), None)

# 後備：檔案若被移到別的子資料夾，直接在專案根目錄底下遞迴找
if target is None:
    for d in string.ascii_uppercase:
        root = Path(f"{d}:/") / project_rel
        if root.is_dir():
            target = next((p.resolve() for p in root.rglob(name)), None)
            if target is not None:
                break

if target is None:
    raise FileNotFoundError(f"Cannot find {name}")

carb.log_warn(f"=== [m3-run] loading: {target} ===")
runpy.run_path(str(target), run_name="__main__")

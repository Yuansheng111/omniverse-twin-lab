# 讀 m5 的 npz：不開模擬器，只要有 NumPy。
# 用法（在 m5 目錄，或任何有 numpy 的 python）：
#   python inspect_dataset.py
# 若系統 python 沒有 numpy，可用 Isaac Sim 那份：
#   D:\Desktop\git\tt\IsaacLab\_isaac_sim\kit\python\python.exe inspect_dataset.py

from pathlib import Path
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NPZ_PATH = Path(__file__).resolve().parent / "datasets" / "run_001" / "trajectory_0.npz"


def main() -> None:
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"找不到 {NPZ_PATH}，先跑 collect_franka_dataset.py")

    data = np.load(NPZ_PATH)
    print(f"檔案: {NPZ_PATH}")
    print(f"裡面的鍵: {list(data.keys())}\n")

    for key in data.files:
        arr = data[key]
        print(f"{key:12s}  shape={str(arr.shape):20s}  dtype={arr.dtype}")

    rgb = data["rgb"]
    pose = data["ee_pose"]
    actions = data["actions"]
    times = data["timestamps"]

    print("\n--- 第 0 步 ---")
    print(f"時間      {times[0]}")
    print(f"手掌 xyz  {pose[0, :3]}")
    print(f"動作      {actions[0]}")
    print(f"畫面平均亮度 {float(rgb[0].mean()):.1f}  (全黑會接近 0)")

    print("\n--- 最後一步 ---")
    print(f"時間      {times[-1]}")
    print(f"手掌 xyz  {pose[-1, :3]}")
    print(f"畫面平均亮度 {float(rgb[-1].mean()):.1f}")

    print("\n圖用人眼看: datasets/run_001/rgb_0000.png")
    print("程式看某一幀: data['rgb'][步數]  → 形狀 (200, 200, 3)")


if __name__ == "__main__":
    main()

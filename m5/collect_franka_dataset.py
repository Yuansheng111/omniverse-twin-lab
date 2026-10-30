# =============================================================================
# m5 在做什麼（請先讀這段）
# =============================================================================
# m4 是「讓小車自己學會站穩(Isaac-Cartpole-v0)」，產出是一條獎勵曲線。
# m5 的這個檔案是「從模擬器錄一段資料」，產出是一份資料檔。還沒有在訓練策略。
#
# 為什麼要錄資料？之後若要做模仿學習、視覺策略、或把真機資料接進來，
# 都需要同一種格式：每一步對齊的「我看到什麼 / 手臂在哪 / 我下了什麼指令」。
#
# 這支腳本錄 100 步、1 條軌跡，存成 NumPy 壓縮包（副檔名 .npz）：
#   rgb         手腕相機彩色圖，每張 200x200、三個顏色通道
#   ee_pose     末端手掌在世界座標的位置 xyz + 旋轉四元數，長度 7
#   actions     該步送給 7 個手臂關節的指令
#   timestamps  該步對應的模擬時間（秒）
#
# 四個陣列的第 0 維都是「第幾步」。第 i 步的圖、位姿、動作、時間是同一瞬間。
# 這叫對齊；對齊壞掉，後面再漂亮的模型也會學錯。
#
# 環境：官方 Isaac-Reach-Franka-v0（Franka Panda 伸手去追目標）。
# 官方這份任務沒有手腕相機，所以我們在開環境前自己掛上一顆（視覺那一層）。
# 動作目前是亂抽，目的是先打通「模擬器 → 檔案」這條管線；
# 之後可以把亂抽換成你規劃好的關節軌跡。
#
# 啟動方式（在 IsaacLab 目錄）：
#   .\isaaclab.bat -p ..\m5\collect_franka_dataset.py --headless --enable_cameras
# =============================================================================

# -----------------------------------------------------------------------------
# 第一段：先開 Isaac Sim，再 import 其他東西
# -----------------------------------------------------------------------------
# Isaac Lab 的環境、機器人、相機都活在 Isaac Sim 裡面。
# 若先 import gym / 任務套件，再開模擬器，常會因為擴充還沒載入而炸掉。
# 所以官方範例（包含 m4 的 train.py）都是：啟動器 → 模擬器起來 → 其餘程式。
# 下面這句英文字串是官方慣例，提醒「這行之前不要亂 import 模擬相關套件」。

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect a tiny Franka RGB + end-effector pose dataset.")
parser.add_argument("--num_steps", type=int, default=100, help="要錄幾步。預設 100。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 相機要靠模擬器「畫出畫面」才有彩色圖。enable_cameras 會打開離屏渲染
# （沒有跳出視窗也會畫，適合 --headless）。
# 就算你在命令列忘了打 --enable_cameras，這裡也強制打開，避免錄到空圖。
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# -----------------------------------------------------------------------------
# 第二段：模擬器已起來，現在才能載入環境與數學工具
# -----------------------------------------------------------------------------
import numpy as np
import torch
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.math import quat_unique

# 這行看起來沒被使用，其實有副作用：import 時會向 Gymnasium 登記任務名稱，
# 後面 gym.make("Isaac-Reach-Franka-v0") 才找得到這個環境。
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# 輸出放在本腳本旁邊，不寫進 IsaacLab 原始碼目錄，避免之後 git pull 被蓋掉。
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "datasets" / "run_001"
OUT_NPZ = OUT_DIR / "trajectory_0.npz"
PREVIEW_PNG = OUT_DIR / "rgb_0000.png"

NUM_ENVS = 1  # 並行環境數。第一批資料只要一條軌跡，開 1 個就好。
EE_BODY_NAME = "panda_hand"  # Franka 末端那塊手掌剛體的名字，用來查 xyz 與旋轉。


def attach_wrist_camera(env_cfg) -> None:
    """把一顆針孔相機掛在 panda_hand 上，當成手腕視角。

    官方 Reach 任務只有手臂與目標點，沒有相機。
    我們改的是「還沒 gym.make 之前的設定物件」，不是改 Isaac Lab 套件原始碼。
    相機參數（焦距、偏移）抄官方 visuomotor Franka 範例，讓鏡頭大致朝工作空間。
    """
    env_cfg.scene.wrist_cam = CameraCfg(
        # {ENV_REGEX_NS} 會被換成這個環境在場景樹裡的路徑前綴。
        # 掛在 panda_hand 底下，手臂一動，相機就跟著動。
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=0.0,  # 0 = 每步都更新畫面，不要跳幀
        height=200,
        width=200,
        data_types=["rgb"],  # 只要彩色圖；深度之後若要再加
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 2.0),  # 太近或太遠的東西不畫
        ),
        # pos / rot 是相機相對 panda_hand 的平移與旋轉，不是世界座標。
        # convention="ros"：旋轉用機器人學常見的座標慣例（和 USD 預設不同）。
        offset=CameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15),
            rot=(-0.70614, 0.03701, 0.03701, -0.70614),
            convention="ros",
        ),
    )


def sample_random_action(env) -> torch.Tensor:
    """在動作範圍 [-1, 1] 裡均勻亂抽一組關節指令。

    今天目標是管線能跑通，不是漂亮的揮手。亂抽手臂會抖、會撞，都正常。
    之後若要把規劃好的軌跡灌進來，只改這個函式即可。
    回傳形狀是 (環境數, 動作維度)；我們只有 1 個環境、7 個手臂關節。
    """
    unwrapped = env.unwrapped  # gym.make 會包一層；真正的 Isaac Lab 環境在裡面
    action_dim = unwrapped.action_manager.total_action_dim
    return 2.0 * torch.rand((unwrapped.num_envs, action_dim), device=unwrapped.device) - 1.0


def read_rgb(env) -> np.ndarray:
    """讀手腕相機這一幀的彩色圖，轉成 CPU 上的 NumPy 陣列。

    相機資料還在 GPU 上；存檔前要搬到 CPU。
    [0] 是第 0 個並行環境（我們只開 1 個）。
    ..., :3 只留紅綠藍；若模擬器給了透明通道（第四個）就丟掉。
    """
    rgb = env.unwrapped.scene["wrist_cam"].data.output["rgb"]
    return rgb[0, ..., :3].detach().cpu().numpy()


def read_ee_pose(env) -> np.ndarray:
    """讀末端手掌在世界座標的位姿，長度 7：xyz + 四元數 (w, x, y, z)。

    這不是關節角度，而是「手掌這塊剛體現在在空間裡的位置與朝向」。
    同一旋轉可以用 q 或 -q 兩種四元數表示；quat_unique 把符號固定成一種，
    之後比對或學習時比較不會被正負翻轉干擾。
    """
    robot = env.unwrapped.scene["robot"]
    body_ids, _ = robot.find_bodies(EE_BODY_NAME)
    pose = robot.data.body_pose_w[0, body_ids[0], :7]
    pos = pose[:3]
    quat = quat_unique(pose[3:7].unsqueeze(0)).squeeze(0)
    return torch.cat([pos, quat], dim=0).detach().cpu().numpy()


def main() -> None:
    # 從任務登記表把官方 Reach 設定抄出來，並改成 1 個環境、用你指定的裝置（通常是 GPU）。
    # use_fabric=False：相機位姿改走 USD，不走 Warp。
    # Isaac Lab 2.3.2 的 Fabric 路徑會呼叫 wp.transform_compose，
    # 你這套 Isaac Sim 5.0 內建的 Warp 沒有這個函式，reset 手腕相機時會炸。
    # 只有 1 個環境，走 USD 夠用，資料格式不變。
    env_cfg = parse_env_cfg(
        "Isaac-Reach-Franka-v0",
        device=args_cli.device,
        num_envs=NUM_ENVS,
        use_fabric=False,
    )
    attach_wrist_camera(env_cfg)
    # 環境 reset 後，相機畫面可能還停在舊的一幀。多渲幾次，第一張圖才是現在的場景。
    env_cfg.num_rerenders_on_reset = 3

    # 真正把場景、機器人、相機建出來。cfg 是我們改過的那份，不是套件內建原檔。
    env = gym.make("Isaac-Reach-Franka-v0", cfg=env_cfg)

    env.reset()
    # 上面的 num_rerenders_on_reset 管「reset 當下」；這裡再保險渲幾幀。
    for _ in range(4):
        env.unwrapped.sim.render()

    rgb_list = []
    pose_list = []
    action_list = []
    time_list = []
    step_dt = env.unwrapped.step_dt  # 每做一次 env.step，模擬往前走多少秒

    for step_idx in range(args_cli.num_steps):
        action = sample_random_action(env)
        env.step(action)  # 把指令送進模擬器：關節動、物理往前、相機更新

        # 四個 list 用同一個 step_idx 往後 append，所以第 i 筆永遠對齊。
        rgb_list.append(read_rgb(env))
        pose_list.append(read_ee_pose(env))
        action_list.append(action[0].detach().cpu().numpy())  # [0] = 唯一那個環境
        time_list.append(step_idx * step_dt)

        if step_idx == 0:
            print(f"[m5] step 0 rgb shape={rgb_list[0].shape} ee_pose={pose_list[0]}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # stack：把「100 張各別陣列」黏成一個大陣列，第 0 維是時間步。
    rgb = np.stack(rgb_list, axis=0)
    ee_pose = np.stack(pose_list, axis=0)
    actions = np.stack(action_list, axis=0)
    timestamps = np.asarray(time_list, dtype=np.float32)

    # 一個 .npz 裡可以放多個具名陣列，之後 np.load 用鍵名取出。
    np.savez_compressed(
        OUT_NPZ,
        rgb=rgb,
        ee_pose=ee_pose,
        actions=actions,
        timestamps=timestamps,
    )
    print(f"[m5] saved {OUT_NPZ}")
    print(f"[m5] rgb {rgb.shape}  ee_pose {ee_pose.shape}  actions {actions.shape}  timestamps {timestamps.shape}")

    # 另存第一幀成 PNG，用眼睛確認相機不是全黑、不是全灰。不是訓練需要的格式。
    try:
        from PIL import Image

        preview = np.clip(rgb[0], 0, 255).astype(np.uint8)
        Image.fromarray(preview).save(PREVIEW_PNG)
        print(f"[m5] preview {PREVIEW_PNG}")
    except Exception as exc:
        print(f"[m5] skip png preview ({exc})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()  # 關掉 Isaac Sim，否則行程可能掛著不退出

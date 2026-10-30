"""
m3 - 讓 Franka 末端避開障礙, 從 A 走到 B

本次目標:
  場景裡有障礙方塊, 以及兩個小標記:
    - 藍方塊 = 起點 A (起始末端位置)
    - 綠方塊 = 終點 B (固定目標位置)
  機器人要用路徑規劃自行決定關節怎麼動, 繞過障礙, 把末端從 A 帶到 B.

怎麼做到:
  1. Lula RRT: 全域路徑規劃 -> 一串 joint-space waypoints (避開障礙)
  2. RMPflow: 局部反應式控制 -> 每步把目標轉成 joint 命令並執行
  3. 記錄實際末端軌跡 -> CSV; 可選雙相機 -> GIF

執行: 請送 m3-run.py (避免直接送本檔觸發 UDP 大小限制 / WinError 10040).
"""

from __future__ import annotations

import asyncio
import csv
import os
import time
import traceback
from dataclasses import dataclass
from typing import List, Optional, Tuple

import carb
import numpy as np
import omni.kit.app
import omni.kit.commands
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects import FixedCuboid, GroundPlane, VisualCuboid
from omni.isaac.core.utils.extensions import enable_extension, get_extension_path_from_name
from omni.isaac.core.utils.numpy.rotations import rot_matrices_to_quats
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.motion_generation import ArticulationKinematicsSolver, ArticulationMotionPolicy
from omni.isaac.motion_generation import interface_config_loader
from omni.isaac.motion_generation.lula import RRT, RmpFlow
from omni.isaac.motion_generation.path_planner_visualizer import PathPlannerVisualizer
from PIL import Image
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics


# ---------------------------------------------------------------------------
# 設定（改參數優先改這裡）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CameraSpec:
    """單一相機：name 決定輸出子目錄 / GIF 檔名。"""

    name: str
    prim_path: str
    translation: Tuple[float, float, float]
    rotate_xyz_deg: Tuple[float, float, float]
    resolution: Tuple[int, int] = (640, 480)
    clipping: Tuple[float, float] = (0.02, 100.0)
    focal_length: float = 18.0


@dataclass(frozen=True)
class Config:
    project_rel: str = "Desktop/Experiment/DT_learning"
    output_dirname: str = "m3_result"
    usd_filename: str = "franka_rrt_rmp.usd"
    csv_filename: str = "m3_trajectory.csv"
    gif_frame_duration_ms: int = 80

    # 物理 / 控制
    physics_dt: float = 1.0 / 60.0
    composition_wait_frames: int = 5
    settle_frames: int = 30
    warmup_frames: int = 12
    capture_every_n: int = 3  # 運動迴圈每 N 幀截一次圖（GIF 才不會太長）

    # 起始姿態：與 Lula robot_descriptor default_q 對齊（7 臂）+ 手指微開
    start_arm: Tuple[float, ...] = (0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75)
    finger_open: float = 0.04

    # 固定目標 B（task space，公尺）；A = 起始姿態的 FK
    target_b: Tuple[float, float, float] = (0.45, 0.35, 0.35)
    # orientation 可選；None = 只約束位置（RRT / RMP 較容易收斂）
    target_b_quat_wxyz: Optional[Tuple[float, float, float, float]] = None

    # A/B 視覺標記（VisualCuboid，不參與碰撞）；A 在 FK 後才建立
    marker_scale: Tuple[float, float, float] = (0.02, 0.02, 0.02)
    marker_a_color: Tuple[float, float, float] = (0.15, 0.35, 0.95)  # 藍：起點 A
    marker_b_color: Tuple[float, float, float] = (0.1, 0.9, 0.2)  # 綠：終點 B

    # RRT
    rrt_max_iterations: int = 50000
    rrt_seed: int = 123456
    interpolate_max_cspace_dist: float = 0.08

    # 跟隨：每個 waypoint 最多跑幾幀；整體超時
    frames_per_waypoint: int = 8
    max_follow_frames: int = 2400
    # 視為到達 B 的距離（公尺）；先前 0.04 在畫面上看起來還差一截
    reach_thresh_m: float = 0.015
    # 跟完 RRT waypoints 後，再專門盯著 B（只約束位置）最多跑幾幀
    final_approach_max_frames: int = 900
    # 到達門檻後再多撐幾幀，讓畫面停在綠點附近
    settle_frames_at_b: int = 45

    # 5 個障礙：擋在 A≈(0.27,0,0.43) → B=(0.45,0.35,0.35) 的直線上
    # 形成一道牆（中高度），逼 RRT 繞上方；勿擋死 A/B 本身
    # 方塊參數: (position_xyz, scale_xyz, color_rgb)；position 是方塊中心
    obstacles: Tuple[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]], ...] = (
        ((0.36, 0.12, 0.30), (0.10, 0.08, 0.36), (0.85, 0.25, 0.20)),
        ((0.34, 0.20, 0.30), (0.09, 0.08, 0.36), (0.85, 0.35, 0.15)),
        ((0.38, 0.26, 0.28), (0.09, 0.08, 0.34), (0.80, 0.20, 0.30)),
        ((0.33, 0.06, 0.28), (0.09, 0.08, 0.34), (0.75, 0.30, 0.20)),
        ((0.40, 0.18, 0.26), (0.08, 0.08, 0.32), (0.90, 0.40, 0.20)),
    )

    light_intensity: float = 1500.0

    # False：暫時關掉相機截圖 / GIF，方便在 viewport 直接看運動
    # True：恢復 Step 9/10 截圖 + Step 11 GIF
    enable_camera_capture: bool = False

    # False：匯入 omni.importer.urdf 的 panda_arm_hand.urdf（有 mesh 可看）
    # True：改走 get_assets_root_path_async → Nucleus franka.usd
    # 注意：不要用 lula_franka_gen.urdf 當場景模型——它的 package:// mesh 解析不到，畫面會只剩方塊
    # RRT/RMPflow 仍透過 load_supported_* 讀自己的 lula urdf（運動學用），與場景關節名相容即可
    prefer_nucleus_franka: bool = False

    # UI viewport（/OmniverseKit_Persp）：對準 A→B 障礙牆與目標（略拉遠）
    viewport_eye: Tuple[float, float, float] = (1.15, -0.95, 0.75)
    viewport_target: Tuple[float, float, float] = (0.36, 0.18, 0.35)

    # 錄影相機路徑會在匯入機器人後依實際 base 補上；這裡先放 placeholder
    cameras: Tuple[CameraSpec, ...] = (
        CameraSpec(
            name="hand",
            prim_path="/PLACEHOLDER/panda_hand/Camera",
            translation=(0.0, 0.0, 0.12),
            rotate_xyz_deg=(180.0, 0.0, 90.0),
        ),
        CameraSpec(
            name="side",
            prim_path="/World/SideCamera",
            translation=(1.0, -1.3, 1.0),
            rotate_xyz_deg=(60.0, 0.0, 40.0),
        ),
    )


CFG = Config()


# ---------------------------------------------------------------------------
# 少數可重用工具
# ---------------------------------------------------------------------------
async def wait_frames(n: int = 1) -> None:
    """等待 Kit 真正跑完 n 幀。"""
    for _ in range(n):
        await omni.kit.app.get_app().next_update_async()


def rgba_to_uint8(rgba) -> Optional[np.ndarray]:
    """annotator 可能回傳 None / 空陣列 / float；統一成 uint8。"""
    if rgba is None:
        return None
    if hasattr(rgba, "cpu"):
        rgba = rgba.cpu().numpy()
    rgba = np.asarray(rgba)
    if rgba.size == 0:
        return None
    if rgba.dtype != np.uint8:
        if float(np.max(rgba)) <= 1.0:
            rgba = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)
        else:
            rgba = np.clip(rgba, 0, 255).astype(np.uint8)
    return rgba


def resolve_output_dir(cfg: Config = CFG) -> str:
    """
    決定 CSV / GIF / USD 要存到哪個資料夾（預設 m3_result）。

    為什麼不能寫死、也不能盲目相信 __file__？
      本腳本常經 m3-run.py → Send to Isaac 以「字串」在 Kit 裡執行。
      此時：
        - __file__ 可能指到 omni.isaac.vscode 擴充套件目錄（不是專案）
        - cwd 常是 Isaac 安裝目錄（也不是 DT_learning）
      若照錯誤路徑存檔，結果會落到奇怪地方或寫失敗。

    尋找順序（找到就回傳）：
      1) __file__ 真的是本檔 m3-obstacle+Lula_RRT.py，
         且路徑不在 omni.isaac.vscode 底下
         → 用「本檔所在目錄 / m3_result」
      2) 掃 C~Z 碟，找 Desktop/Experiment/DT_learning 是否存在
         → 用「該專案目錄 / m3_result」
      3) 以上都失敗 → 退回「目前 cwd / m3_result」

    Returns:
        輸出目錄的絕對或可寫路徑字串（呼叫端會 os.makedirs）。
    """
    # --- 候選 1：腳本旁（僅在 __file__ 可信時）---
    this_file = globals().get("__file__")
    if isinstance(this_file, str) and this_file.replace("\\", "/").endswith("m3-obstacle+Lula_RRT.py"):
        candidate = os.path.join(os.path.dirname(os.path.abspath(this_file)), cfg.output_dirname)
        # Send to Isaac 時 __file__ 常落在擴充套件目錄，必須排除
        if "omni.isaac.vscode" not in candidate.replace("\\", "/"):
            return candidate

    # --- 候選 2：依專案相對路徑掃各磁碟（Windows）---
    for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        root = f"{letter}:/{cfg.project_rel}"
        if os.path.isdir(root):
            return os.path.join(root, cfg.output_dirname)

    # --- 候選 3：最後手段 ---
    return os.path.join(os.getcwd(), cfg.output_dirname)


def save_gif(frames: List[Image.Image], gif_path: str, duration_ms: int) -> Optional[str]:
    """把 PIL RGB 幀列表存成 GIF。"""
    if not frames:
        return None
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return gif_path


def find_hand_path(stage) -> str:
    """依實際匯入結果找 panda_hand（Nucleus USD → /panda；Lula URDF → /franka）。"""
    for path in ("/panda/panda_hand", "/franka/panda_hand"):
        if stage.GetPrimAtPath(path).IsValid():
            return path
    raise RuntimeError("Cannot find panda_hand under /panda or /franka")


def build_full_joint_target(robot: Articulation, arm7: np.ndarray, finger: float) -> np.ndarray:
    """把 7 軸手臂目標寫進完整 DOF 向量（含手指）。"""
    q = robot.get_joint_positions()
    if q is None:
        q = np.zeros(robot.num_dof, dtype=np.float32)
    else:
        q = np.array(q, dtype=np.float32)
    n_arm = min(7, robot.num_dof)
    q[:n_arm] = np.asarray(arm7[:n_arm], dtype=np.float32)
    if robot.num_dof >= 9:
        q[7] = finger
        q[8] = finger
    return q


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
async def main() -> None:
    """
    全流程說明（m3：障礙物 + Lula RRT + RMPflow + CSV / GIF）
    ================================================================

    【怎麼跑】
      請送 m3-run.py。不要直接送本檔（UDP 容易 10040）。

    【總目標】
      1. 載入 Franka 場景模型（預設 panda_arm_hand.urdf 有視覺；RRT/RMP 仍用官方 Lula config）
      2. 擺 5 個 FixedCuboid 擋在 A→B 直線上（逼 RRT 繞行）
      3. RRT 規劃 A→B（B 固定座標），RMPflow 跟隨
      4. 輸出 CSV + USD；相機截圖/GIF 由 Config.enable_camera_capture 控制（預設關）

    【RRT / RMPflow 是什麼】
      - 不是 Franka 硬體或 USD 內建；是 omni.isaac.motion_generation（Lula）的演算法
      - Isaac 對 Franka 有官方 config；load_supported_*("Franka", ...) 載入的是規劃器/控制器參數
        （yaml、urdf、descriptor 路徑等），不是機器人起始關節姿態
      - 起始姿態由 Step 7 的 start_arm / set_joint_positions 決定

    【輸出】
      m3_result/
        m3_trajectory.csv
        hand/frame_XXX.png , side/frame_XXX.png
        hand.gif , side.gif
        franka_rrt_rmp.usd

    Step 1  輸出目錄
    Step 2  空白舞台
    Step 3  燈 + 地面
    Step 4  載入 Franka（官方 Lula 資產）
    Step 5  5 障礙方塊 + 目標標記
    Step 6  啟動物理 + initialize
    Step 7  設起始姿態 → A = FK；B = 固定點
    Step 8  建立 RRT + RMPflow，註冊障礙，compute_path
    Step 9  雙相機 + 暖機
    Step 10 用 RMPflow 跟隨插值路徑，同步記錄 CSV / 截圖
    Step 11 GIF + USD
    """

    cfg = CFG
    try:
        # ==================================================================
        # Step 1：準備輸出目錄
        # ==================================================================
        carb.log_warn("=== [Step 1] Entered main() ===")
        output_dir = resolve_output_dir(cfg)
        os.makedirs(output_dir, exist_ok=True)
        carb.log_warn(f"=== [Step 1] Output dir: {output_dir} ===")

        # ==================================================================
        # Step 2：空白舞台
        # ==================================================================
        carb.log_warn("=== [Step 2] Creating new stage ===")
        await omni.usd.get_context().new_stage_async()
        stage = omni.usd.get_context().get_stage()

        # ==================================================================
        # Step 3：燈 + 地面（GroundPlane 物件後面要給 RRT/RMP add_obstacle）
        # ==================================================================
        carb.log_warn("=== [Step 3] Building scene base ===")

        # 明確建立物理場景：改用 reference 匯入機器人後，機器人 USD 內的
        # /physicsScene 不會被帶進主場景，這裡自建一個，timeline.play() 才會模擬。
        physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

        light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
        light.CreateIntensityAttr(cfg.light_intensity)
        light.AddTransformOp().Set(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), -45)))

        ground = GroundPlane(
            prim_path="/World/GroundPlane",
            name="ground_plane",
            size=10.0,
            color=np.array([0.45, 0.45, 0.45]),
        )

        # ==================================================================
        # Step 4：載入 Franka 場景模型（要看得見手臂）
        # -----------------------------------------------------------------
        # 場景模型 ≠ Lula 內部運動學模型：
        #   - 場景：panda_arm_hand.urdf（相對路徑 mesh，匯入後看得到手臂；同 m2）
        #   - 規劃：load_supported_* 仍用官方 lula_franka_gen.urdf + yaml
        # 不要拿 lula_franka_gen.urdf 當場景：mesh 是 package://franka_description/...
        #   ROS_PACKAGE_PATH 未設會全部解析失敗 → 只剩少數方塊/空 link 在動
        # ==================================================================
        carb.log_warn("=== [Step 4] Loading Franka for Lula ===")
        articulation_root = None
        load_mode = None

        if cfg.prefer_nucleus_franka:
            # 必須用 async；同步 get_assets_root_path() 會卡 UI
            from omni.isaac.core.utils.stage import add_reference_to_stage
            from omni.isaac.nucleus import get_assets_root_path_async

            carb.log_warn("=== [Step 4] prefer_nucleus=True, resolving assets async ===")
            assets_root = await get_assets_root_path_async()
            if not assets_root:
                raise RuntimeError(
                    "Nucleus assets root not found. "
                    "Set Config.prefer_nucleus_franka=False to use panda_arm_hand.urdf."
                )
            usd_path = f"{assets_root}/Isaac/Robots/Franka/franka.usd"
            add_reference_to_stage(usd_path=usd_path, prim_path="/panda")
            articulation_root = "/panda"
            load_mode = f"nucleus:{usd_path}"
            await wait_frames(cfg.composition_wait_frames)
        else:
            # URDF importer 擴充套件在不同 Isaac 版本名稱不同：
            #   Isaac Sim 4.5+ : isaacsim.asset.importer.urdf
            #   舊版           : omni.importer.urdf
            # 依序嘗試啟用，取第一個成功者的實際套件路徑（get_extension_path_from_name
            # 找不到會回傳空字串，導致後面拼出相對路徑而抓不到 URDF）。
            ext_mgr = omni.kit.app.get_app().get_extension_manager()
            urdf_ext_name = None
            for candidate_ext in ("isaacsim.asset.importer.urdf", "omni.importer.urdf"):
                if not ext_mgr.is_extension_enabled(candidate_ext):
                    try:
                        enable_extension(candidate_ext)
                        await wait_frames(1)
                    except Exception:
                        continue
                if ext_mgr.is_extension_enabled(candidate_ext):
                    urdf_ext_name = candidate_ext
                    break
            if urdf_ext_name is None:
                raise RuntimeError(
                    "URDF importer extension not found "
                    "(tried isaacsim.asset.importer.urdf / omni.importer.urdf)"
                )

            # 與 m2 相同：內建 panda_arm_hand.urdf（mesh 可解析）
            urdf_path = os.path.join(
                get_extension_path_from_name(urdf_ext_name),
                "data",
                "urdf",
                "robots",
                "franka_description",
                "robots",
                "panda_arm_hand.urdf",
            )
            if not os.path.exists(urdf_path):
                raise FileNotFoundError(f"Franka URDF not found: {urdf_path}")

            # 4.5 的 URDF importer 會把 .dae 視覺網格逐一轉檔（同步、慢），
            # 且過程噴大量 log，容易灌爆 VS Code UDP（WinError 10040）而「看似卡住」。
            # 對策：只匯入成 USD 檔一次，之後偵測到快取檔就直接 reference，秒開又不洗版。
            from omni.isaac.core.utils.stage import add_reference_to_stage

            cached_usd = os.path.join(output_dir, "franka_imported.usd")
            if not os.path.exists(cached_usd):
                carb.log_warn(
                    f"=== [Step 4] First import → {cached_usd} "
                    f"(converting .dae meshes, may take a while) ==="
                )
                _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
                import_config.merge_fixed_joints = False
                import_config.convex_decomp = False
                import_config.import_inertia_tensor = True
                import_config.fix_base = True
                import_config.make_default_prim = True  # reference 需要 defaultPrim
                import_config.create_physics_scene = True

                omni.kit.commands.execute(
                    "URDFParseAndImportFile",
                    urdf_path=urdf_path,
                    import_config=import_config,
                    dest_path=cached_usd,  # 寫成獨立 USD，而非直接 in-memory
                )
                await wait_frames(2)
                carb.log_warn(f"=== [Step 4] Import done, cached at {cached_usd} ===")
            else:
                carb.log_warn(f"=== [Step 4] Reusing cached Franka USD: {cached_usd} ===")

            # 把匯入好的機器人 reference 進當前場景
            add_reference_to_stage(usd_path=cached_usd, prim_path="/panda")
            articulation_root = "/panda"
            load_mode = f"urdf_cached:{cached_usd}"
            await wait_frames(cfg.composition_wait_frames)

        hand_path = find_hand_path(stage)
        carb.log_warn(
            f"=== [Step 4] mode={load_mode}, root={articulation_root}, hand={hand_path} ==="
        )

        # ==================================================================
        # Step 5：5 障礙 + B 點視覺標記
        # ==================================================================
        carb.log_warn("=== [Step 5] Spawning obstacles ===")
        obstacles = []
        for i, (pos, scale, color) in enumerate(cfg.obstacles):
            cube = FixedCuboid(
                prim_path=f"/World/Obstacle_{i}",
                name=f"obstacle_{i}",
                position=np.array(pos, dtype=np.float64),
                scale=np.array(scale, dtype=np.float64),
                size=1.0,
                color=np.array(color, dtype=np.float64),
            )
            obstacles.append(cube)
            carb.log_warn(f"=== [Step 5] Obstacle_{i} pos={pos} scale={scale} ===")

        # 綠色小方塊：終點 B 標記（僅視覺，不是要夾的物體）
        VisualCuboid(
            prim_path="/World/TargetB",
            name="target_b",
            position=np.array(cfg.target_b, dtype=np.float64),
            scale=np.array(cfg.marker_scale, dtype=np.float64),
            size=1.0,
            color=np.array(cfg.marker_b_color, dtype=np.float64),
        )

        # UI 透視相機拉近對準工作區（手臂 + 障礙 + B），避免地面佔滿畫面
        set_camera_view(
            eye=np.array(cfg.viewport_eye, dtype=np.float64),
            target=np.array(cfg.viewport_target, dtype=np.float64),
        )
        carb.log_warn(
            f"=== [Step 5] viewport eye={cfg.viewport_eye} target={cfg.viewport_target} ==="
        )

        # ==================================================================
        # Step 6：啟動物理 + initialize Articulation
        # ==================================================================
        carb.log_warn("=== [Step 6] Initializing Articulation ===")
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        timeline.play()
        await wait_frames(1)

        robot = Articulation(prim_path=articulation_root)
        robot.initialize()
        if not robot.handles_initialized:
            raise RuntimeError(f"Articulation failed to initialize at {articulation_root}")
        carb.log_warn(f"=== [Step 6] DOF={robot.num_dof}, names={robot.dof_names} ===")

        # ==================================================================
        # Step 7：設起始姿態，讀 A（FK）；B 為固定座標
        # ==================================================================
        carb.log_warn("=== [Step 7] Set start pose / read A ===")
        
        # 設定起始姿態，讀取起始姿態的 FK
        q0 = build_full_joint_target(
            robot, np.array(cfg.start_arm, dtype=np.float32), cfg.finger_open
        )
        
        # 設定起始姿態
        robot.set_joint_positions(q0)
        
        # 設定起始姿態的關節速度為 0
        robot.set_joint_velocities(np.zeros(robot.num_dof, dtype=np.float32))
        
        # 等待 settle_frames 幀，讓機器人穩定
        for _ in range(cfg.settle_frames):
            timeline.forward_one_frame()
            await wait_frames(1)

        # ==================================================================
        # Step 8：建立 RRT、RMPflow，註冊障礙，執行路徑規劃
        # -----------------------------------------------------------------
        # 兩層分工：
        #   RRT(Rapidly-exploring Random Tree, 快速探索隨機樹) -> path planner 尋找全域路徑
        #   -> 找出「從現在關節角到能到達 B」的可行序列。
        #     - 全域：全域路徑規劃，整段流程通常只呼叫一次
        #     - 在 c-space（configuration space / 關節空間）搜尋
        #     - 輸出：一串 joint-space waypoints（關節角序列）
        #     - 不是 task-space 的 (x,y,z) 點列
        #     - 計算較重；找不到路徑時回傳 None 或空序列
        
        #   RMPflow(Reactive Motion Policy Flow, 反應式運動策略流) -> motion policy 追蹤局部路徑
        #   -> 找出「從現在關節角到能到達 B」的可行序列。
        #     - 類型：局部反應式 motion policy
        #     - 每個物理步根據當前 end-effector target 產出 joint command
        #     - 不做長距離全域搜尋；全域路徑由 RRT 提供
        #
        #   1. RRT 先給全域路線（關節空間的點列）
        #   2. RMPflow 負責執行：把當前該追的點當成 end-effector target，逐步驅動機器人 -> 把路徑變成機器人的關節角序列(考慮危險的地方調整速度等, 避免碰撞)
        #   3. 執行時若有障礙／誤差，RMPflow 會在當步命令裡做反應式調整（平滑、避障）
        
        # Isaac Sim 4.0 呼叫順序：
        #   1) interface_config_loader 載入官方的 RRT / RMPflow config dict
        #   2) 建構 RRT / RmpFlow
        #   3) RRT 與 RmpFlow 都呼叫 add_obstacle
        #   4) set_end_effector_target(B) → update_world()
        #   5) compute_path(active_q, watched_q) → joint waypoints
        #   6) interpolate_path 加密後，交給 Step 10 用 RMPflow 追蹤
        # ==================================================================
        carb.log_warn("=== [Step 8] Build RRT + RMPflow ===")

        # --- 8a) 載入官方 Franka config ---
        # 從 omni.isaac.motion_generation 讀 path_planner_configs /
        # motion_policy_configs，回傳可直接 **kwargs 傳入建構子的 dict。
        # dict 內通常含 robot_description_path、urdf_path、rrt_config_path 等。
        rrt_cfg = interface_config_loader.load_supported_path_planner_config("Franka", "RRT")
        rmp_cfg = interface_config_loader.load_supported_motion_policy_config("Franka", "RMPflow")
        if rrt_cfg is None or rmp_cfg is None:
            raise RuntimeError("Failed to load Franka RRT / RMPflow config")

        # --- 8b) 建立 RRT 實例並覆寫少數參數 ---
        # 這裡尚未開始搜尋路徑；只是用 rrt_cfg 建構規劃器物件。
        # set_max_iterations / set_random_seed：覆寫 config 裡對應欄位。
        # 真正執行 RRT 搜尋的是後面的 rrt.compute_path(...)。
        rrt = RRT(**rrt_cfg)
        rrt.set_max_iterations(cfg.rrt_max_iterations)
        rrt.set_random_seed(cfg.rrt_seed)

        # --- 8c) 建構 RMPflow 與相關介面物件 ---
        # RmpFlow：輸入 task-space target，輸出該步的 joint command。
        # ArticulationMotionPolicy：把 RmpFlow 輸出對應到本機 Articulation 的 DOF 順序。
        # kinematics：Lula kinematics solver；用於 joint waypoint → end-effector pose（FK）。
        # ArticulationKinematicsSolver：依目前實際關節角計算 end-effector pose
        #   （讀取起點 A、後續記錄軌跡都用它）。
        # PathPlannerVisualizer：取得 active/watched joint subset，並提供路徑插值。
        rmp = RmpFlow(**rmp_cfg)
        artic_rmp = ArticulationMotionPolicy(robot, rmp, cfg.physics_dt)
        kinematics = rmp.get_kinematics_solver()
        artic_kin = ArticulationKinematicsSolver(robot, kinematics, rmp.end_effector_frame_name)
        planner_viz = PathPlannerVisualizer(robot, rrt)

        # --- 8d) 註冊障礙物 ---
        # RRT 與 RmpFlow 都必須註冊同一組障礙，否則兩邊碰撞模型不一致。
        # static=True：標記為靜態障礙（位姿不變）；RRT 可對此做較有效的內部處理。
        rrt.add_obstacle(ground, static=True)
        rmp.add_obstacle(ground, static=True)
        for cube in obstacles:
            rrt.add_obstacle(cube, static=True)
            rmp.add_obstacle(cube, static=True)

        # --- 8e) 設定起點 A 與終點 B ---
        # A：目前 end-effector 的世界座標位置，由 FK 依實際關節角算出。
        # B：Config 中的固定目標位置（單位：公尺）。
        # target_orient：
        #   None → 只約束位置，不約束 orientation（規劃較容易成功）
        #   quat (w,x,y,z) → 同時約束末端朝向（搜尋較難）
        ee_pos_a, ee_rot_a = artic_kin.compute_end_effector_pose()
        target_b = np.array(cfg.target_b, dtype=np.float64)
        target_orient = None
        if cfg.target_b_quat_wxyz is not None:
            target_orient = np.array(cfg.target_b_quat_wxyz, dtype=np.float64)

        # 藍色小方塊：起點 A 標記（位置 = 目前末端 FK，僅視覺）
        VisualCuboid(
            prim_path="/World/TargetA",
            name="target_a",
            position=np.array(ee_pos_a, dtype=np.float64),
            scale=np.array(cfg.marker_scale, dtype=np.float64),
            size=1.0,
            color=np.array(cfg.marker_a_color, dtype=np.float64),
        )
        carb.log_warn(
            f"=== [Step 8] A={np.round(ee_pos_a, 4).tolist()} "
            f"B={target_b.tolist()} ee_frame={rmp.end_effector_frame_name} "
            f"markers scale={cfg.marker_scale} ==="
        )

        # --- 8f) 設定 RRT 終點，並更新碰撞世界 ---
        # set_end_effector_target：終點以 task-space（位置 ± 可選 orientation）指定。
        # update_world：把已註冊障礙的當前位姿寫入 Lula 內部碰撞模型。
        # 若省略 update_world，規劃時可能未使用最新障礙狀態。
        rrt.set_end_effector_target(target_b, target_orient)
        rrt.update_world()

        # --- 8g) 讀取規劃用的關節角 ---
        # active_q：RRT 會規劃的關節（Franka 通常為 7 軸手臂）；作為路徑起點。
        # watched_q：不納入規劃變量、但仍參與碰撞檢查的關節（例如手指）。
        # 關節順序須符合 Lula robot descriptor，因此透過 PathPlannerVisualizer 的 subset 讀取。
        active_q = planner_viz.get_active_joints_subset().get_joint_positions()
        watched_q = planner_viz.get_watched_joints_subset().get_joint_positions()

        # --- 8h) 執行 RRT（本段關鍵呼叫；可能耗時數秒）---
        # compute_path 回傳值：
        #   - np.ndarray，形狀約 (N, num_active_dof)：N 個稀疏 joint waypoints
        #   - None 或 []：在 max_iterations 內找不到可行路徑
        # API 說明：終點用 task-space 指定，路徑結果是 c-space；
        # 沒有 plan_task_space_path() 這類回傳 (x,y,z) 序列的函式。
        t_plan0 = time.perf_counter()
        plan = rrt.compute_path(active_q, watched_q)
        t_plan = time.perf_counter() - t_plan0

        if plan is None or len(plan) == 0:
            raise RuntimeError(
                f"RRT failed to find a path A→B in {t_plan:.3f}s. "
                "Try moving obstacles or target_b."
            )

        # --- 8i) 將稀疏路徑插值成密集路徑 ---
        # RRT 相鄰 waypoint 的關節角差可能很大，直接交給 RMPflow 容易追蹤不穩。
        # interpolate_path：在相鄰點之間線性插值，
        # 使相鄰點各維度最大差值 ≤ interpolate_max_cspace_dist。
        # dense_plan 仍是 joint-space；Step 10 會對各點做 FK，再設為 RMPflow target。
        plan = np.asarray(plan, dtype=np.float64)
        dense_plan = planner_viz.interpolate_path(plan, cfg.interpolate_max_cspace_dist)
        carb.log_warn(
            f"=== [Step 8] RRT OK in {t_plan:.3f}s: "
            f"sparse={len(plan)} → dense={len(dense_plan)} waypoints ==="
        )

        # ==================================================================
        # Step 9：雙相機 + Replicator 暖機（enable_camera_capture=False 時整段跳過）
        # ==================================================================
        cameras = []
        if cfg.enable_camera_capture:
            carb.log_warn("=== [Step 9] Setting up cameras ===")
            carb.settings.get_settings().set("/rtx/rendermode", "RayTracedLighting")

            camera_specs = [
                CameraSpec(
                    name="hand",
                    prim_path=f"{hand_path}/Camera",
                    translation=cfg.cameras[0].translation,
                    rotate_xyz_deg=cfg.cameras[0].rotate_xyz_deg,
                ),
                cfg.cameras[1],
            ]

            for spec in camera_specs:
                cam = UsdGeom.Camera.Define(stage, spec.prim_path)
                xform = UsdGeom.Xformable(cam.GetPrim())
                xform.ClearXformOpOrder()
                xform.AddTranslateOp().Set(Gf.Vec3f(*spec.translation))
                xform.AddRotateXYZOp().Set(Gf.Vec3f(*spec.rotate_xyz_deg))
                cam.GetClippingRangeAttr().Set(Gf.Vec2f(*spec.clipping))
                cam.GetFocalLengthAttr().Set(spec.focal_length)

                rp = rep.create.render_product(spec.prim_path, spec.resolution)
                annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                annotator.attach([rp])

                png_dir = os.path.join(output_dir, spec.name)
                os.makedirs(png_dir, exist_ok=True)
                cameras.append(
                    {"name": spec.name, "annotator": annotator, "png_dir": png_dir, "frames": []}
                )
                carb.log_warn(f"=== [Step 9] camera {spec.name} @ {spec.prim_path} ===")

            for cam in cameras:
                ready = False
                for i in range(cfg.warmup_frames):
                    await rep.orchestrator.step_async()
                    await wait_frames(1)
                    rgba = rgba_to_uint8(cam["annotator"].get_data())
                    if rgba is not None and rgba.max() > 0:
                        ready = True
                        break
                carb.log_warn(
                    f"=== [Step 9] {cam['name']} {'ready' if ready else 'still dark'} ==="
                )
        else:
            carb.log_warn("=== [Step 9] skipped (enable_camera_capture=False) ===")

        # ==================================================================
        # Step 10：RMPflow 跟隨 dense waypoints + 記錄軌跡（可選截圖）
        # -----------------------------------------------------------------
        # 對每個 c-space waypoint：
        #   FK → task-space target → rmp.set_end_effector_target
        #   跑數幀：update_world → get_next_articulation_action → apply
        # 最後再盯著 B 跑到收斂或超時。
        # CSV 記的是「實際執行」的末端位姿，不是 RRT 原始節點。
        # ==================================================================
        carb.log_warn("=== [Step 10] Follow plan with RMPflow ===")
        traj_rows = []
        step_idx = 0
        t0 = time.perf_counter()
        capture_i = 0
        rmp_times: List[float] = []

        async def _physics_and_maybe_capture() -> None:
            nonlocal step_idx, capture_i
            timeline.forward_one_frame()
            await wait_frames(1)

            ee_pos, ee_rot = artic_kin.compute_end_effector_pose()
            ee_quat = rot_matrices_to_quats(ee_rot)
            now = time.perf_counter() - t0
            traj_rows.append(
                {
                    "step": step_idx,
                    "time": now,
                    "x": float(ee_pos[0]),
                    "y": float(ee_pos[1]),
                    "z": float(ee_pos[2]),
                    "qw": float(ee_quat[0]),
                    "qx": float(ee_quat[1]),
                    "qy": float(ee_quat[2]),
                    "qz": float(ee_quat[3]),
                }
            )
            step_idx += 1

            # 截圖暫時關閉：只推進物理，方便在 Isaac viewport 直接看
            if not cfg.enable_camera_capture:
                return
            if step_idx % cfg.capture_every_n == 0:
                await rep.orchestrator.step_async()
                await wait_frames(1)
                for cam in cameras:
                    rgba = rgba_to_uint8(cam["annotator"].get_data())
                    if rgba is None:
                        continue
                    img = Image.fromarray(rgba[:, :, :3])
                    img.save(os.path.join(cam["png_dir"], f"frame_{capture_i:03d}.png"))
                    cam["frames"].append(img)
                capture_i += 1

        # --- 跟隨插值路徑 ---
        for wp_i, cspace_pt in enumerate(dense_plan):
            if step_idx >= cfg.max_follow_frames:
                break
            ts_pos, ts_rot = kinematics.compute_forward_kinematics(
                rmp.end_effector_frame_name, cspace_pt
            )
            ts_quat = rot_matrices_to_quats(ts_rot)

            for _ in range(cfg.frames_per_waypoint):
                if step_idx >= cfg.max_follow_frames:
                    break
                t_rmp0 = time.perf_counter()
                rmp.set_end_effector_target(ts_pos, ts_quat)
                rmp.update_world()
                action = artic_rmp.get_next_articulation_action()
                rmp_times.append(time.perf_counter() - t_rmp0)
                robot.get_articulation_controller().apply_action(action)
                await _physics_and_maybe_capture()

            if wp_i == 0 or (wp_i + 1) % 10 == 0 or wp_i + 1 == len(dense_plan):
                cur_ee, _ = artic_kin.compute_end_effector_pose()
                dist_b = float(np.linalg.norm(cur_ee - target_b))
                carb.log_warn(
                    f"=== follow wp {wp_i+1}/{len(dense_plan)} "
                    f"ee={np.round(cur_ee, 3).tolist()} distB={dist_b:.3f} ==="
                )

        # --- 最後專門逼近真正的 B（只約束位置，避免 orientation 把末端拉開）---
        # RRT 最後一個 waypoint 的 FK 不一定等於 B；這裡用 RMPflow 盯著綠點收斂
        approach_i = 0
        reached = False
        while (
            step_idx < cfg.max_follow_frames
            and approach_i < cfg.final_approach_max_frames
        ):
            cur_ee, _ = artic_kin.compute_end_effector_pose()
            dist_b = float(np.linalg.norm(cur_ee - target_b))
            if dist_b < cfg.reach_thresh_m:
                reached = True
                carb.log_warn(
                    f"=== [Step 10] Reached B at step={step_idx}, "
                    f"dist={dist_b:.4f}m, ee={np.round(cur_ee, 4).tolist()} ==="
                )
                break

            t_rmp0 = time.perf_counter()
            # 強制 position-only：比帶 orientation 更容易貼到綠點
            rmp.set_end_effector_target(target_b, None)
            rmp.update_world()
            action = artic_rmp.get_next_articulation_action()
            rmp_times.append(time.perf_counter() - t_rmp0)
            robot.get_articulation_controller().apply_action(action)
            await _physics_and_maybe_capture()
            approach_i += 1

            if approach_i == 1 or approach_i % 60 == 0:
                carb.log_warn(
                    f"=== final approach {approach_i}: distB={dist_b:.4f}m ==="
                )

        # 到達後再多跑幾幀，畫面停穩在 B 附近
        for _ in range(cfg.settle_frames_at_b):
            if step_idx >= cfg.max_follow_frames:
                break
            rmp.set_end_effector_target(target_b, None)
            rmp.update_world()
            action = artic_rmp.get_next_articulation_action()
            robot.get_articulation_controller().apply_action(action)
            await _physics_and_maybe_capture()

        final_ee, _ = artic_kin.compute_end_effector_pose()
        final_dist = float(np.linalg.norm(final_ee - target_b))
        avg_rmp_ms = (float(np.mean(rmp_times)) * 1000.0) if rmp_times else 0.0
        carb.log_warn(
            f"=== [Step 10] Done: reached={reached}, steps={step_idx}, "
            f"final_distB={final_dist:.4f}m (thresh={cfg.reach_thresh_m}), "
            f"RRT={t_plan:.3f}s, avg_RMPflow={avg_rmp_ms:.2f}ms/step ==="
        )
        # 視覺上：綠點對齊的是 Lula ee_frame（right_gripper），不是夾爪外觀正中心
        # 若 final_distB 已很小但畫面仍略偏，多半是 frame 與手指 mesh 的幾何差

        # --- 寫 CSV ---
        csv_path = os.path.join(output_dir, cfg.csv_filename)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["step", "time", "x", "y", "z", "qw", "qx", "qy", "qz"]
            )
            writer.writeheader()
            writer.writerows(traj_rows)
        carb.log_warn(f"=== [CSV] {csv_path} ({len(traj_rows)} rows) ===")

        timeline.pause()

        # ==================================================================
        # Step 11：GIF + USD（GIF 僅在啟用截圖時）
        # ==================================================================
        carb.log_warn("=== [Step 11] GIF + USD ===")
        if cfg.enable_camera_capture:
            for cam in cameras:
                gif_path = os.path.join(output_dir, f"{cam['name']}.gif")
                saved = save_gif(cam["frames"], gif_path, cfg.gif_frame_duration_ms)
                if saved:
                    carb.log_warn(f"=== [GIF] {saved} ({len(cam['frames'])} frames) ===")
                else:
                    carb.log_warn(f"=== [GIF] skip {cam['name']} (no frames) ===")
        else:
            carb.log_warn("=== [Step 11] GIF skipped (enable_camera_capture=False) ===")

        usd_out = os.path.join(output_dir, cfg.usd_filename)
        omni.usd.get_context().save_as_stage(usd_out)
        carb.log_warn(f"=== [Done] CSV + USD → {output_dir} ===")

    except Exception as e:
        tb = traceback.format_exc()
        carb.log_error(f"=== Critical error: {e} ===")
        carb.log_error(tb[:2000] + ("...(truncated)" if len(tb) > 2000 else ""))


carb.log_warn("=== [Entry] Script received, scheduling async task ===")
asyncio.ensure_future(main())

"""
m2 — 從「單一剛體」升級到「Articulation 機器人」

1. Articulation：多連桿 + 關節 + 慣性的群組；模擬器會展開成多個剛體用 joint 串起來。
2. URDF：機器人 XML；omni.importer.urdf 轉成 USD Articulation。
3. 擷取：Script Editor 下用 Replicator annotator（m1 同款），比 Camera.get_rgba() 穩定。

執行：請送 m2-run.py（避免直接送本檔觸發 UDP 大小限制 / WinError 10040）。
"""

from __future__ import annotations

import asyncio
import os
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
from omni.isaac.core.utils.extensions import enable_extension, get_extension_path_from_name
from PIL import Image
from pxr import Gf, PhysicsSchemaTools, UsdGeom, UsdLux, UsdPhysics


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
    output_dirname: str = "m2_result"
    usd_filename: str = "franka_panda_camera.usd"
    gif_frame_duration_ms: int = 100

    # articulation_root 通常是 /panda/root_joint；相機要掛 /panda/panda_hand
    hand_path: str = "/panda/panda_hand"

    # hand：夾爪相機（Rx=180 對準 +Z；Rz=90 讓地面傾向畫面下方）
    # side：世界座標固定側拍
    cameras: Tuple[CameraSpec, ...] = (
        CameraSpec(
            name="hand",
            prim_path="/panda/panda_hand/Camera",
            translation=(0.0, 0.0, 0.12),
            rotate_xyz_deg=(180.0, 0.0, 90.0),
        ),
        CameraSpec(
            name="side",
            prim_path="/World/SideCamera",
            translation=(0.8, -1.2, 0.9),
            rotate_xyz_deg=(65.0, 0.0, 35.0),
        ),
    )

    num_steps: int = 30
    warmup_frames: int = 15
    composition_wait_frames: int = 5

    # 官方 import_franka 伸展姿態（弧度）+ 手指微開
    target_arm: Tuple[float, ...] = (0.012, -0.57, 0.0, -2.81, 0.0, 3.037, 0.741)
    finger_open: float = 0.04

    light_intensity: float = 1500.0
    ref_cube_size: float = 0.2
    ref_cube_pos: Tuple[float, float, float] = (0.6, 0.0, 0.1)


CFG = Config()


# ---------------------------------------------------------------------------
# 少數可重用工具（有踩坑 / 多處重複才抽出）
# ---------------------------------------------------------------------------
async def wait_frames(n: int = 1) -> None:
    """等待 Kit 真正跑完 n 幀（forward_one_frame 只是下指令）。"""
    for _ in range(n):
        await omni.kit.app.get_app().next_update_async()  # next_update_async 是 Kit 的非同步函數，用於等待 Kit 真正跑完 n 幀


def rgba_to_uint8(rgba) -> Optional[np.ndarray]:
    """annotator 可能回傳 None / 空陣列 / float；統一成 uint8，避免 .max() 炸掉。"""
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
    Send to Isaac 時 __file__ 可能落在 vscode 擴充套件目錄，
    所以優先用腳本旁路徑，否則掃各磁碟 Desktop/Experiment/DT_learning。
    """
    this_file = globals().get("__file__")
    if isinstance(this_file, str) and this_file.replace("\\", "/").endswith("m2-urdf.py"):
        candidate = os.path.join(os.path.dirname(os.path.abspath(this_file)), cfg.output_dirname)
        if "omni.isaac.vscode" not in candidate.replace("\\", "/"):
            return candidate

    for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        root = f"{letter}:/{cfg.project_rel}"
        if os.path.isdir(root):
            return os.path.join(root, cfg.output_dirname)
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


# ---------------------------------------------------------------------------
# 主流程（步驟都寫在這裡，方便從頭讀到尾）
# ---------------------------------------------------------------------------
async def main() -> None:
    """
    全流程說明（m2：URDF 匯入 Franka + 雙相機連續截圖）
    ================================================================

    【怎麼跑】
      請送 m2-run.py（短啟動器）。不要直接送本檔：
      omni.isaac.vscode 的 UDP 有大小限制，整份腳本 / 超大 log
      容易觸發 WinError 10040。

    【總目標】
      1. 用 URDF 把 Franka Panda（7 臂 + 2 指）匯進 Isaac Sim
      2. 掛兩台相機：hand（夾爪視角）+ side（側拍）
      3. 關節角從起始姿態線性插值到伸展姿態，跑 30 step
      4. 每 step 同步擷取兩台相機 → PNG + GIF + 場景 USD

    【輸出長這樣】
      m2_result/
        hand/frame_000.png ... frame_029.png
        side/frame_000.png ... frame_029.png
        hand.gif
        side.gif
        franka_panda_camera.usd

    ------------------------------------------------
    Step 1  準備輸出目錄
    ------------------------------------------------
      - resolve_output_dir() 找專案旁的 m2_result/
      - 避開 vscode 擴充套件目錄（Send to Isaac 時 __file__ 會騙人）
      - 目錄不存在就建立

    ------------------------------------------------
    Step 2  開啟空白舞台
    ------------------------------------------------
      - new_stage_async() 清舊場景、開新 USD stage
      - 必須非同步，避免和 Kit 主迴圈搶時序
      - 之後所有物件（燈、地面、機器人、相機）都掛在這個 stage 上

    ------------------------------------------------
    Step 3  建立場景基礎
    ------------------------------------------------
      - DistantLight：平行光，讓畫面夠亮
      - GroundPlane：地面參考（側邊相機才看得出「地面在下方」）
      - ReferenceCube：放在手臂前方，給末端相機當視覺目標

    ------------------------------------------------
    Step 4  URDF 匯入 Franka
    ------------------------------------------------
      - 啟用 omni.importer.urdf（已開就跳過，避免超大 log）
      - 動態拼出內建 panda_arm_hand.urdf 路徑
      - URDFCreateImportConfig + URDFParseAndImportFile
      - fix_base=True：底座鎖死，否則重力會推倒
      - 重要路徑區分：
          articulation_root ≈ /panda/root_joint  → Articulation API
          hand_path         = /panda/panda_hand   → 末端相機父節點
      - 匯入後多等幾幀，等 USD composition / articulation 展開完成

    ------------------------------------------------
    Step 5  啟動物理 + initialize 機器人
    ------------------------------------------------
      - timeline.stop() → play()：物理引擎開始跑
      - Articulation(articulation_root).initialize()
        把 link/joint 綁到 PhysX；不 init 則 set_joint_positions 無效
      - 印 DOF 數與 dof_names（關節角陣列順序以此為準）
      - 注意：play 之後、正式拍照之前可能先跑幾幀（暖機），
        那些幀不會存成 frame_XXX；正式序列從 Step 7 才開始。

    ------------------------------------------------
    Step 6  建立雙相機 + Replicator + 暖機
    ------------------------------------------------
      - 強制 RayTracedLighting（避免 path tracing 單幀怪異）
      - 對 Config.cameras 每一台：
          1) 建 USD Camera prim（位置 / 朝向）
          2) render_product =「底片」（哪台相機、多大解析度）
          3) rgb annotator = 從底片讀像素
          4) 建立 {output_dir}/{name}/ 子目錄
      - hand：掛在 panda_hand，視線沿夾爪接近方向（跟著末端動）
      - side：掛在 /World，固定機位側拍整隻手臂
      - 暖機：step_async 數次，等到畫面 max>0（前幾幀常是空/全黑）
        暖機圖只用來確認就緒，不寫進正式輸出

    ------------------------------------------------
    Step 7  運動模擬 + 多相機同步截圖（核心迴圈）
    ------------------------------------------------
      - 起始角 = 當下 get_joint_positions()
      - 目標角 = Config.target_arm（7 臂）+ finger_open（2 指）
      - 迴圈 num_steps（預設 30）次，每步：
          a) 線性插值關節角 → set_joint_positions
          b) timeline.forward_one_frame + wait_frames
             （物理真正推進）
          c) rep.orchestrator.step_async 一次
             （所有已 attach 相機同一幀被渲染，幀號對齊）
          d) 各 annotator 讀 RGBA → 存 PNG，並留在記憶體給 GIF
      - 如何判斷「有在動」：log 裡各相機 mean= 應隨 step 變化

    ------------------------------------------------
    Step 8  合成 GIF
    ------------------------------------------------
      - 把 Step 7 收集的 PIL 幀列表存成：
          {output_dir}/hand.gif
          {output_dir}/side.gif
      - duration=100ms ≈ 10 FPS，loop=0 無限循環

    ------------------------------------------------
    Step 9  儲存場景 USD
    ------------------------------------------------
      - save_as_stage → franka_panda_camera.usd
      - 之後可直接開啟檢查機器人 / 相機 / 燈光
      - 在編輯器按 Play = 跑物理 + 關節驅動，不是預錄動畫

    【和 m1 的差別（一句話）】
      m1：單一剛體方塊 + 模擬完拍一張
      m2：Articulation 機器人 + 關節插值運動 + 雙相機連續幀
    """

    cfg = CFG
    try:
        # ==================================================================
        # Step 1：準備輸出目錄
        # -----------------------------------------------------------------
        # 要存：
        #   {output_dir}/hand/frame_XXX.png
        #   {output_dir}/side/frame_XXX.png
        #   {output_dir}/hand.gif 、 side.gif
        #   {output_dir}/franka_panda_camera.usd
        # resolve_output_dir() 會避開 vscode 擴充套件目錄，找專案旁的 m2_result。
        # ==================================================================
        carb.log_warn("=== [Step 1] Entered main() ===")
        output_dir = resolve_output_dir(cfg)
        os.makedirs(output_dir, exist_ok=True)
        carb.log_warn(f"=== [Step 1] Output dir: {output_dir} ===")

        # ==================================================================
        # Step 2：開啟空白舞台
        # -----------------------------------------------------------------
        # new_stage_async()：清掉舊場景、開新 USD stage。
        # 必須用非同步版，才不會和 Kit 主迴圈搶時序導致當機。
        # stage = 之後所有 prim（燈、地面、機器人、相機）都掛在這上面。
        # ==================================================================
        carb.log_warn("=== [Step 2] Creating new stage ===")
        await omni.usd.get_context().new_stage_async()
        stage = omni.usd.get_context().get_stage()

        # ==================================================================
        # Step 3：建立場景基礎（燈 + 地面 + 參考立方體）
        # -----------------------------------------------------------------
        # DistantLight：平行光，強度夠機器人才看得清楚。
        # GroundPlane：視覺參考，也讓側邊相機看得出「地面在下方」。
        # ReferenceCube：放在手臂前方；末端相機若朝外，畫面才有明顯目標。
        # ==================================================================
        carb.log_warn("=== [Step 3] Building scene ===")

        # 平行光源（模擬太陽光），繞 X 轉 -45° 讓陰影方向自然一點
        light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
        light.CreateIntensityAttr(cfg.light_intensity)
        light.AddTransformOp().Set(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), -45)))

        # 地面：法線朝 Z（Isaac 預設向上），灰色大平面
        PhysicsSchemaTools.addGroundPlane(
            stage, "/World/GroundPlane", "Z", 10.0, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5, 0.5, 0.5)
        )

        # 參考立方體：給相機當視覺目標（不一定要剛體，碰撞可有可無）
        cube = UsdGeom.Cube.Define(stage, "/World/ReferenceCube")
        cube.GetSizeAttr().Set(cfg.ref_cube_size)
        cube.AddTranslateOp().Set(Gf.Vec3f(*cfg.ref_cube_pos))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

        # ==================================================================
        # Step 4：用 URDF 匯入 Franka Panda
        # -----------------------------------------------------------------
        # Isaac Sim 4.0 正確做法是 Kit Command，不是舊偽代碼的 UrdfImporter：
        #   URDFCreateImportConfig  → 建立匯入設定
        #   URDFParseAndImportFile  → 解析 URDF 寫入舞台
        #
        # 兩個路徑不要搞混：
        #   articulation_root ≈ /panda/root_joint  → 給 Articulation API 用
        #   hand_path         = /panda/panda_hand  → 給末端相機掛載用
        # 若相機誤掛在 root_joint 底下，路徑不存在 → 畫面不會跟著手動。
        # ==================================================================
        carb.log_warn("=== [Step 4] Importing Franka URDF ===")

        # 啟用 URDF importer（已開就跳過，避免 enable_extension 噴超大 log → 10040）
        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        if not ext_mgr.is_extension_enabled("omni.importer.urdf"):
            enable_extension("omni.importer.urdf")
            await wait_frames(1)

        # 動態找內建 panda_arm_hand.urdf，避免硬編碼版本號資料夾
        urdf_path = os.path.join(
            get_extension_path_from_name("omni.importer.urdf"),
            "data", "urdf", "robots", "franka_description", "robots", "panda_arm_hand.urdf",
        )
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"Franka URDF not found: {urdf_path}")
        carb.log_warn(f"=== [Step 4] URDF: {urdf_path} ===")

        # 匯入設定：固定基座、帶慣性、建立物理場景
        _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.merge_fixed_joints = False    # 保留固定關節，結構較完整
        import_config.convex_decomp = False         # 不做凸分解，加快匯入
        import_config.import_inertia_tensor = True  # 慣性張量，物理才正確
        import_config.fix_base = True               # 底座鎖死，否則重力會推倒
        import_config.make_default_prim = False
        import_config.create_physics_scene = True

        # 真正匯入：舞台上出現 /panda 這棵 link+joint 樹
        _, articulation_root = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
            import_config=import_config,
            get_articulation_root=True,  # 回傳 PhysX articulation 根
        )
        if not stage.GetPrimAtPath(cfg.hand_path).IsValid():
            raise RuntimeError(f"Expected hand prim missing: {cfg.hand_path}")

        # 第一次展開 articulation 會觸發 USD composition，稍慢正常，多等幾幀
        await wait_frames(cfg.composition_wait_frames)
        carb.log_warn(
            f"=== [Step 4] articulation={articulation_root}, hand={cfg.hand_path} ==="
        )

        # ==================================================================
        # Step 5：啟動時間軸 + initialize Articulation
        # -----------------------------------------------------------------
        # timeline.play()：物理引擎開始跑（不 play，關節驅動不會動）
        # Articulation.initialize()：把 link/joint 綁到 PhysX
        #   一定要呼叫，否則後面 set_joint_positions 無效
        # dof_names：之後設關節角陣列時，順序以此為準
        # ==================================================================
        carb.log_warn("=== [Step 5] Initializing Articulation ===")
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()   # 從頭開始，避免繼承上次模擬狀態
        timeline.play()   # 啟動物理
        await wait_frames(1)

        robot = Articulation(prim_path=articulation_root)
        robot.initialize()
        if not robot.handles_initialized:
            raise RuntimeError(f"Articulation failed to initialize at {articulation_root}")
        carb.log_warn(f"=== [Step 5] DOF={robot.num_dof}, names={robot.dof_names} ===")

        # ==================================================================
        # Step 6：建立多相機 + Replicator 管線 + 暖機
        # -----------------------------------------------------------------
        # 每台相機做三件事：
        #   1) USD Camera prim（位置 / 朝向）
        #   2) render_product（底片：哪台相機、多大解析度）
        #   3) rgb annotator（從底片讀像素）
        #
        # hand：掛在 panda_hand，視線沿夾爪 +Z，跟著末端動（eye-in-hand）
        # side：掛在 /World，固定側拍整隻手臂
        #
        # 暖機原因：annotator 剛 attach 時前幾幀常是空陣列或全黑，
        # 必須 step_async 幾次等到 max>0 再正式擷取。
        #
        # 輸出約定：
        #   PNG → {output_dir}/{name}/frame_XXX.png
        #   GIF → {output_dir}/{name}.gif
        # ==================================================================
        carb.log_warn("=== [Step 6] Setting up cameras ===")
        # 強制即時光線追蹤，避免 path tracing 單幀怪異 / 過慢
        carb.settings.get_settings().set("/rtx/rendermode", "RayTracedLighting")

        # cameras 執行期狀態：[{name, annotator, png_dir, frames}, ...]
        cameras = []
        for spec in cfg.cameras:
            # --- 建立 Camera prim 並設定位姿 ---
            cam = UsdGeom.Camera.Define(stage, spec.prim_path)
            xform = UsdGeom.Xformable(cam.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3f(*spec.translation))
            xform.AddRotateXYZOp().Set(Gf.Vec3f(*spec.rotate_xyz_deg))
            cam.GetClippingRangeAttr().Set(Gf.Vec2f(*spec.clipping))
            cam.GetFocalLengthAttr().Set(spec.focal_length)

            # --- 接到 Replicator：底片 + RGB 讀取器 ---
            rp = rep.create.render_product(spec.prim_path, spec.resolution)
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach([rp])

            # --- 這台相機的 PNG 子目錄 ---
            png_dir = os.path.join(output_dir, spec.name)
            os.makedirs(png_dir, exist_ok=True)
            cameras.append(
                {"name": spec.name, "annotator": annotator, "png_dir": png_dir, "frames": []}
            )
            carb.log_warn(f"=== [Step 6] camera {spec.name} @ {spec.prim_path} ===")

        # --- 逐台暖機：等到出現非全黑畫面 ---
        for cam in cameras:
            ready = False
            for i in range(cfg.warmup_frames):
                # orchestrator.step_async = 強制渲染一幀（總調度員發號施令）
                await rep.orchestrator.step_async()
                await wait_frames(1)
                rgba = rgba_to_uint8(cam["annotator"].get_data())
                if rgba is None:
                    carb.log_warn(f"=== warmup[{cam['name']}] {i+1}: empty ===")
                    continue
                carb.log_warn(
                    f"=== warmup[{cam['name']}] {i+1}: "
                    f"min={rgba.min()} max={rgba.max()} mean={rgba.mean():.1f} ==="
                )
                if rgba.max() > 0:
                    ready = True
                    break
            if ready:
                carb.log_warn(f"=== [Step 6] {cam['name']} ready ===")
            else:
                carb.log_warn(f"=== [Step 6] {cam['name']} still dark after warmup ===")

        # ==================================================================
        # Step 7：運動模擬 + 多相機同步截圖
        # -----------------------------------------------------------------
        # 目標：從目前關節角，線性插值到 target_arm（手臂伸展姿態）。
        #
        # 每一 step 做：
        #   a) 算插值關節角，set_joint_positions（教學用直接設角）
        #   b) timeline.forward_one_frame + wait → 物理真正推進
        #   c) step_async 一次 → 所有相機同一幀被渲染（幀號對齊）
        #   d) 各 annotator 讀像素 → 存 PNG，並留在記憶體給後面組 GIF
        #
        # 如何判斷相機有跟著動：看 log 的 mean= 是否隨 step 變化。
        # 若 30 幀 mean 完全一樣，多半是相機沒掛對父節點。
        # ==================================================================
        carb.log_warn("=== [Step 7] Motion + capture ===")

        # 起始姿態 = 目前關節角；目標姿態 = Config.target_arm + 手指微開
        start = robot.get_joint_positions()
        if start is None:
            start = np.zeros(robot.num_dof, dtype=np.float32)
        else:
            start = np.array(start, dtype=np.float32)

        target = start.copy()
        arm = np.array(cfg.target_arm, dtype=np.float32)
        n_arm = min(7, robot.num_dof)       # 前 7 個通常是手臂關節
        target[:n_arm] = arm[:n_arm]
        if robot.num_dof >= 9:              # 後 2 個是手指
            target[7] = cfg.finger_open
            target[8] = cfg.finger_open

        n = cfg.num_steps
        for i in range(n):
            # a) 線性插值(從起始姿態到目標姿態的過程中，關節角度的變化是線性的)：i=0 → start，i=n-1 → target
            alpha = float(i) / float(n - 1)
            robot.set_joint_positions((1.0 - alpha) * start + alpha * target)

            # b) 推進一幀物理，等 Kit 真正處理完
            timeline.forward_one_frame()
            await wait_frames(1)

            # c) 一次渲染：hand / side 對齊同一幀
            await rep.orchestrator.step_async()
            await wait_frames(1)

            # d) 各相機取圖並存檔
            mean_parts = []
            for cam in cameras:
                rgba = rgba_to_uint8(cam["annotator"].get_data())
                if rgba is None:
                    mean_parts.append(f"{cam['name']}=empty")
                    continue
                img = Image.fromarray(rgba[:, :, :3])
                img.save(os.path.join(cam["png_dir"], f"frame_{i:03d}.png"))
                cam["frames"].append(img)  # 留給 Step 8 組 GIF
                mean_parts.append(f"{cam['name']}={float(rgba.mean()):.1f}")

            # 每 5 步印一次進度（mean 有變 = 畫面有在變）
            if i == 0 or (i + 1) % 5 == 0 or i + 1 == n:
                cur = robot.get_joint_positions()
                joint2 = float(cur[1]) if cur is not None else float("nan")
                carb.log_warn(
                    f"=== Step {i+1}/{n}: joint2={joint2:.3f}, " + ", ".join(mean_parts) + " ==="
                )

        timeline.pause()  # 停在最後一幀，方便存場景

        # ==================================================================
        # Step 8：把各相機的連續幀合成 GIF
        # -----------------------------------------------------------------
        # 輸入：Step 7 存在 cam["frames"] 的 PIL Image 列表
        # 輸出：{output_dir}/hand.gif 、 {output_dir}/side.gif
        # duration=100ms ≈ 10 FPS，loop=0 無限循環
        # ==================================================================
        carb.log_warn("=== [Step 8] Building GIFs ===")
        for cam in cameras:
            gif_path = os.path.join(output_dir, f"{cam['name']}.gif")
            saved = save_gif(cam["frames"], gif_path, cfg.gif_frame_duration_ms)
            if saved:
                carb.log_warn(f"=== [GIF] {saved} ({len(cam['frames'])} frames) ===")
            else:
                carb.log_warn(f"=== [GIF] skip {cam['name']} (no frames) ===")

        # ==================================================================
        # Step 9：儲存最終場景為 .usd
        # -----------------------------------------------------------------
        # 之後可直接在 Isaac 開啟檢查機器人、相機、燈光配置。
        # 按 Play 會跑物理 + 關節驅動，不是預錄動畫。
        # ==================================================================
        usd_path = os.path.join(output_dir, cfg.usd_filename)
        omni.usd.get_context().save_as_stage(usd_path)
        carb.log_warn(f"=== [Done] PNG + GIF + USD → {output_dir} ===")

    except Exception as e:
        # traceback 截短，避免 UDP 回傳超大 log 觸發 WinError 10040
        tb = traceback.format_exc()
        carb.log_error(f"=== Critical error: {e} ===")
        carb.log_error(tb[:2000] + ("...(truncated)" if len(tb) > 2000 else ""))


# Entry：腳本送進 Isaac 後，把 main() 排進非同步任務佇列
carb.log_warn("=== [Entry] Script received, scheduling async task ===")
asyncio.ensure_future(main())

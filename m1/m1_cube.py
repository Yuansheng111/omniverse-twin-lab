"""
1. pxr 不是有不足，而是「職責不同 (Scope)」
pxr (Pixar USD) 是一套開源的通用場景描述標準。

它的職責：只負責定義「資料長什麼樣子」。例如：這裡有一個頂點、那裡有一張貼圖、這個方塊有多重。
它的限制：它完全沒有「軟體介面」的概念。它不知道什麼是滑鼠點擊、不知道什麼是時間軸播放、
也不知道 NVIDIA 的 RTX 顯示卡長什麼樣子。它就像是網頁設計裡的 HTML 程式碼，只負責描述結構。

2. omni 是負責讓資料「活起來」的引擎介面
omni (Omniverse API) 則是 NVIDIA 為自家軟體寫的核心引擎 (Kit) 與擴充套件。

它的職責：負責管理視窗、UI 介面、外掛程式、事件廣播，以及把底層資料餵給物理引擎 (PhysX) 和渲染器 (RTX)。
它就像是網頁瀏覽器（如 Chrome），負責把 HTML 渲染出來並加上互動功能。

總結來說 (Summary)
pxr 負責 "最純粹的資料定義與讀寫"。
只要是跟「修改場景屬性、層級結構」有關的核心操作，你最終都得回到 pxr。

omni "負責軟體互動與高階封裝"。
它補足了 USD 沒有的 UI 與事件系統，
並提供大量「捷徑 (Wrappers)」，讓你不用每次都寫落落長的 pxr 原生程式碼。

所以，在 Isaac Sim 的開發中，永遠是兩者混用的：
用 omni 管理環境與軟體狀態，然後在需要精細微調物件屬性時，伸手進去呼叫 pxr 的 API！
"""

import omni.usd                                     # 管理 USD 場景上下文（開啟、儲存舞台）
import omni.replicator.core as rep                  # 合成資料生成，用於建立攝影機與擷取影像, Omniverse Replicator (一個專門用來產生 AI 訓練資料的工具)
import omni.timeline                                # 控制模擬時間軸（播放、暫停、步進）
import omni.kit.app                                 # 取得應用程式主迴圈，用於等待幀更新
import carb                                         # Omniverse 原生日誌系統,

from pxr import UsdGeom, Gf, UsdPhysics, UsdLux     # USD 核心函式庫：幾何、數學、物理、燈光
import asyncio                                      # Python 非同步任務排程
import os                                           # 檔案系統操作（路徑處理、目錄建立）
import traceback                                    # 擷取並格式化完整的例外堆疊資訊

async def main():
    try:
        carb.log_warn("=== [Step 1] Successfully entered main() async task ===")
        
        # 1. 定義並建立相對輸出目錄
        output_dir_raw = os.path.join(os.path.dirname(os.path.dirname(__file__)), "m1_result")
        output_absolute_path = os.path.abspath(output_dir_raw)
        
        if not os.path.exists(output_absolute_path):
            os.makedirs(output_absolute_path)
            carb.log_warn(f"=== [Step 1 Confirmed] Directory created: {output_absolute_path} ===")
        else:
            carb.log_warn(f"=== [Step 1 Confirmed] Directory already exists: {output_absolute_path} ===")

        # 2. 開啟新的空白舞台 (改用非同步方法，避免時序當機)
        carb.log_warn("=== [Step 2] Creating new stage ===")
        await omni.usd.get_context().new_stage_async()  # 獲得當前場景的上下文，並建立新的空白舞台
        stage = omni.usd.get_context().get_stage()      # 獲得當前場景的舞台

        # 建立 World 根節點 Xform
        UsdGeom.Xform.Define(stage, "/World")

        # 3. 新增平行光源（Distant Light）
        light_path = "/World/DistantLight"
        distant_light = UsdLux.DistantLight.Define(stage, light_path)
        distant_light.CreateIntensityAttr(1500) # 設定光源的強度為 1500
        distant_light.AddTransformOp().Set(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), -45)))
        # AddTransformOp -> 新增一個變換操作, Set -> 旋轉光源45度(建立旋轉矩陣, 繞x軸旋轉-45度)

        # 4. 使用 USD API 建立立方體 Prim
        carb.log_warn("=== [Step 3/4] Creating light and cube ===")
        cube_path = "/World/Cube"
        cube = UsdGeom.Cube.Define(stage, cube_path)
        cube.GetSizeAttr().Set(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 5.0))
        
        # 套用物理 API
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())   # 把碰撞 API 套用到立方體的 Prim 上
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())   # 把剛體 API 套用到立方體的 Prim 上

        # 5. 設置攝影機與影像輸出管線
        carb.log_warn("=== [Step 5] Setting up Replicator camera ===")

        # 在 3D 空間中放一台虛擬攝影機，位置 (x=3, y=4, z=6)，自動旋轉對準立方體
        camera = rep.create.camera(position=(3.0, 4.0, 6.0), look_at=cube_path)

        # 將攝影機綁定到一個「渲染輸出埠」，解析度 1024x1024
        # 可以把 render_product 想成：攝影機底片，決定要用哪台相機、拍多大張
        render_product = rep.create.render_product(camera, (1024, 1024))

        # 取得 BasicWriter：負責把渲染結果寫成實際的圖片檔案
        writer = rep.WriterRegistry.get("BasicWriter")
        # 告訴 writer 要把圖片存到哪個資料夾，rgb=True 表示輸出彩色影像
        writer.initialize(output_dir=output_absolute_path, rgb=True)
        # 注意：尚未 attach！attach 放在模擬迴圈結束後，
        # 避免 writer 在每次 next_update_async() 渲染時自動擷取，導致輸出 10 張圖

        # 6. 驅動物理模擬逐幀推進
        carb.log_warn("=== [Step 6] Starting physics simulation steps ===")
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()   # 先確保時間軸從頭開始，避免繼承上次狀態
        timeline.play()   # 啟動物理引擎（不 play 的話重力不會生效）

        for i in range(10):
            # 手動推進一幀（讓物理引擎計算這一幀的重力、碰撞等）
            timeline.forward_one_frame()
            # 等待 Kit 主迴圈真正處理完這一幀，確保物理結果已寫入場景再繼續
            await omni.kit.app.get_app().next_update_async()
            carb.log_warn(f"=== Physics simulation progress: {i+1}/10 ===")

        timeline.pause()  # 停在第 10 幀，保持場景狀態供後續擷取

        # 模擬結束後才把「底片」接到「輸出裝置」，確保只擷取最終這一幀
        writer.attach([render_product])

        # 7. 觸發 Replicator 擷取這一幀的畫面並寫入磁碟
        carb.log_warn("=== [Step 7] Capturing image ===")
        # orchestrator 是 Replicator 的「總調度員」：
        #   當場景中有多台攝影機、多個 writer 時，
        #   它負責統一發號施令，確保所有人在同一幀同時拍照、同時寫檔，
        #   而不是各自為政導致幀號對不上。
        # step_async() = 向 orchestrator 發出「執行一次擷取」的指令，
        # 必須 await 等它完成，否則圖片可能還沒寫入磁碟程式就結束了
        await rep.orchestrator.step_async()
        
        # 8. 將最終場景儲存為 .usd 檔案
        usd_path = os.path.join(output_absolute_path, "my_physics_cube.usd")
        omni.usd.get_context().save_as_stage(usd_path)
        
        carb.log_warn(f"=== [Done] File saved to: {usd_path} ===")

    except Exception as e:
        carb.log_error(f"=== Critical error during execution: {e} ===")
        carb.log_error(traceback.format_exc())

# Entry point: confirm the script was successfully sent from VS Code
carb.log_warn("=== [Entry] Script received from VS Code, scheduling async task ===")
asyncio.ensure_future(main())

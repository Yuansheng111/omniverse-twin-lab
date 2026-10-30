# M5：從模擬器摸到第一份訓練資料長什麼樣

這份筆記用白話說明 M5 在做什麼、為什麼要做、磁碟上那包東西怎麼讀。  
先讀最上面幾節就夠建立圖像；後面才是形狀、決策過程、怎麼重跑。

---

## 一句話

M4 是「讓小車用分數學會站穩」，產出是一條獎勵曲線。  
M5 是「讓模擬器吐出一份資料檔」，產出是可以打開來看的訓練資料格式。

**今天沒有在訓練策略。** 亂甩手臂錄 100 步，只為了親手摸到：以後做訓練，前面餵進去的資料會長什麼樣。

跑蒐集前的環境（Lab 2.3.2 + Sim 5.0、不要用舊 conda）見下面「跑之前要建好的環境」。

---

## 為什麼要有 M5

C1 一路從空白場景（M1）、Franka 加相機（M2）、路徑規劃（M3）、用分數學習（M4）走過來。  
後面無論是：

- 用獎勵更新網路（M4 那條線）
- 看示範學會動作（模仿學習）
- 看圖輸出動作（視覺策略）
- 把真機資料接進來

都需要同一種東西：**每一步對齊的「我看到什麼 / 手在哪 / 我下了什麼指令」。**

講義把這叫資料飛輪從模擬器端起飛。M5 是飛輪的第一圈：管子先通、格式先對。  
100 步對真正訓練太少、內容也是亂數；差距不是「再多錄幾幀就變產品」，而是結構、多樣性、真不真實。今天只摸手感。

---

## 你實際做出來的東西

都在 `m5/datasets/run_001/`：

| 檔案 | 角色 |
|------|------|
| `trajectory_0.npz` | 主資料。一個壓縮包裡放四個具名陣列。 |
| `rgb_0000.png` | 第 0 步畫面預覽，給人看。訓練以後讀的是 npz，不是這張圖。 |

蒐集腳本是 `collect_franka_dataset.py`。環境用官方 `Isaac-Reach-Franka-v0`（Franka Panda 伸手任務），再自己在手腕掛一顆相機。官方這份任務預設沒有手腕相機。

---

## npz 裡面有什麼

四個陣列，第 0 維都是「第幾步」，共 100 步。第 `i` 步的圖、位姿、動作、時間是同一拍。這叫對齊。

| 名字 | 形狀(張量的描述) | 白話 |
|------|------|------|
| `rgb` | `(100, 200, 200, 3)` | 手腕相機彩色圖。100 張，每張高 200、寬 200，紅綠藍三通道。 |
| `ee_pose` | `(100, 7)` | 手掌在世界座標：位置 xyz（3 個）+ 旋轉四元數（4 個）。 |
| `actions` | `(100, 7)` | 那一步送給七個手臂關節的指令。 |
| `timestamps` | `(100,)` | 模擬時間（秒），這次是 0.0 到 3.3。 |

`ee_pose` 不是關節角度，而是「手掌這塊剛體現在在空間裡的位置與朝向」。

動作目前是範圍大約 \([-1, 1]\) 的亂數。手臂會抖、姿勢會醜，都正常。之後若換成規劃軌跡或遙控示範，欄位可以不變，只換「動作從哪來」。

---

## 「形狀」是什麼意思

`(100, 7)` 這種寫法叫 **shape（形狀）**。NumPy、PyTorch 都這樣寫。

從左到右讀：

- 第一個數字：最外層有幾個。這裡幾乎都是「100 個時間步」。
- 後面的數字：那一步裡面長什麼樣。

可以這樣記：

- `(7,)`：一排 7 個數字（向量）
- `(100, 7)`：100 列 × 7 欄（矩陣／二維張量）
- `(100, 200, 200, 3)`：100 張圖（四維張量）

**張量**是統稱：1 維叫向量，2 維叫矩陣，再高維還是張量。深度學習常把這些一概叫 tensor。  
所以 `rgb` 確實是「用一個張量把每次截到的彩色畫面疊起來」；npz 裡另外三個也是張量，不是只有畫面。

讀檔大意：

```python
import numpy as np

data = np.load("datasets/run_001/trajectory_0.npz")
print(data["rgb"].shape)       # (100, 200, 200, 3)
print(data["ee_pose"][0])      # 第 0 步的手掌位姿，長度 7
```

`rgb[0]` 就是 `rgb_0000.png` 那一幀。

npz **不能當 Word 那樣雙擊預覽裡的數字**。它是給程式讀的壓縮包。看的方式：

1. **畫面**：直接開 `rgb_0000.png`。
2. **形狀與數字**：跑 `inspect_dataset.py`（不用開模擬器）：

```powershell
cd D:\Desktop\git\tt\m5
D:\Desktop\git\tt\IsaacLab\_isaac_sim\kit\python\python.exe inspect_dataset.py
```

`np.load` 之後用鍵名取值，和字典很像：`data["ee_pose"][5]` 是第 5 步的手掌位姿。

---

## 為什麼用 npz？講義不是還寫了 HDF5？

兩個都是「一個檔裡放好多個具名陣列」。差異在給誰用、檔大了好不好讀。

**npz（這次用的）**

- NumPy 內建：`np.savez_compressed` / `np.load`，不必再裝套件。
- 本質是 zip，裡面幾個 `.npy`。小檔、欄位少、軌跡只有 1 條時最省事。
- 講義寫「半小時實作建議先用 npz」：地板低、沒依賴。
- 這台機器 M4 開 GUI 時 `h5py` 曾經 DLL 炸掉，第一次落地更不該綁 HDF5。

**HDF5（講義另一個選項，副檔名常是 `.h5` / `.hdf5`）**

- 用 `h5py` 讀寫，像硬碟上的資料夾樹：`data/demo_0/obs/rgb` 這種路徑。
- 可以只讀其中一個欄位、甚至只讀某幾幀，不必整包載進記憶體。軌跡很多、畫面很大時比較合適。
- 視覺語言模型、擴散策略、NVIDIA 那條合成資料線，公開資料集很多用這個。Isaac Lab 官方 Recorder 預設也常導出 HDF5。

**今天為什麼不先上 HDF5**

M5 是 N=1、T=100 的格式樣品。四個陣列一次載進記憶體毫無壓力。  
欄位契約（`rgb` / `ee_pose` / `actions` / `timestamps`）跟用哪種容器無關；以後真的要並排很多條示範，再把同樣的鍵寫進 HDF5 即可，不是換一種「訓練理論」。

一句話：**npz 是小作業的塑膠袋；HDF5 是大資料庫的櫃子。講義兩個都認，第一次用袋子就對了。**

---

## 為什麼是這四個欄位

不是教科書上馬可夫決策過程的四元組直接抄下來的。

馬可夫決策過程（Markov Decision Process）通常講：

- 狀態：現在世界長怎樣
- 動作：我做了什麼
- 獎勵：這步好不好
- 下一狀態：做完變成怎樣

對到這份檔：

| 決策過程概念 | 這份資料 | 說明 |
|--------------|----------|------|
| 觀測／狀態的一部分 | `rgb`、`ee_pose` | 眼睛 + 手在哪。不是完整世界狀態（沒存全部關節、目標點等）。 |
| 動作 | `actions` | 對得上。 |
| 獎勵 | 沒存 | 環境當下有算，這次沒寫進檔。 |
| 下一狀態 | 沒單獨開鍵 | `rgb[t+1]` 就是下一步。 |
| （不是決策欄位） | `timestamps` | 對齊相機與手臂用的時間戳。 |

這四個是作業選的**最小儲存契約**：接 M2 的相機、把「圖 + 手 + 指令」定下來，給以後視覺策略／模仿學習當格式樣品。

### 能不能加工成用決策過程訓練的資料？

格式上能靠，內容不行。

- **模仿學習**（看當時做了什麼）：觀測＋動作已經在，示範很爛而已。
- **用獎勵訓練**（M4 那類）：還要補獎勵、結束與否；下一步可用 `t+1`。就算補齊，亂甩手臂也沒有「該學的好行為」。

契約對了；教材還沒換進去。

---

## 腳本在做什麼（對照程式）

`collect_franka_dataset.py` 順序和 M4 的 `train.py` 同一規矩：

1. **先開 Isaac Sim**，再 import 環境。模擬器沒起來就載入任務，會炸。
2. **強制開相機渲染**（`enable_cameras`）。沒視窗也要畫圖，否則 `rgb` 會是空的。
3. 載入官方伸手任務，**在 `gym.make` 之前掛手腕相機**。改的是設定物件，不是改 Isaac Lab 套件原始碼。
4. 迴圈 100 次：亂抽動作 → `env.step` → 立刻讀圖、位姿、動作、時間。
5. 壓成 npz，並把第 0 幀另存 PNG。

相機掛在 `panda_hand` 底下，手臂一動鏡頭跟著動。偏移抄官方 visuomotor Franka 範例。

這台機器上，Isaac Lab 2.3.2 預設的 Fabric 高速路徑會呼叫 Warp 的 `transform_compose`，Isaac Sim 5.0 內建 Warp 沒有這個函式，一 `reset` 相機就炸。腳本因此設了 `use_fabric=False`，相機位姿改走 USD。只有 1 個環境，夠用，產出欄位不變。

---

## 跑之前要建好的環境

蒐集腳本活在 Isaac Sim 裡面，**不是**用系統 Python 或隨便一個 conda 就能 `python collect_franka_dataset.py`。  
這台電腦上已經對齊、M4／M5 能跑通的是下面這套；換一套就會 `ModuleNotFoundError` 或相機 reset 炸掉。

**1. 兩塊要對在一起**

| 東西 | 這台機器上的位置 | 為什麼 |
|------|------------------|--------|
| Isaac Lab **2.3.2** | `D:\Desktop\git\tt\IsaacLab` | 套件名稱是 `isaaclab` / `isaaclab_tasks`（不是舊的 `omni.isaac.lab_tasks`） |
| Isaac Sim **5.0** 獨立版 | 用資料夾連結 `IsaacLab\_isaac_sim` 指到 Sim 5.0 安裝目錄 | Lab 必須掛在 Sim 的 Kit 上；USD、相機、PhysX 都在 Sim 裡 |

M4 已做過：`isaaclab.bat --install`（訓練時還加了 `sb3`）。蒐集資料**不需要** Stable-Baselines3；但 `isaaclab` 與 `isaaclab_tasks` 必須已經裝進 Sim 那份 Python。

**2. 用哪一個 Python**

用 Lab 目錄裡的啟動器，它會呼叫 `_isaac_sim` 裡的 Python **3.11**：

```powershell
cd D:\Desktop\git\tt\IsaacLab
.\isaaclab.bat -p ..\m5\collect_franka_dataset.py --headless --enable_cameras
```

不要用：

- conda 環境名叫 `isaaclab` 的那份（這台是舊專案：Python 3.10、Isaac Sim 4.0、`omni.isaac.lab`，路徑還指到別的工作區）
- 系統 Python、`python -c "import isaaclab_tasks"` 這種沒開 Sim 的測法（擴充沒載入，一定失敗）
- PowerShell 裡跑 Sim 的 `activate.bat`（那是 cmd 腳本，在 PowerShell 會壞）

**3. 硬體與套件小條件**

- **NVIDIA 獨立顯卡**（這台是 RTX 4070 Ti SUPER）。日誌裡跳過 AMD 內顯是正常的。
- 要出 RGB：命令列加 `--headless --enable_cameras`。腳本也會強制 `enable_cameras`，啟動器才會載入「無頭但會渲染」的 kit，而不是純物理、不畫圖的 kit。
- NumPy 必須是 **2 以下**。若中途為了畫圖裝過 matplotlib，有可能把 NumPy 拉到 2.x，Isaac Lab 會不穩。需要時在 kit 的 python 上再裝回：`pip install "numpy<2"`。
- 只讀現成 npz、不開模擬器：跑 `inspect_dataset.py` 時，用同一份 kit python 即可，不必再啟動 Isaac Sim。

**4. 工作目錄**

指令要在 `IsaacLab` 底下打，因為 `isaaclab.bat` 和 `_isaac_sim` 都在那裡。腳本本身在隔壁的 `m5\`，用 `..\m5\collect_franka_dataset.py` 指過去。產出寫進 `m5\datasets\`，不寫進 Lab 官方原始碼。

---

## 怎麼再跑一次

在 Isaac Lab 目錄：

```powershell
cd D:\Desktop\git\tt\IsaacLab
.\isaaclab.bat -p ..\m5\collect_franka_dataset.py --headless --enable_cameras
```

成功時會看到類似：

```text
[m5] saved ...\m5\datasets\run_001\trajectory_0.npz
[m5] rgb (100, 200, 200, 3)  ee_pose (100, 7)  actions (100, 7)  timestamps (100,)
```

開頭那些 Warp CUDA、跳過 AMD 內顯、DLSS 解析度太小，這次是警告，沒擋住存檔。

輸出寫在腳本旁邊的 `datasets/run_001/`，不寫進 Isaac Lab 官方 `source/`，以免之後更新官方碼被蓋掉。

---

## 跟 M1～M4 怎麼接

| 前面 | M5 用到什麼 |
|------|-------------|
| M1 | 場景會出圖 |
| M2 | Franka 七軸 + 末端相機 |
| M3 | 動作以後可換成規劃軌跡；今天用亂抽先通管子 |
| M4 | 同一套「先開模擬器再開環境」。M4 產出成績單，M5 產出原料。 |

M4 cartpole 沒有手臂相機，不能拿來當 M5 的環境。

---

## 收束時記得的三件事

1. **資料集 vs 獎勵曲線**：同一條時間軸，一個記原料（圖／手／指令），一個記成績（這局好不好加總）。曲線從互動算出來，不是從 npz 畫出來的。
2. **若餵給 M4 那種迴圈**：觀測用 `rgb`（可加 `ee_pose`），動作用 `actions`，獎勵這包沒有、要另存或事後算。網路也要會看圖，不能直接套 cartpole 那顆小網路。
3. **模擬器 vs 真機**：欄位結構應該一樣（同一套鍵名、同一種對齊）；光、摩擦、相機噪聲不會一樣。後面要對齊的是內容，不是再發明一套格式。

M5 要你帶走的不是「亂數很有價值」，而是：**訓練前面的資料契約長這樣；模擬器真的吐得出來。**

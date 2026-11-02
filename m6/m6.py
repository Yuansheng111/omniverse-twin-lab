"""Isaac Lab humanoid 站立平衡的 PPO 訓練框架.

流程是五段: 啟動 Isaac Sim、把人形環境收成站立平衡、用 RSL-RL 的 PPO
訓練、存 checkpoint、從 TensorBoard 畫 episode reward 曲線。

三項平衡分數 (高度、角速度、接觸力平滑) 與單腳支撐的入口留在
``_attach_balance_rewards``，這一版只先把「往前走」的分數關掉。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from isaaclab.app import AppLauncher

# SimulationApp 建立之後才能 import 其餘 isaaclab 套件。
parser = argparse.ArgumentParser(description="Humanoid standing balance PPO.")
parser.add_argument("--num_envs", type=int, default=4096, help="平行環境數量.")
parser.add_argument("--max_iterations", type=int, default=5000, help="PPO 學習迭代次數.")
parser.add_argument(
    "--stance",
    choices=("double", "single"),
    default="double",
    help="雙腳站立或單腳站立. 單腳的接觸遮罩尚未接上.",
)
parser.add_argument(
    "--plot_path",
    type=str,
    default="humanoid_balance_reward.png",
    help="reward 曲線輸出路徑.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401  # 註冊 Isaac-Humanoid-v0
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)
from isaaclab_tasks.manager_based.classic.humanoid.humanoid_env_cfg import HumanoidEnvCfg
from rsl_rl.runners import OnPolicyRunner

TASK_ID = "Isaac-Humanoid-v0"
LOG_ROOT = Path("logs") / "humanoid_balance"
REWARD_TAG = "Train/mean_reward"
Stance = Literal["double", "single"]


@dataclass(frozen=True)
class BalanceRewardWeights:
    """站立平衡三項分數的權重.

    沿用 Isaac Lab ``RewTerm.weight`` 的慣例: 正值加分, 負值懲罰.
    ``height`` 鼓勵軀幹維持高度, ``angular_velocity`` 懲罰晃動,
    ``contact_force_smoothness`` 懲罰腳與地面的力突然暴衝.

    Attributes:
        height: 站立高度項的權重.
        angular_velocity: 角速度懲罰的權重.
        contact_force_smoothness: 接觸力抖動懲罰的權重.
    """

    height: float = 1.0
    angular_velocity: float = -0.05
    contact_force_smoothness: float = -1.0e-3


@configclass
class HumanoidBalancePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """站立平衡用的 PPO 設定.

    ``max_iterations`` 是學習更新次數, 不是模擬器裡的單幀.
    每次更新會先讓所有平行環境各走 ``num_steps_per_env`` 步.
    """

    num_steps_per_env = 32
    max_iterations = 5000
    save_interval = 250
    experiment_name = "humanoid_balance"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[400, 200, 100],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[400, 200, 100],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def build_env_cfg(num_envs: int, stance: Stance) -> HumanoidEnvCfg:
    """由官方人形環境做出站立平衡設定.

    Args:
        num_envs: 同時模擬的機器人數.
        stance: ``double`` 為雙腳, ``single`` 為單腳.

    Returns:
        已關掉前進獎勵的環境設定.
    """
    env_cfg = HumanoidEnvCfg()
    env_cfg.scene.num_envs = num_envs
    apply_standing_balance(env_cfg, BalanceRewardWeights(), stance)
    return env_cfg


def apply_standing_balance(
    env_cfg: HumanoidEnvCfg,
    weights: BalanceRewardWeights,
    stance: Stance,
) -> None:
    """把走路任務收成站立: 前進分數歸零, 再交給平衡分數的掛點.

    官方 ``Isaac-Humanoid-v0`` 的 ``progress`` 與 ``move_to_target`` 是在獎
    勵往前走. 站立任務不使用這兩項. 存活與直立的分數先留著, 讓這一版框架
    仍然有「不要倒」的目標.

    Args:
        env_cfg: 官方人形環境設定, 會被就地修改.
        weights: 三項平衡分數的權重.
        stance: 雙腳或單腳.
    """
    env_cfg.rewards.progress.weight = 0.0
    env_cfg.rewards.move_to_target.weight = 0.0
    _attach_balance_rewards(env_cfg, weights, stance)


def _attach_balance_rewards(
    env_cfg: HumanoidEnvCfg,
    weights: BalanceRewardWeights,
    stance: Stance,
) -> None:
    """掛上高度、角速度、接觸力三項, 以及單腳支撐.

    這一版只保留入口. 單腳需要改腳與地面的接觸, 雙腳的三項 ``RewTerm``
    也還沒接到環境的 reward manager.

    Args:
        env_cfg: 站立平衡環境設定.
        weights: 三項平衡分數的權重.
        stance: 雙腳或單腳.
    """
    if stance == "single":
        raise NotImplementedError("單腳支撐的接觸遮罩還沒接上.")
    del env_cfg, weights


def build_runner_cfg(max_iterations: int) -> HumanoidBalancePPORunnerCfg:
    """建立 PPO runner 設定.

    Args:
        max_iterations: 學習迭代次數, 對應 ``OnPolicyRunner.learn`` 的
            ``num_learning_iterations``.

    Returns:
        已寫入迭代次數的 runner 設定.
    """
    runner_cfg = HumanoidBalancePPORunnerCfg()
    runner_cfg.max_iterations = max_iterations
    return runner_cfg


def train(env_cfg: HumanoidEnvCfg, runner_cfg: HumanoidBalancePPORunnerCfg) -> Path:
    """包裝環境並訓練, checkpoint 寫進 ``logs/humanoid_balance/<timestamp>/``.

    Args:
        env_cfg: 站立平衡環境設定.
        runner_cfg: PPO runner 設定.

    Returns:
        這次訓練的 log 目錄.
    """
    log_dir = LOG_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(TASK_ID, cfg=env_cfg)
    try:
        env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
        runner = OnPolicyRunner(
            env,
            runner_cfg.to_dict(),
            log_dir=str(log_dir),
            device=runner_cfg.device,
        )
        runner.learn(
            num_learning_iterations=runner_cfg.max_iterations,
            init_at_random_ep_len=True,
        )
    finally:
        env.close()
    return log_dir


def load_episode_rewards(log_dir: Path) -> tuple[list[int], list[float]]:
    """讀 TensorBoard 裡每個 iteration 的平均 episode reward.

    RSL-RL 把一個 episode 的分數加總後, 再對該輪收集到的 episode 取平均,
    寫在 ``Train/mean_reward``. 圖上的一個點就是這個回合總分.

    Args:
        log_dir: ``OnPolicyRunner`` 的 log 目錄.

    Returns:
        iteration 編號, 以及對應的平均 episode reward.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    if REWARD_TAG not in accumulator.Tags().get("scalars", []):
        available = ", ".join(accumulator.Tags().get("scalars", [])) or "(none)"
        raise KeyError(f"找不到 {REWARD_TAG}. 現有 scalar: {available}")

    events = accumulator.Scalars(REWARD_TAG)
    iterations = [int(event.step) for event in events]
    rewards = [float(event.value) for event in events]
    return iterations, rewards


def plot_reward_curve(log_dir: Path, output_path: Path) -> None:
    """把 episode reward 畫成 PNG.

    Args:
        log_dir: 含 TensorBoard event 的目錄.
        output_path: 圖片輸出路徑.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iterations, rewards = load_episode_rewards(log_dir)
    plt.figure()
    plt.plot(iterations, rewards)
    plt.xlabel("iteration")
    plt.ylabel("episode reward")
    plt.title(f"Humanoid Standing Balance (PPO, {iterations[-1]} iterations)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    """組環境、訓練、畫圖."""
    stance: Stance = args_cli.stance
    env_cfg = build_env_cfg(args_cli.num_envs, stance)
    runner_cfg = build_runner_cfg(args_cli.max_iterations)
    log_dir = train(env_cfg, runner_cfg)
    plot_path = Path(args_cli.plot_path)
    plot_reward_curve(log_dir, plot_path)
    print(f"log: {log_dir}")
    print(f"reward curve: {plot_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

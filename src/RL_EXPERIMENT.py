# ============================================================
# SimPy + PPO 실험 자동화 (Meta-RL 제거 버전)
# ============================================================

import os
import pandas as pd
import time
from GymWrapper import GymInterface
from config_SimPy import *
from config_RL import *
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from torch.utils.tensorboard import SummaryWriter

# ============================================================
# 기본 설정
# ============================================================
SIM_TIME = 200
N_EPISODES = 1000
RESULT_FILE = "PPO_Experiment_Results.csv"

# ============================================================
# Custom Callback for Evaluation
# ============================================================
class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super(CustomEvalCallback, self).__init__(*args, **kwargs)
        self.rewards = []

    def _on_step(self) -> bool:
        result = super(CustomEvalCallback, self)._on_step()
        if self.n_calls % self.eval_freq == 0:
            self.rewards.append(self.last_mean_reward)
        return result


def make_callback(env):
    return CustomEvalCallback(
        env,
        eval_freq=SIM_TIME * 2,  # 평가 주기
        n_eval_episodes=15,
        log_path='./RL_logs/',
        best_model_save_path='./RL_logs/',
        deterministic=True,
        render=False
    )


def build_model(env):
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0001,
        batch_size=32,
        n_steps=SIM_TIME,
        verbose=0
    )
    return model


# ============================================================
# 시나리오 정의
# ============================================================
def get_scenarios():
    scenarios = []

    # -------- Gaussian --------
    gaussian_cases = [
        (12, 1, 1, 0),
        (12, 1, 1, 1),
        (12, 1, 1, 2),
        (12, 1, 1, 3),
        (12, 1, 1, 4),
        (13, 1, 1, 0),
        (13, 1, 1, 1),
        (13, 1, 1, 2),
        (13, 1, 1, 3),
        (13, 1, 1, 4),
        (14, 1, 1, 0),
        (14, 1, 1, 1),
        (14, 1, 1, 2),
        (14, 1, 1, 3),
        (14, 1, 1, 4),
    ]
    for mean, std, l_mean, l_std in gaussian_cases:
        scenarios.append({
            "Scenario": "Gaussian",
            "Demand": {"Dist_Type": "GAUSSIAN", "mean": mean, "std": std},
            "Leadtime": {"Dist_Type": "GAUSSIAN", "mean": l_mean, "std": l_std}
        })

    # -------- Uniform --------
    uniform_cases = [
        (10, 14, 0, 2),
        (10, 14, 1, 3),
        (10, 13, 0, 2),
        (10, 13, 1, 3),
        (11, 13, 0, 2),
        (11, 13, 1, 3),
        (11, 15, 0, 2),
        (11, 15, 1, 3),
        (12, 14, 0, 2),
        (12, 14, 1, 3),
        (12, 16, 0, 2),
        (12, 16, 1, 3),
        (13, 15, 0, 2),
        (13, 15, 1, 3),
        (13, 14, 0, 2),
    ]
    for d_min, d_max, l_min, l_max in uniform_cases:
        scenarios.append({
            "Scenario": "Uniform",
            "Demand": {"Dist_Type": "UNIFORM", "min": d_min, "max": d_max},
            "Leadtime": {"Dist_Type": "UNIFORM", "min": l_min, "max": l_max}
        })

    # -------- Complex --------
    complex_cases = [
        (11, 13, 1, 1),
        (11, 13, 2, 1),
        (11, 14, 2, 1),
        (11, 14, 1, 1),
        (12, 14, 1, 1),
        (12, 15, 1, 1),
        (12, 14, 2, 1),
        (12, 15, 2, 1),
        (12, 13, 1, 1),
        (13, 15, 1, 1),
        (13, 15, 2, 1),
        (13, 14, 1, 1),
        (13, 14, 1, 1),
        (13, 13, 1, 1),
        (11, 15, 1, 1),
    ]
    for d_min, d_max, l_mean, l_std in complex_cases:
        scenarios.append({
            "Scenario": "Complex",
            "Demand": {"Dist_Type": "UNIFORM", "min": d_min, "max": d_max},
            "Leadtime": {"Dist_Type": "GAUSSIAN", "mean": l_mean, "std": l_std}
        })

    return scenarios



# ============================================================
# 메인 루프
# ============================================================
def run_experiments():
    scenarios = get_scenarios()
    results = []

    case_num = 1
    for s in scenarios:
        print(f"\n=== [CASE {case_num:02}] {s['Scenario']} ===")
        print(f"Demand: {s['Demand']} | Leadtime: {s['Leadtime']}")

        # 환경 생성
        env = GymInterface()
        env.scenario["DEMAND"] = s["Demand"]
        env.scenario["LEADTIME"] = s["Leadtime"]

        # 로그 폴더
        log_dir = os.path.join(EXPERIMENT_LOGS, f"Case_{case_num:02}")
        os.makedirs(log_dir, exist_ok=True)
        env.writer = SummaryWriter(log_dir)

        # PPO 모델 및 콜백
        model = build_model(env)
        rl_callback = make_callback(env)

        # 학습
        model.learn(total_timesteps=SIM_TIME * N_EPISODES, callback=rl_callback)
        # 평균 보상 계산
        avg_reward = float(sum(rl_callback.rewards) / len(rl_callback.rewards)) if rl_callback.rewards else 0

        results.append({
            "Scenario": s["Scenario"],
            "Case": case_num,
            "Demand": str(s["Demand"]),
            "Leadtime": str(s["Leadtime"]),
            "Avg_Reward": avg_reward
        })

        case_num += 1

    # CSV 저장
    df = pd.DataFrame(results)
    csv_path = os.path.join(RESULT_CSV_EXPERIMENT, RESULT_FILE)
    df.to_csv(csv_path, index=False)
    print(f"\n✅ 모든 PPO 시나리오 완료! 결과 저장: {csv_path}")


# ============================================================
# 실행부
# ============================================================
if __name__ == "__main__":
    start_time = time.time()
    run_experiments()
    print(f"\nTotal Execution Time: {time.time() - start_time:.2f} seconds")

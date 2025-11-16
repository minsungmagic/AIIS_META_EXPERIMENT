import numpy as np
from envs.config_SimPy import *
from envs.promp_env import MetaEnv
from AIIS_META.Agents.Gaussian.Meta_Gaussian import MetaGaussianAgent
from AIIS_META.Baselines.linear_baseline import LinearFeatureBaseline
from AIIS_META.Algos.MAML.promp import ProMP
from AIIS_META.Agents.Simple_Mlp import SimpleMLP
import torch
import torch.optim as optim
from envs.config_folders import *


import pandas as pd
import os

from envs.scenarios import get_scenarios
from collections import OrderedDict

class ScenarioEvalCallback:
    """
    각 시나리오에서 학습이 끝난 뒤 reward_history를 받아서
    마지막 window개 reward 평균을 results 리스트에 한 줄로 추가하고,
    시나리오별 요약 정보(분포, demand, leadtime, avg_reward)를 엑셀로 저장.
    """
    def __init__(self, window, scenario_idx, scenario, results_list):
        self.window = window
        self.scenario_idx = scenario_idx
        self.scenario = scenario
        self.results_list = results_list

    def __call__(self, reward_history):
        print(f"[Scenario {self.scenario_idx}] reward_history length = {len(reward_history)}")

        if len(reward_history) < self.window:
            print(f"[Scenario {self.scenario_idx}] reward 개수가 {self.window}개보다 적어서 평균 계산 생략")
            return

        # 마지막 window개 reward로 평균 계산
        last_rewards = reward_history[-self.window:]
        mean_reward = float(np.mean(last_rewards))

        print(f"[Scenario {self.scenario_idx}] 마지막 {self.window}개 reward 평균: {mean_reward:.3f}")

        # ✅ 최종 결과용 리스트에는 네 개 필드만 저장
        self.results_list.append({
            "scenario_type": self.scenario["Scenario"],   # 분포 타입 (Gaussian / Uniform / Complex 등)
            "demand": str(self.scenario["DEMAND"]),       # demand 설정 (dict를 문자열로)
            "leadtime": str(self.scenario["LEADTIME"]),   # leadtime 설정 (dict를 문자열로)
            "avg_reward": mean_reward,                    # 마지막 window 평균
        })

        # ================================
        # 🔥 시나리오 요약 엑셀 저장 부분
        # ================================
        summary_df = pd.DataFrame([{
            "scenario_type": self.scenario["Scenario"],   # 분포 타입
            "demand": str(self.scenario["DEMAND"]),
            "leadtime": str(self.scenario["LEADTIME"]),
            "avg_reward": mean_reward,                    # 마지막 window 평균
        }])

        save_path = os.path.join(
            SAVED_MODEL_PATH,
            f"scenario_{self.scenario_idx}_summary.xlsx"
        )
        summary_df.to_excel(save_path, index=False)
        print(f"[Scenario {self.scenario_idx}] 요약 Excel 저장 완료 → {save_path}")





class EvalCSVCallback:
    """
    학습이 끝난 뒤 reward_history(에폭별 reward 리스트)를 받아서
    마지막 window개 평균을 계산하고 CSV로 저장하는 콜백.
    """
    def __init__(self, window: int = 15, csv_path: str = None):
        self.window = window
        # 기본 경로: SAVED_MODEL_PATH / eval_rewards.csv
        if csv_path is None:
            self.csv_path = os.path.join(SAVED_MODEL_PATH, "eval_rewards.csv")
        else:
            self.csv_path = csv_path

    def __call__(self, reward_history):
        print("========== reward_history ==========")
        print(reward_history)

        if len(reward_history) < self.window:
            print(f"[EvalCSVCallback] reward 개수가 {self.window}개보다 적어서 평균을 낼 수 없습니다.")
            return

        # 마지막 window개의 reward 사용
        last_rewards = reward_history[-self.window:]
        mean_reward = float(np.mean(last_rewards))

        print(f"[EvalCSVCallback] 마지막 {self.window}개 reward 평균: {mean_reward:.3f}")
        print(f"[EvalCSVCallback] CSV 저장 경로: {self.csv_path}")

        # CSV로 저장
        df = pd.DataFrame({
            "mean_reward_last_window": [mean_reward],
            "num_rewards": [len(last_rewards)],
        })

        # 한 번만 쓰면 되니까 항상 덮어쓰기(mode='w')
        df.to_csv(self.csv_path, index=False)

def main(params):

    # 0) 환경과 시나리오 리스트 준비
    env = MetaEnv()
    scenario_list = get_scenarios()  # 네가 정의한 시나리오들

    # 모든 시나리오 결과를 담을 리스트
    all_results = []

    # ===== 시나리오별 루프 =====
    for idx, scenario in enumerate(scenario_list):
        print("\n" + "=" * 80)
        print(f"[{idx+1}/{len(scenario_list)}] Scenario Fine-tuning Start")
        print("Scenario:", scenario)
        print("=" * 80)

        # 1) 해당 시나리오를 환경에 세팅
        env.set_task(scenario)

        # 2) 네트워크 / 에이전트 / 알고리즘 새로 만들기
        mlp = SimpleMLP(
            np.prod(env.observation_space.shape),
            np.prod(env.action_space.shape),
            hidden_layers=params["Layers"]
        )

        # 이 실험은 "해당 시나리오 1개만" 쓰니까 num_tasks=1
        agent = MetaGaussianAgent(
            mlp=mlp,
            num_tasks=1,
            learn_std=params["learn_std"]
        )

        meta_algo = ProMP(
            env=env,
            max_path_length=params["max_path_length"],
            agent=agent,
            alpha=params["alpha"],
            beta=params["beta"],
            baseline=LinearFeatureBaseline(),
            tensor_log=params["tensor_log"],
            inner_grad_steps=params["num_inner_grad"],
            num_tasks=1,  # 여기도 1
            outer_iters=params["outer_iters"],
            parallel=params["parallel"],
            rollout_per_task=params["rollout_per_task"],
            clip_eps=params["clip_eps"],
            device=params["device"]
        )

        # 3) pre-trained 모델 로드
        meta_algo.load_state_dict(torch.load("preTrainModel/saved_model", map_location="cpu")) # 얘를 이용해서 중간에 불러올 수 있음


        # 4) 이 시나리오용 콜백 생성 (마지막 15개 평균)
        cb = ScenarioEvalCallback(
            window=15,
            scenario_idx=idx,
            scenario=scenario,
            results_list=all_results
        )

        # 5) 1000 epoch 파인튜닝
        meta_algo.learn(epochs=params["epochs"], callback=cb)

        # 필요하면 시나리오별로 모델도 따로 저장 가능
        scenario_model_path = os.path.join(SAVED_MODEL_PATH, f"scenario_{idx}_model.pt")
        torch.save(meta_algo.state_dict(), scenario_model_path)

        meta_algo.close()
        print(all_results)

    # ===== 모든 시나리오 결과를 하나의 CSV로 저장 =====
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(SAVED_MODEL_PATH, "scenario_finetune_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n[Done] 모든 시나리오 결과를 {results_path} 에 저장했습니다.")

if __name__ == "__main__":
    
    params = {
        "Layers":[64, 64], # layers of Network
        "rollout_per_task": 20,
        "num_task": 1, # Number of tasks
        "max_path_length": SIM_TIME,
        "tensor_log": TENSORFLOW_LOGS,
        "alpha": 0.002,
        "beta": 0.0005,
        "outer_iters": 5, # number of ProMp steps without re-sampling
        "clip_eps": 0.3, # clip range for ProMP(outer) update
        "num_inner_grad": 1,
        "epochs": 1000,
        "discount": 0.99,
        "gae_lambda": 1,
        "parallel": False,
        "learn_std": True,
        "device":torch.device("cpu")
    }
    
    main(params)
   
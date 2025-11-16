from envs.config_SimPy import *
from envs.config_RL import *
import itertools
import random
import numpy as np
from collections import defaultdict

# 1. Task Sampling Related

def create_scenarios():
    """
    Creates a set of tasks (e.g., environment or scenario definitions).
    
    This function defines the demand and lead time ranges and creates all possible 
    combinations of these parameters for the tasks in the environment.
    
    Returns:
        scenarios (list): A list of dictionaries representing all possible scenarios 
                          combining demand and leadtime configurations.
    """
    # DEMAND
    demand_uniform_range = [
        (i, j)
        for i in range(5, 16)  # Range for demand min values
        for j in range(i, 16)  # Range for demand max values
        if i <= j
    ]
    
    # Define the uniform demand distribution
    demand_uniform = [
        {"Dist_Type": "UNIFORM", "min": min_val, "max": max_val}
        for min_val, max_val in demand_uniform_range
    ]

    # LEADTIME
    leadtime_uniform_range = [
        (i, j)
        for i in range(1, 5)  # Range for lead time min values
        for j in range(i, 5)  # Range for lead time max values
        if i <= j
    ]
    
    # Define the uniform lead time distribution
    leadtime_uniform = [
        {"Dist_Type": "UNIFORM", "min": min_val, "max": max_val}
        for min_val, max_val in leadtime_uniform_range
    ]

    # Create all combinations of demand and leadtime
    scenarios = [{"DEMAND": demand, "LEADTIME": [random.sample(leadtime_uniform, 1)[0] for _ in range(MAT_COUNT)]} for demand in demand_uniform]

    return scenarios

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
        demand_cfg = {
            "Dist_Type": "GAUSSIAN",
            "mean": mean,
            "std": std,
        }

        # 환경에서 lead_time_dict[self.item_id-1]로 접근하므로
        # 0 ~ MAT_COUNT-1 인덱스를 키로 갖는 dict-of-dicts 형태로 맞춰줌
        base_lead_cfg = {
            "Dist_Type": "GAUSSIAN",
            "mean": l_mean,
            "std": l_std,
        }
        leadtime_cfg = {i: base_lead_cfg for i in range(MAT_COUNT)}

        scenarios.append({
            "Scenario": "Gaussian",
            "DEMAND": demand_cfg,
            "LEADTIME": leadtime_cfg,
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
        demand_cfg = {
            "Dist_Type": "UNIFORM",
            "min": d_min,
            "max": d_max,
        }

        base_lead_cfg = {
            "Dist_Type": "UNIFORM",
            "min": l_min,
            "max": l_max,
        }
        leadtime_cfg = {i: base_lead_cfg for i in range(MAT_COUNT)}

        scenarios.append({
            "Scenario": "Uniform",
            "DEMAND": demand_cfg,
            "LEADTIME": leadtime_cfg,
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
        demand_cfg = {
            "Dist_Type": "UNIFORM",
            "min": d_min,
            "max": d_max,
        }

        base_lead_cfg = {
            "Dist_Type": "GAUSSIAN",
            "mean": l_mean,
            "std": l_std,
        }
        leadtime_cfg = {i: base_lead_cfg for i in range(MAT_COUNT)}

        scenarios.append({
            "Scenario": "Complex",
            "DEMAND": demand_cfg,
            "LEADTIME": leadtime_cfg,
        })

    return scenarios

"""
A simple script for testing positional movement
"""
import argparse
import os

import numpy as np
import yaml
from matplotlib import pyplot as plt

import robots
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# positional interpolation
def get_waypoint(start_pt, target_pt, max_delta=0.005):
    total_delta = (target_pt - start_pt)
    num_steps = (np.linalg.norm(total_delta) // max_delta)
    remainder = (np.linalg.norm(total_delta) % max_delta)
    if remainder > 1e-3:
        num_steps += 1
    delta = total_delta / num_steps
    def gen_waypoint(i):
        return start_pt + delta * min(i, num_steps)
    return gen_waypoint, int(num_steps)

# rotation interpolation
def get_ori(initial_euler, final_euler, num_steps):
    ori_chg = R.from_euler("xyz", [initial_euler.copy(), final_euler.copy()], degrees=False)
    slerp = Slerp([1,num_steps], ori_chg)
    def gen_ori(i): 
        interp_euler = slerp(i).as_euler("xyz")
        return interp_euler
    return gen_ori

def move_robot(env, target_pos, target_euler):
    obs = env._get_obs()
    ee_pos = obs['state']['ee_pos']
    ee_quat = obs['state']['ee_quat']
    ee_euler = R.from_quat(ee_quat).as_euler("xyz")

    gen_waypoint, num_steps = get_waypoint(ee_pos, target_pos)
    gen_ori = get_ori(ee_euler, target_euler, num_steps)

    for i in range(1, num_steps+1):
        next_ee_pos = gen_waypoint(i)
        next_ee_euler = gen_ori(i)
        gripper = np.array([0])
        action = np.hstack((next_ee_pos, next_ee_euler, gripper))
        env.step(action)
        #print(np.round(ee_pos, 2), np.round(next_pos, 2))
        #print(np.round(ee_euler, 2), np.round(next_euler, 2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    env = robots.RobotEnv(**config)
    obs, _ = env.reset(reset_controller=True)
    obs = env._get_obs()
    print("[robots] Finished reset.")

    ee_pos = obs['state']['ee_pos']
    ee_quat = obs['state']['ee_quat']
    ee_euler = R.from_quat(ee_quat).as_euler("xyz")

    target_pos = ee_pos + np.array([0.05, 0, -0.05])
    target_euler = ee_euler.copy() + np.array([0, 0, np.pi/4])

    #move_robot(env, target_pos, target_euler)

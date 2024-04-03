import pyrealsense2 as rs
from scipy.spatial.transform import Rotation
import threading
import yaml
import os
import random
import cv2
import open3d as o3d
from vision_utils.pc_utils import *
from vision_utils.calib_utils import detect_calibration_marker, Solver
import argparse
import robots
from scipy.spatial.transform import Rotation as R
from manip_utils.robot_utils import get_waypoint, get_ori

import threading
import math
import ast
import json
import asyncio
import multiprocessing
import websockets
from http.server import HTTPServer, SimpleHTTPRequestHandler
import subprocess
import sys
import time

import psutil
from queue import Queue

def adjust_rotation(angle1, angle2):
    if np.sign(angle1) != np.sign(angle2):
        return angle2
    return angle1

class MyWebSocketHandler:
    def __init__(self):
        self.waypoints = []

    async def websocket_handler(self, websocket, path):
        async for message in websocket:
            message = json.loads(message)
            if len(message):
                for data in message:
                    data['gripper_open'] = data.get('url') == 'http://localhost:8080/robotiq.obj'
                    if 'url' in data: del data['url']
                    fingertip_ui_pos = np.array([data['position']['x'], data['position']['y'], data['position']['z']]).squeeze()
                    rotation = np.array([data['orientation']['x'], data['orientation']['y'], data['orientation']['z']])
                    self.waypoints.append([fingertip_ui_pos[0], fingertip_ui_pos[1], fingertip_ui_pos[2], rotation[0], rotation[1], rotation[2], float(not data['gripper_open'])])

class InteractiveBot:
    def __init__(self, config):
        self.env = robots.RobotEnv(**config)
        obs, _ = self.env.reset(reset_controller=True)
        self.transforms = None
        if os.path.exists('calib/transforms.npy'):
            self.transforms = np.load('calib/transforms.npy', allow_pickle=True).item()
            self.transforms['agent']['tcr'][:,3][2] -= 0.02

        self.HOME_POS = np.array([0.30496958, -0.00216635, 0.57])

    def take_rgbd(self):
        frames = self.env._get_frames()
        rgb_frame = frames['agent_image']
        depth_frame = frames['agent_depth']
        return rgb_frame, depth_frame

    def rgbd2pointCloud(self, rgb_frame, depth_frame):
        depth_frame = depth_frame.squeeze()
        agent_intrinsics = self.env.cameras['agent'].get_intrinsics()['matrix']
        denoised_idxs = denoise(depth_frame)
        if self.transforms is not None:
            tf = self.transforms['agent']['tcr']
        else:
            tf = np.eye(4)
        
        points_3d = deproject(depth_frame, agent_intrinsics, tf)
        colors = rgb_frame.reshape(points_3d.shape)/255.

        points_3d = points_3d[denoised_idxs]
        colors = colors[denoised_idxs]

        idxs = crop(points_3d, min_bound=[0.25, -0.2, -0.03], max_bound=[0.76, 0.2, 0.3])
        points_3d = points_3d[idxs]
        colors = colors[idxs]

        pcd_merged = merge_pcls([points_3d], [colors])
        pcd_merged.remove_duplicated_points()
        return pcd_merged


    def rotate_robot(self, target_euler, num_steps=15):
        obs = self.env._get_obs()
        ee_pos = obs['state']['ee_pos']
        ee_quat = obs['state']['ee_quat']
        ee_euler = R.from_quat(ee_quat).as_euler("xyz")
        gripper_width = obs['state']['gripper_pos']

        gen_ori = get_ori(ee_euler, target_euler, num_steps)
        for i in range(1, num_steps+1):
            next_ee_euler = gen_ori(i)
            action = np.hstack((ee_pos, next_ee_euler, gripper_width))
            self.env.step(action)

    def move_robot_constant(self, target_pos, target_euler, max_delta=0.005):
        obs = self.env._get_obs()
        ee_pos = obs['state']['ee_pos']
        gripper_width = obs['state']['gripper_pos']

        positional_delta = np.linalg.norm(target_pos - ee_pos)

        gen_waypoint, num_steps = get_waypoint(ee_pos, target_pos, max_delta=max_delta)
        for i in range(1, num_steps+1):
            next_ee_pos = gen_waypoint(i)
            action = np.hstack((next_ee_pos, target_euler, gripper_width))
            self.env.step(action)

    def move_robot(self, target_pos, target_euler, max_delta=0.025):
        obs = self.env._get_obs()
        ee_pos = obs['state']['ee_pos']
        ee_quat = obs['state']['ee_quat']
        ee_euler = R.from_quat(ee_quat).as_euler("xyz")
        gripper_width = obs['state']['gripper_pos']
    

        positional_delta = np.linalg.norm(target_pos - ee_pos)
        rotational_delta = np.linalg.norm(target_euler - ee_euler)

        gen_waypoint, num_steps = get_waypoint(ee_pos, target_pos, max_delta=max_delta)
        gen_ori = get_ori(ee_euler, target_euler, num_steps)

        data = {'obs':[], 'action': []}
        for i in range(1, num_steps+1):
            next_ee_pos = gen_waypoint(i)
            next_ee_euler = gen_ori(i)
            action = np.hstack((next_ee_pos, next_ee_euler, gripper_width))
            self.env.step(action)
            rgb_frame, _ = self.take_rgbd()

            obs = self.env._get_obs()
            ee_pos = obs['state']['ee_pos']
            ee_quat = obs['state']['ee_quat']
            ee_euler = R.from_quat(ee_quat).as_euler("xyz")
            gripper_width = obs['state']['gripper_pos']

            data['obs'].append(rgb_frame)
            data['action'].append([ee_pos[0], ee_pos[1], ee_pos[2], ee_euler[0], ee_euler[1], ee_euler[2], gripper_width])
        return data

    def close_gripper(self):
        obs = self.env._get_obs()
        ee_pos = obs['state']['ee_pos']
        ee_quat = obs['state']['ee_quat']
        ee_euler = R.from_quat(ee_quat).as_euler("xyz")
        gripper_width = obs['state']['gripper_pos']
        action = np.hstack((ee_pos, ee_euler, [1.0]))
        self.env.step(action)
        time.sleep(0.5)

    def open_gripper(self):
        obs = self.env._get_obs()
        ee_pos = obs['state']['ee_pos']
        ee_quat = obs['state']['ee_quat']
        ee_euler = R.from_quat(ee_quat).as_euler("xyz")
        gripper_width = obs['state']['gripper_pos']
        action = np.hstack((ee_pos, ee_euler, [0.0]))
        self.env.step(action)
        time.sleep(0.5)

    def calculate_fingertip_offset(self, ee_euler):
        home_fingertip_offset = np.array([0, 0, -0.17])
        ee_euler_adjustment = ee_euler.copy() - np.array([-np.pi, 0, 0])
        fingertip_offset = R.from_euler("xyz", ee_euler_adjustment).as_matrix() @ home_fingertip_offset
        return fingertip_offset

    def transform_robotframe_to_uiframe(self, waypoints):
        waypoints = np.array(waypoints)
        waypoints_ui = np.zeros_like(np.array(waypoints))
        transf = R.from_euler('x', -90, degrees=True)
        waypoints_ui = transf.apply(waypoints)
        rescale_amt = 10
        waypoints_ui *= rescale_amt
        return waypoints_ui

    def transform_uiframe_to_robotframe(self, waypoints):
        waypoints_rob = np.array(waypoints.copy())
        waypoints_rob /= 10
        transf = R.from_euler('x', 90, degrees=True)
        waypoints_rob = transf.apply(waypoints_rob)
        return waypoints_rob

    def reset(self):
        #_, _ = self.env.reset(reset_controller=True)
        self.open_gripper()
        target_euler = np.array([np.pi, 0, 0])
        self.move_robot(self.HOME_POS, target_euler)

    def record_demo(self, demo_folder):
        obs = self.env._get_obs()
        ee_pos = obs['state']['ee_pos']
        ee_quat = obs['state']['ee_quat']
        ee_euler = R.from_quat(ee_quat).as_euler("xyz")
        fingertip_pos = ee_pos.copy() + self.calculate_fingertip_offset(ee_euler)
        gripper_closed = obs['state']['gripper_pos'] > 0.5

        fingertip_pos_ui = self.transform_robotframe_to_uiframe(fingertip_pos.reshape(1, 3)).squeeze()
        ee_euler_ui = np.zeros(3)

        fingertip_pos_code = 'new THREE.Vector3(%.2f, %.2f, %.2f);\n'%(fingertip_pos_ui[0], fingertip_pos_ui[1], fingertip_pos_ui[2])
        ee_euler_code = 'new THREE.Euler(%.2f, %.2f, %.2f);\n'%(ee_euler_ui[0], ee_euler_ui[1], ee_euler_ui[2])

        rgb_frame, depth_frame = self.take_rgbd()
        pointcloud = self.rgbd2pointCloud(rgb_frame, depth_frame)

        idxs = np.random.choice(np.arange(len(pointcloud.points)), 3000, replace=False)
        points = np.asarray(pointcloud.points)[idxs]
        colors = np.asarray(pointcloud.colors)[idxs]
        points_ui = self.transform_robotframe_to_uiframe(points)
        pointcloud_points_code = '[\n' + ',\n'.join(['%.2f, %.2f, %.2f'%(pos[0], pos[1], pos[2]) for pos in points_ui]) + '\n];'
        pointcloud_colors_code = '[\n' + ',\n'.join(['%.2f, %.2f, %.2f'%(color[0], color[1], color[2]) for color in colors]) + '\n];'

        with open('interactive_scripts/interactive_utils/template_demo.html') as f:
            html_content = f.read()

        html_content = html_content%(pointcloud_points_code, pointcloud_colors_code, fingertip_pos_code, ee_euler_code)
        with open('interactive_scripts/interactive_utils/index.html', 'w') as f:
            f.write(html_content)

        server_process = subprocess.Popen([sys.executable, "interactive_scripts/interactive_utils/serve.py"])

        # Initialize the handler with the shared array
        handler = MyWebSocketHandler()
        
        # Define the server coroutine
        async def server_coroutine():
            async with websockets.serve(handler.websocket_handler, "localhost", 8765):
                await asyncio.Future()  # Run indefinitely
        
        # Function to start the asyncio event loop in a separate thread
        def start_asyncio_event_loop():
            asyncio.run(server_coroutine())
        
        # Start the event loop in a separate thread
        loop_thread = threading.Thread(target=start_asyncio_event_loop, daemon=True)
        loop_thread.start()

        angle_offset = ee_euler.copy()[0]
        
        print('Start!')
        while not len(handler.waypoints):
            continue

        input('Replay demo?')
        MAX_DELTA = 0.02

        idx = len(os.listdir(demo_folder))

        demonstration = {}
        for pose in handler.waypoints:
            closed = pose[-1]
            fingertip_pos_ui = pose[:3]
            ee_pos_cmd = self.transform_uiframe_to_robotframe(np.array(fingertip_pos_ui).reshape(1, 3)).squeeze()
            ee_pos_cmd -= self.calculate_fingertip_offset(ee_euler)

            ee_euler_cmd = [pose[3]+angle_offset, -pose[5], pose[4]]
            ee_euler_cmd[0] = adjust_rotation(ee_euler_cmd[0], ee_euler[0])

            demonstration.update(self.move_robot(ee_pos_cmd, ee_euler_cmd, max_delta=MAX_DELTA))

            if closed:
                self.close_gripper()
            else:
                self.open_gripper()

            time.sleep(0.5)

        demonstration['obs'] = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in demonstration['obs']]
        np.savez(os.path.join(demo_folder, '%05d.npz'%idx), demonstration)
        self.reset()
        exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    robot = InteractiveBot(config)

    demo_folder = 'demos'
    if not os.path.exists(demo_folder):
        os.mkdir(demo_folder)

    robot.record_demo(demo_folder)

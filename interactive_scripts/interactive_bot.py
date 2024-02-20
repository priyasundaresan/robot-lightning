import pyrealsense2 as rs
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

class InteractiveBot:
    def __init__(self, config):
        self.env = robots.RobotEnv(**config)
        obs, _ = self.env.reset(reset_controller=True)
        self.transforms = None
        if os.path.exists('calib/transforms.npy'):
            self.transforms = np.load('calib/transforms.npy', allow_pickle=True).item()

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
        pcd_merged = merge_pcls([points_3d], [colors])
        pcd_merged.remove_duplicated_points()
        return pcd_merged

    def move_robot(self, target_pos, target_euler, max_delta=0.005):
        obs = self.env._get_obs()
        ee_pos = obs['state']['ee_pos']
        ee_quat = obs['state']['ee_quat']
        ee_euler = R.from_quat(ee_quat).as_euler("xyz")
        gripper_width = obs['state']['gripper_pos']
    
        gen_waypoint, num_steps = get_waypoint(ee_pos, target_pos, max_delta=max_delta)
        gen_ori = get_ori(ee_euler, target_euler, num_steps)

        for i in range(1, num_steps+1):
            next_ee_pos = gen_waypoint(i)
            next_ee_euler = gen_ori(i)
            action = np.hstack((next_ee_pos, next_ee_euler, gripper_width))
            self.env.step(action)

    def close_gripper(self):
        obs = self.env._get_obs()
        ee_pos = obs['state']['ee_pos']
        ee_quat = obs['state']['ee_quat']
        ee_euler = R.from_quat(ee_quat).as_euler("xyz")
        gripper_width = obs['state']['gripper_pos']
        action = np.hstack((ee_pos, ee_euler, [1.0]))
        self.env.step(action)
        time.sleep(1.5)

    def open_gripper(self):
        obs = self.env._get_obs()
        ee_pos = obs['state']['ee_pos']
        ee_quat = obs['state']['ee_quat']
        ee_euler = R.from_quat(ee_quat).as_euler("xyz")
        gripper_width = obs['state']['gripper_pos']
        action = np.hstack((ee_pos, ee_euler, [0.0]))
        self.env.step(action)
        time.sleep(1.5)

    def calculate_fingertip_offset(self, ee_euler):
        home_fingertip_offset = np.array([0, 0, -0.17])
        ee_euler_adjustment = ee_euler.copy() - np.array([-np.pi, 0, 0])
        fingertip_offset = R.from_euler("xyz", ee_euler_adjustment).as_matrix() @ home_fingertip_offset
        return fingertip_offset

    def test_calibration(self):
        obs = self.env._get_obs()
        ee_pos = obs['state']['ee_pos']
        ee_quat = obs['state']['ee_quat']
        ee_euler = R.from_quat(ee_quat).as_euler("xyz")

        def gen_calib_waypoints(start_pos):
            waypoints = []
            for i in np.linspace(0.05,0.15,3):
                for j in np.linspace(-0.1,0.1,3):
                    for k in np.linspace(-0.35,-0.15,3):
                        waypoints.append(start_pos + [i,j,k])
            return waypoints
        
        waypoints = gen_calib_waypoints(ee_pos)

        for waypoint in waypoints:
            target_euler = ee_euler.copy() + np.random.uniform(-np.pi/12, np.pi/12, 3)
            new_ee_pos = waypoint - self.calculate_fingertip_offset(target_euler)

            #self.move_robot(waypoint, target_euler)
            self.move_robot(new_ee_pos, target_euler)
            rgb_frame, depth_frame = self.take_rgbd()
            pointcloud = self.rgbd2pointCloud(rgb_frame, depth_frame)
            # Create a sphere
            radius = 0.02
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            yellow_color = [1.0, 1.0, 0.0]  # Yellow color (R, G, B)
            num_vertices = len(sphere.vertices)
            sphere.vertex_colors = o3d.utility.Vector3dVector([yellow_color] * num_vertices)
            sphere.translate(waypoint)

            # Create a visualizer object
            vis = o3d.visualization.Visualizer()
            # Add geometries to the visualizer
            geometry_list = [pointcloud, sphere]
            o3d.visualization.draw_geometries(geometry_list)

    def run_calibration(self):
        obs = self.env._get_obs()
        ee_pos = obs['state']['ee_pos']
        ee_quat = obs['state']['ee_quat']
        ee_euler = R.from_quat(ee_quat).as_euler("xyz")

        def gen_calib_waypoints(start_pos):
            waypoints = []
            for i in np.linspace(0.05,0.15,3):
                for j in np.linspace(-0.1,0.1,3):
                    for k in np.linspace(-0.25,-0.05,3):
                        waypoints.append(start_pos + [i,j,k])
            return waypoints
        
        waypoints = gen_calib_waypoints(ee_pos)
        #print(waypoints)

        solver = Solver()

        self.open_gripper()
        input('Ready to close gripper?')
        self.close_gripper()

        agent_intrinsics = self.env.cameras['agent'].get_intrinsics()['matrix']

        waypoints_cam = []
        waypoints_rob = []
        transforms = {}

        for waypoint in waypoints:
            self.move_robot(waypoint, ee_euler)
            rgb_frame, depth_frame = self.take_rgbd()
            vis, (u,v) = detect_calibration_marker(rgb_frame)
            cam_point = deproject_pixels(np.array([[u,v]]), depth_frame.squeeze(), agent_intrinsics)[0]
            waypoints_cam.append(cam_point)
            #waypoints_rob.append(waypoint)
            waypoint_fingertip = waypoint + self.calculate_fingertip_offset(ee_euler.copy())
            waypoints_rob.append(waypoint_fingertip)
            cv2.imshow('img', vis)
            cv2.waitKey(200)

        trc, tcr = solver.solve_transforms(np.array(waypoints_rob), np.array(waypoints_cam))
        transforms['agent'] = {'trc':trc, 'tcr':tcr}

        if not os.path.exists('calib'):
            os.mkdir('calib')
        np.save('calib/transforms.npy', transforms)

    def test_ui(self):
        with open('interactive_scripts/interactive_utils/with_guides.html') as f:
            html_content = f.read()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    robot = InteractiveBot(config)
    #rgb_frame, depth_frame = robot.take_rgbd()
    #pointcloud = robot.rgbd2pointCloud(rgb_frame, depth_frame)

    #robot.run_calibration()
    robot.test_calibration()
    #robot.test_ui()

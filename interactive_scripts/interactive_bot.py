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

        idxs = crop(points_3d)
        points_3d = points_3d[idxs]
        colors = colors[idxs]

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
            target_ee_euler = ee_euler.copy() + np.random.uniform(-np.pi/12, np.pi/12, 3)
            fingertip_offset = self.calculate_fingertip_offset(target_ee_euler)
            target_ee_pos = waypoint - fingertip_offset

            self.move_robot(target_ee_pos, target_ee_euler)
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

        if not os.path.exists('calib'):
            os.mkdir('calib')

        for idx, waypoint in enumerate(waypoints):
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
            cv2.imwrite('calib/%05d.jpg'%idx, vis)

        trc, tcr = solver.solve_transforms(np.array(waypoints_rob), np.array(waypoints_cam))
        transforms['agent'] = {'trc':trc, 'tcr':tcr}

        np.save('calib/transforms.npy', transforms)

    def transform_waypoints_to_uiframe(self, waypoints):
        waypoints = np.array(waypoints)
        waypoints_ui = np.zeros_like(np.array(waypoints))
        transf = R.from_euler('x', -90, degrees=True)
        waypoints_ui = transf.apply(waypoints)
        #waypoints_ui[:,0] = waypoints[:,0]
        #waypoints_ui[:,1] = waypoints[:,2]
        #waypoints_ui[:,2] = waypoints[:,1]
        waypoints_ui *= 10
        return waypoints_ui

    def test_ui(self):
        obs = self.env._get_obs()
        ee_pos = obs['state']['ee_pos']
        ee_quat = obs['state']['ee_quat']
        ee_euler = R.from_quat(ee_quat).as_euler("xyz")

        rgb_frame, depth_frame = self.take_rgbd()
        pointcloud = self.rgbd2pointCloud(rgb_frame, depth_frame)
        idxs = np.random.choice(np.arange(len(pointcloud.points)), 5000, replace=False)
        points = np.asarray(pointcloud.points)[idxs]
        colors = np.asarray(pointcloud.colors)[idxs]
        points_ui = self.transform_waypoints_to_uiframe(points)
        pointcloud_points_code = '[\n' + ',\n'.join(['%.2f, %.2f, %.2f'%(pos[0], pos[1], pos[2]) for pos in points_ui]) + '\n];'
        pointcloud_colors_code = '[\n' + ',\n'.join(['%.2f, %.2f, %.2f'%(color[0], color[1], color[2]) for color in colors]) + '\n];'

        waypoints = []
        waypoints_fingertip = []
        for i in np.linspace(0.05,0.35,5):
            waypoints.append(ee_pos + [i,0,-0.1])

        for waypoint in waypoints:
            target_ee_euler = ee_euler.copy()
            fingertip_offset = self.calculate_fingertip_offset(target_ee_euler)
            waypoints_fingertip.append(waypoint + fingertip_offset)
            #self.move_robot(waypoint, target_ee_euler)
        
        with open('interactive_scripts/interactive_utils/template.html') as f:
            html_content = f.read()

        waypoints_fingertip_ui = self.transform_waypoints_to_uiframe(waypoints_fingertip)
        waypoints_fingertip_code = '[\n' + '\n,'.join(['{x: %.2f, y: %.2f, z: %.2f}'%(pos[0], pos[1], pos[2]) for pos in waypoints_fingertip_ui]) + '\n];'
                                      
        html_content = html_content%(len(waypoints), waypoints_fingertip_code, pointcloud_points_code, pointcloud_colors_code)
        with open('interactive_scripts/interactive_utils/tmp.html', 'w') as f:
            f.write(html_content)
        #os.system('firefox interactive_scripts/interactive_utils/tmp.html')

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
    #robot.test_calibration()
    robot.test_ui()

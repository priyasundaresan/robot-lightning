import pyrealsense2 as rs
import numpy as np
import cv2
from cv2 import aruco

class MarkSearch:

    def __init__(self):
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters =  aruco.DetectorParameters()

        self.detector = aruco.ArucoDetector(self.dictionary, self.parameters)

    def find_marker(self, image):
        """
        Obtain marker id list from still image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray)
        
        if ids is None:
            return (None,None),None
        
        ids = ids.flatten()

        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            u = np.mean((topLeft[0], bottomRight[0]))
            v = np.mean((topLeft[1], bottomRight[1]))

            cv2.circle(image, (int(u),int(v)), 5, (0,0,255), -1)
            cv2.imshow("rgb", image)
            cv2.waitKey(1)

            return (u,v), image

class RealsenseStreamer():
    def __init__(self, serial_no=None):

        # LEFT SERIAL NO: 145422071576
        # RIGHT SERIAL NO: 241222076578

        # Configure depth and color streams
        self.pipeline = rs.pipeline()


        self.config = rs.config()

        if serial_no is not None:
            self.config.enable_device(serial_no)

        self.width = 640
        self.height = 480

        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)

        self.align_to_color = rs.align(rs.stream.color)

        # Start streaming
        self.pipe_profile = self.pipeline.start(self.config)

        profile = self.pipeline.get_active_profile()

        ## Configure depth sensor settings
        depth_sensor = profile.get_device().query_sensors()[0]
        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
        depth_sensor.set_option(rs.option.depth_units, 0.001)
        preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
        for i in range(int(preset_range.max)):
            visualpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
            if visualpreset == "Default":
                depth_sensor.set_option(rs.option.visual_preset, i)

        color_sensor = profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)

        self.serial_no = serial_no

        # Intrinsics & Extrinsics
        frames = self.pipeline.wait_for_frames()
        frames = self.align_to_color.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

        self.depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
        self.colorizer = rs.colorizer()

        self.K = np.array([[self.depth_intrin.fx, 0, self.depth_intrin.ppx],
                           [0, self.depth_intrin.fy, self.depth_intrin.ppy],
                           [0, 0, 1]])

        self.dec_filter = rs.decimation_filter()
        self.spat_filter = rs.spatial_filter()
        self.temp_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter()

    def deproject(self, px, depth_frame):
        u,v = px
        depth = depth_frame.get_distance(u,v)
        xyz = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [u,v], depth)
        return xyz

    def capture_rgb(self):
        color_frame = None
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame is not None:
                color_image = np.asanyarray(color_frame.get_data())
                break
        return color_image

    def filter_depth(self, depth_frame):
        filtered = depth_frame
        #filtered = self.hole_filling_filter.process(filtered)
        #filtered = self.temp_filter.process(filtered)
        #filtered = self.spat_filter.process(filtered)
        return filtered.as_depth_frame()

    def capture_rgbd(self):
        frame_error = True
        while frame_error:
            try:
                frames = self.align_to_color.process(frames)  
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                frame_error = False
            except:
                frames = self.pipeline.wait_for_frames()
                continue
        color_image = np.asanyarray(color_frame.get_data())
        depth_frame = self.filter_depth(depth_frame)
        depth_image = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
        return color_frame, color_image, depth_frame, depth_image

    def stop_stream(self):
        self.pipeline.stop()

    def show_image(self, image):
        cv2.imshow('img', image)
        cv2.waitKey(0)

if __name__ == '__main__':
    #realsense_streamer = RealsenseStreamer()
    realsense_streamer = RealsenseStreamer('042222070680')
    #realsense_streamer = RealsenseStreamer('241222076578')
    marker_search = MarkSearch()

    frames = []
    while True:
        _, rgb_image, depth_frame, depth_img = realsense_streamer.capture_rgbd()
        cv2.imshow('img', np.hstack((depth_img, rgb_image)))
        cv2.waitKey(1)


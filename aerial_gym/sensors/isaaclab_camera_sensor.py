from aerial_gym.sensors.base_sensor import BaseSensor
import torch
import numpy as np
from aerial_gym.utils.logging import CustomLogger

try:
    from omni.isaac.lab.sensors import Camera, CameraCfg
    from omni.isaac.lab.utils import math as math_utils
except ImportError:
    pass

logger = CustomLogger("IsaacLabCameraSensor")

class IsaacLabCameraSensor(BaseSensor):
    """
    Camera sensor class for Isaac Lab. Inherits from BaseSensor.
    Supports depth, segmentation, and RGB images using Isaac Lab's Camera sensor.
    """

    def __init__(self, sensor_config, num_envs, device):
        super().__init__(sensor_config=sensor_config, num_envs=num_envs, device=device)
        self.device = device
        self.num_envs = num_envs
        self.cfg = sensor_config
        
        logger.info("Initializing Isaac Lab Camera Sensor")
        self.cameras = []

    def add_sensor_to_env(self, env_id, env_handle, actor_handle):
        from omni.isaac.lab.sim.spawners.sensors import spawn_camera, PinholeCameraCfg
        from aerial_gym.utils.math import quat_from_euler_xyz
        
        for i in range(self.cfg.num_sensors):
            prim_path = f"/World/Env_{env_id}/Robot/Camera_Sensor_{i}"
            
            # Orientation conversion
            euler_deg = self.cfg.nominal_orientation_euler_deg # [x, y, z] in degrees
            # If multiple sensors have different configs, we should index them.
            # aerial_gym config usually implies identical sensors or handled via list?
            # Looking at IGE implementation, it seems to assume same config for all 'num_sensors' unless overridden?
            # Actually IGE implementation loops `self.cfg.num_sensors` but uses `self.cfg` properties.
            # So identical sensors.
            
            euler_rad = torch.tensor(euler_deg, device=self.device) * torch.pi / 180.0
            q = quat_from_euler_xyz(euler_rad[0], euler_rad[1], euler_rad[2])
            orientation = np.array([q[3].item(), q[0].item(), q[1].item(), q[2].item()])
            
            spawn_cfg = PinholeCameraCfg(
                func=spawn_camera,
                update_period=0,
                width=self.cfg.width,
                height=self.cfg.height,
                offset=self.cfg.nominal_position,
            )
            
            spawn_camera(
                prim_path=prim_path,
                cfg=spawn_cfg,
                translation=self.cfg.nominal_position,
                orientation=orientation
            )

    def init_tensors(self, global_tensor_dict):
        super().init_tensors(global_tensor_dict)
        self.rgb_pixels = global_tensor_dict.get("rgb_pixels")
        self.depth_pixels = global_tensor_dict.get("depth_range_pixels")
        self.segmentation_pixels = global_tensor_dict.get("segmentation_pixels")
        
        # Now that all prims are spawned, create the Camera View
        # We assume a regex path matching all sensors in all envs
        camera_cfg = CameraCfg(
            prim_path="/World/Env_.*/Robot/Camera_Sensor_.*",
            update_period=0, # Update every step
            height=self.cfg.height,
            width=self.cfg.width,
            data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"], 
            spawn=None, # Already spawned
        )
        
        self.camera_view = Camera(camera_cfg)
        self.camera_view.reset()

    def update(self):
        # Update view
        self.camera_view.update(dt=0.0)
        
        # Data shape from view: (num_envs * num_sensors, H, W, C)
        # We need to reshape to (num_envs, num_sensors, H, W, C) first, then permute if needed.
        
        if self.depth_pixels is not None:
            data = self.camera_view.data.output["distance_to_image_plane"]
            # (N*S, H, W, 1) -> (N, S, H, W, 1)
            data = data.view(self.num_envs, self.cfg.num_sensors, self.cfg.height, self.cfg.width, -1)
            # -> (N, S, C, H, W) -> (N, S, H, W) since C=1 and we want to remove it for depth usually?
            # aerial_gym expects (N, S, H, W).
            self.depth_pixels[:] = data.squeeze(-1)
            
            # Post-processing for depth
            self.apply_noise(self.depth_pixels)
            self.apply_range_limits(self.depth_pixels)
            self.normalize_observation(self.depth_pixels)
            
        if self.rgb_pixels is not None:
             data = self.camera_view.data.output["rgb"]
             # (N*S, H, W, 3) -> (N, S, H, W, 3)
             data = data.view(self.num_envs, self.cfg.num_sensors, self.cfg.height, self.cfg.width, -1)
             # aerial_gym expects (N, S, H, W, 3)? Or (N, S, 3, H, W)?
             # IGE was (N, S, H, W, 4).
             # Let's assume (N, S, H, W, 3) for now based on IGE logic 'self.rgb_pixels[env_id, cam_id] = ...'
             # If target is 4 channels, we need to handle that.
             if self.rgb_pixels.shape[-1] == 4:
                 # Add alpha channel
                 self.rgb_pixels[..., :3] = data
                 self.rgb_pixels[..., 3] = 1.0 # Alpha
             else:
                 self.rgb_pixels[:] = data

        if self.segmentation_pixels is not None:
             data = self.camera_view.data.output["semantic_segmentation"]
             # (N*S, H, W, 1) -> (N, S, H, W)
             data = data.view(self.num_envs, self.cfg.num_sensors, self.cfg.height, self.cfg.width, -1).squeeze(-1)
             self.segmentation_pixels[:] = data.int()

    def apply_range_limits(self, pixels):
        if self.cfg.max_range > 0:
            pixels[pixels > self.cfg.max_range] = self.cfg.far_out_of_range_value
        if self.cfg.min_range > 0:
            pixels[pixels < self.cfg.min_range] = self.cfg.near_out_of_range_value

    def normalize_observation(self, pixels):
        if self.cfg.normalize_range and not self.cfg.pointcloud_in_world_frame:
            pixels[:] = pixels / self.cfg.max_range

    def apply_noise(self, pixels):
        if self.cfg.sensor_noise.enable_sensor_noise:
            pixels[:] = torch.normal(
                mean=pixels, std=self.cfg.sensor_noise.pixel_std_dev_multiplier * pixels
            )
            # Dropout
            dropout_mask = torch.bernoulli(torch.ones_like(pixels) * self.cfg.sensor_noise.pixel_dropout_prob)
            pixels[dropout_mask > 0] = self.cfg.near_out_of_range_value

    def reset_idx(self, env_ids):
        # Camera is attached to the robot, so resetting the robot resets the camera pose.
        # No separate state maintenance required for the sensor itself.
        pass

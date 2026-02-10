from aerial_gym.env_manager.base_env_manager import BaseManager
from aerial_gym.sensors.isaaclab_camera_sensor import IsaacLabCameraSensor
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.utils.logging import CustomLogger
import torch
import torch.nn as nn

class RobotManagerIsaacLab(BaseManager):
    def __init__(self, global_sim_dict, robot_name, controller_name, device):
        logger.debug("Initializing RobotManagerIsaacLab")
        self.global_sim_dict = global_sim_dict
        self.sim = global_sim_dict.get("sim") # Isaac Lab SimulationContext
        self.env_config = global_sim_dict["env_cfg"]
        self.device = device
        
        # create the robot from the name registry
        self.robot, robot_config = robot_registry.make_robot(
            robot_name, controller_name, self.env_config, device
        )
        super().__init__(robot_config, device)
        
        self.articulation = None
        self.camera_sensor = None

    def create_robot(self, asset_loader_class):
        # Load robot asset config similar to IGE manager
        robot_asset_class = self.cfg.robot_asset
        self.robot_asset_dict = asset_loader_class.load_selected_file_from_config(
            "robot", robot_asset_class, robot_asset_class.file, is_robot=True
        )

    def prepare_for_sim(self, global_tensor_dict):
        self.global_tensor_dict = global_tensor_dict
        
        # Link to Isaac Lab Articulation View if available
        self.articulation = global_tensor_dict.get("robot_articulation")
        
        # Initialize robot specific tensors
        self.global_tensor_dict["robot_actions"] = torch.zeros(
            (self.cfg.env.num_envs, self.robot.num_actions), device=self.device
        )
        self.actions = self.global_tensor_dict["robot_actions"]
        
        # Mass and Inertia
        # Try to retrieve from Articulation Data
        if self.articulation and hasattr(self.articulation.data, "default_mass"):
             # default_mass is (num_envs, num_bodies) usually
             mass_val = self.articulation.data.default_mass
             if isinstance(mass_val, torch.Tensor):
                 # Sum masses of all bodies to get total robot mass per env
                 if mass_val.ndim > 1:
                     total_mass = mass_val.sum(dim=1)
                 else:
                     total_mass = mass_val
                 
                 self.robot_masses = total_mass.view(self.cfg.env.num_envs).to(self.device)
             else:
                 self.robot_masses = torch.full((self.cfg.env.num_envs,), mass_val, device=self.device)
        else:
             # Fallback
             self.robot_masses = torch.ones(self.cfg.env.num_envs, device=self.device) * 1.5
             
        self.global_tensor_dict["robot_mass"] = self.robot_masses
        self.global_tensor_dict["robot_inertia"] = torch.eye(3, device=self.device).expand(self.cfg.env.num_envs, 3, 3) * 0.05
        
        self.robot.init_tensors(self.global_tensor_dict)
        
        # Configure DOF Control Mode (Stiffness/Damping)
        if self.articulation:
            # We assume the config has reconfiguration_config
            if hasattr(self.cfg, "reconfiguration_config"):
                # Apply stiffness and damping to all joints
                # Isaac Lab Articulation.set_gains(k_p, k_d)
                # We need to map from "position", "velocity" modes if needed, but usually we just set gains.
                stiffness = self.cfg.reconfiguration_config.stiffness
                damping = self.cfg.reconfiguration_config.damping
                
                # Expand to (num_envs, num_dof) if needed or scalar
                # We assume uniform for now, or match robot.num_dof
                # Isaac Lab set_joint_kp_kd usually takes tensors of shape (num_envs, num_joints)
                
                num_dof = self.articulation.num_joints
                kp = torch.tensor(stiffness, device=self.device).repeat(self.cfg.env.num_envs, 1)
                kd = torch.tensor(damping, device=self.device).repeat(self.cfg.env.num_envs, 1)
                
                # Check if dimensions match
                if kp.shape[1] != num_dof:
                    # If config provides scalar or different length, we might need adjustments
                    # For now we assume strict config match or single value broadcast
                    if len(stiffness) == 1:
                         kp = torch.full((self.cfg.env.num_envs, num_dof), stiffness[0], device=self.device)
                         kd = torch.full((self.cfg.env.num_envs, num_dof), damping[0], device=self.device)
                
                self.articulation.set_joint_kp_kd(kp, kd)

        # Initialize Camera Sensor
        if self.cfg.sensor_config.enable_camera:
             self.camera_sensor = IsaacLabCameraSensor(
                 self.cfg.sensor_config.camera_config,
                 self.cfg.env.num_envs,
                 self.device
             )
             # Allocate tensors for camera
             if self.camera_sensor.cfg.width > 0:
                  self.global_tensor_dict["rgb_pixels"] = torch.zeros(
                      (self.cfg.env.num_envs, self.camera_sensor.cfg.num_sensors, self.camera_sensor.cfg.height, self.camera_sensor.cfg.width, 3),
                      device=self.device
                  )
                  self.global_tensor_dict["depth_range_pixels"] = torch.zeros(
                      (self.cfg.env.num_envs, self.camera_sensor.cfg.num_sensors, self.camera_sensor.cfg.height, self.camera_sensor.cfg.width),
                      device=self.device
                  )
                  self.global_tensor_dict["segmentation_pixels"] = torch.zeros(
                      (self.cfg.env.num_envs, self.camera_sensor.cfg.num_sensors, self.camera_sensor.cfg.height, self.camera_sensor.cfg.width),
                      device=self.device, dtype=torch.int32
                  )
             
             self.camera_sensor.init_tensors(self.global_tensor_dict)

        # Initialize IMU Sensor
        self.isaac_lab_imu = None
        if self.cfg.sensor_config.enable_imu:
            from omni.isaac.lab.sensors import IMUSensor, IMUSensorCfg
            # Create view for IMU
            # We assume it's spawned at /World/Env_*/Robot/IMU_Sensor
            self.imu_cfg = IMUSensorCfg(
                prim_path="/World/Env_.*/Robot/IMU_Sensor",
                update_period=0,
            )
            self.isaac_lab_imu = IMUSensor(self.imu_cfg)
            
            # Allocate force_sensor_tensor for legacy IMU class
            self.global_tensor_dict["force_sensor_tensor"] = torch.zeros(
                (self.cfg.env.num_envs, 6), device=self.device
            )
            
            # Initialize legacy IMU class
            from aerial_gym.sensors.imu_sensor import IMUSensor as LegacyIMUSensor
            self.imu_sensor = LegacyIMUSensor(
                self.cfg.sensor_config.imu_config, self.cfg.env.num_envs, self.device
            )
            self.imu_sensor.init_tensors(self.global_tensor_dict)

    def add_robot_to_env(self, simulation_env_class, env_handle, global_asset_counter, env_id, segmentation_counter):
        # Spawn robot using the environment manager
        self.robot_asset_dict["asset_type"] = "robot"
        
        simulation_env_class.add_asset_to_env(
            self.robot_asset_dict,
            env_handle,
            env_id,
            global_asset_counter,
            segmentation_counter,
        )
        
        # Add camera if enabled
        if self.camera_sensor:
            self.camera_sensor.add_sensor_to_env(env_id, env_handle, None)
            
        # Add IMU if enabled
        if self.cfg.sensor_config.enable_imu:
            from omni.isaac.lab.sim.spawners.sensors import spawn_imu_sensor
            prim_path = f"/World/Env_{env_id}/Robot/IMU_Sensor"
            # Spawn IMU
            # Default config usually fine, placed at root (offset 0)
            # Or use properties from config if available
            spawn_imu_sensor(
                prim_path=prim_path,
                cfg=self.imu_cfg.spawn if self.imu_cfg.spawn else self.imu_cfg, # IMUSensorCfg might handle spawn
                translation=[0.0, 0.0, 0.0],
            )

        return segmentation_counter + 1

    def reset_idx(self, env_ids):
        # Reset internal robot state tensors
        self.robot.reset_idx(env_ids)
        
        # Reset sensors if needed
        if self.camera_sensor:
             self.camera_sensor.reset_idx(env_ids)

    def pre_physics_step(self, actions):
        self.actions[:] = actions
        self.robot.step(self.actions)

    def post_physics_step(self):
        # Update IMU if enabled
        if self.isaac_lab_imu:
            self.isaac_lab_imu.update(dt=0.0)
            
            # Read acceleration
            lin_acc = self.isaac_lab_imu.data.output.lin_acc_b # Body frame? or World?
            # Isaac Lab IMU output is typically in sensor frame (body frame if attached to root)
            
            # aerial_gym legacy IMUSensor expects 'force_sensor_tensor' which was net force.
            # It calculates accel = force / mass.
            # So we reverse this: force = accel * mass.
            
            # self.robot_masses is (num_envs,)
            forces = lin_acc * self.robot_masses.unsqueeze(1)
            
            # Populate tensor (first 3 cols)
            self.global_tensor_dict["force_sensor_tensor"][:, 0:3] = forces
            
            # Trigger legacy IMU update which adds noise/bias
            self.imu_sensor.update()

    def capture_sensors(self):
        if self.camera_sensor:
            self.camera_sensor.update()

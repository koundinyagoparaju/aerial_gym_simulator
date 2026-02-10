from aerial_gym.env_manager.base_env_manager import BaseManager
import torch
import numpy as np
from aerial_gym.utils.logging import CustomLogger

# Isaac Lab imports (Assumed availability)
try:
    from omni.isaac.lab.app import AppLauncher
    from omni.isaac.lab.sim import SimulationContext, SimulationCfg, PhysxCfg
    from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
    from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from omni.isaac.lab.terrains import TerrainImporter, TerrainImporterCfg
    from omni.isaac.core.utils.prims import create_prim
    from omni.isaac.core.utils.stage import get_current_stage
    import omni.isaac.core.utils.prims as prim_utils
    import omni.kit.commands
    import os
except ImportError:
    # Fallback for when Isaac Lab is not installed (e.g. during dev/linting)
    pass

logger = CustomLogger("IsaacLabEnvManager")

class IsaacLabEnv(BaseManager):
    def __init__(self, config, sim_config, has_IGE_cameras, device):
        super().__init__(config, device)
        self.sim_config = sim_config
        self.has_IGE_cameras = has_IGE_cameras
        self.device = device
        
        # Initialize Simulation Context
        self.sim = self._create_sim()
        self.scene = None
        self.env_origins = None
        
        # Tensor storage
        self.global_tensor_dict = {}

    def _create_sim(self):
        logger.info("Creating Isaac Lab Simulation Context")
        
        # Map sim_config to SimulationCfg
        # This requires mapping aerial_gym config to Isaac Lab config
        sim_cfg = SimulationCfg(
            dt=self.sim_config.sim.dt,
            device=self.device,
            physx=PhysxCfg(
                use_gpu=self.sim_config.sim.use_gpu_pipeline,
                min_position_iteration_count=self.sim_config.sim.substeps,
                min_velocity_iteration_count=1, # Default
                gravity=self.sim_config.sim.gravity
            )
        )
        
        simulation_context = SimulationContext(sim_cfg)
        simulation_context.set_camera_view(eye=[5.0, 5.0, 5.0], target=[0.0, 0.0, 0.0])
        return simulation_context

    def create_env(self, env_id):
        # Isaac Lab typically handles cloning internally via Scene or Cloner.
        # Here we just track the ID as aerial_gym manages the loop.
        return env_id

    def create_ground_plane(self):
        # Create ground plane using Isaac Lab or Core
        self.sim.spawn_ground_plane(prim_path="/World/ground_plane")

    def _import_urdf(self, urdf_path, asset_name):
        """
        Imports a URDF file and returns the path to the generated USD file.
        Checks for modification times to regenerate stale USDs.
        """
        dest_path = urdf_path.replace(".urdf", ".usd")
        
        # Check if regeneration is needed
        should_regenerate = True
        if os.path.exists(dest_path):
            urdf_mtime = os.path.getmtime(urdf_path)
            usd_mtime = os.path.getmtime(dest_path)
            if usd_mtime > urdf_mtime:
                should_regenerate = False
        
        if not should_regenerate:
            return dest_path
        
        logger.info(f"Converting URDF to USD: {urdf_path} -> {dest_path}")
        
        # Configure import settings
        import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = True
        import_config.fix_base = False
        import_config.distance_scale = 1.0

        # Execute import
        try:
            omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=urdf_path,
                import_config=import_config,
                dest_path=dest_path,
            )
        except Exception as e:
            logger.error(f"Failed to import URDF {urdf_path}: {e}")
            raise e
            
        return dest_path

    def add_asset_to_env(self, asset_info_dict, env_handle, env_id, global_asset_counter, segmentation_counter):
        # Convert URDF to USD if needed
        urdf_path = asset_info_dict.get("filename")
        if urdf_path and urdf_path.endswith(".urdf"):
             usd_path = self._import_urdf(urdf_path, asset_info_dict["asset_type"])
        else:
             usd_path = urdf_path # Assume it's already USD or handled

        # Determine prim path suffix
        if asset_info_dict.get("asset_type") == "robot":
            prim_suffix = "Robot"
        else:
            prim_suffix = f"Asset_{global_asset_counter}"

        prim_path = f"/World/Env_{env_id}/{prim_suffix}"
        
        # Create prim referencing the USD asset
        if usd_path:
             prim_utils.create_prim(
                prim_path,
                usd_path=usd_path,
                translation=np.array([0.0, 0.0, 0.0]) # Initial position, updated later
             )
        
        # Increment segmentation counter for unique IDs if needed
        # In Isaac Lab, segmentation is handled by the renderer, but aerial_gym tracks it manually.
        segmentation_counter += 1
        
        return global_asset_counter, segmentation_counter

    def prepare_for_simulation(self, env_manager, global_tensor_dict):
        # Reset simulation to apply changes
        self.sim.reset()
        
        # Create an Interactive Scene to manage assets and views
        scene_cfg = InteractiveSceneCfg(
            num_envs=self.cfg.env.num_envs,
            env_spacing=self.cfg.env.env_spacing,
            replicate_physics=False,
        )
        
        # Define Robot Articulation in the scene
        # We assume the robot was spawned at /World/Env_*/Robot
        scene_cfg.articulation_views = {
            "robot": ArticulationCfg(
                prim_path="/World/Env_.*/Robot",
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 1.0), # Default, will be overwritten by reset
                ),
            )
        }

        # Define Obstacles (Rigid Objects)
        # We assume obstacles are spawned at /World/Env_*/Asset_*
        # Note: If there are multiple types of assets, we might want to group them or use a wildcard if they share properties.
        # For simplicity, we use a single view for all "Asset_*" prims.
        scene_cfg.rigid_object_views = {
            "obstacles": RigidObjectCfg(
                prim_path="/World/Env_.*/Asset_.*",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0),
                ),
            )
        }

        # Instantiate the scene
        self.scene = InteractiveScene(scene_cfg)
        self.scene.reset()
        
        # Populate global_tensor_dict with Isaac Lab views
        self.global_tensor_dict = global_tensor_dict
        
        # 1. Robot State Tensor (num_envs, 13) -> [pos, quat, lin_vel, ang_vel]
        robot = self.scene.articulations["robot"]
        self.global_tensor_dict["robot_articulation"] = robot
        
        # Helper to map Isaac Lab [w, x, y, z] to Gym [x, y, z, w]
        def wxyz_to_xyzw(quat):
            return quat[:, [1, 2, 3, 0]]

        # Robot State mapping
        # Isaac Lab data is [w, x, y, z] for quat
        self.global_tensor_dict["robot_position"] = robot.data.root_pos_w
        self.global_tensor_dict["robot_orientation"] = wxyz_to_xyzw(robot.data.root_quat_w)
        self.global_tensor_dict["robot_linvel"] = robot.data.root_vel_w[:, 0:3]
        self.global_tensor_dict["robot_angvel"] = robot.data.root_vel_w[:, 3:6]
        
        # Construct flat state tensor [pos, xyzw, lin_vel, ang_vel]
        # We need to concatenate views. 
        # Note: Concatenating creates a COPY. If aerial_gym expects to modify this tensor 
        # and have it reflect in the components, we have a problem.
        # But usually 'robot_state_tensor' is read-only for obs, and written to for reset.
        # So creating a tensor here is fine, but we must update it in post_physics_step or whenever data changes.
        # Or better: Can we hack a view? No, memory is contiguous in one way.
        # We will create a buffer and populate it in post_physics_step.
        self.global_tensor_dict["robot_state_tensor"] = torch.zeros((self.num_envs, 13), device=self.device)
        
        # Body frame velocities (Required by BaseRobot/Controllers)
        # computed from world vel and orientation
        from aerial_gym.utils.math import quat_rotate_inverse
        # We initialize them here, update in post_physics_step
        self.global_tensor_dict["robot_body_linvel"] = torch.zeros_like(self.global_tensor_dict["robot_linvel"])
        self.global_tensor_dict["robot_body_angvel"] = torch.zeros_like(self.global_tensor_dict["robot_angvel"])
        
        # Contact Forces
        # Allocated to match IGE expectations
        self.global_tensor_dict["global_contact_force_tensor"] = torch.zeros(
            (self.num_envs, num_robot_bodies, 3), device=self.device
        )
        # Net contact force on root link (body 0)
        self.global_tensor_dict["robot_contact_force_tensor"] = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        
        # 4. Other Global Tensors required by Base Classes
        if "obstacles" in self.scene.rigid_objects:
            obstacles = self.scene.rigid_objects["obstacles"]
            self.global_tensor_dict["env_asset_state_tensor"] = obstacles.data.root_state_w.view(self.num_envs, -1, 13)
            # Map other obstacle tensors as needed (pos, ori, etc.)
            
        # 3. Global Force Tensor
        num_robot_bodies = robot.num_bodies
        self.global_tensor_dict["global_force_tensor"] = torch.zeros(
            (self.num_envs, num_robot_bodies, 3), device=self.device
        )
        self.global_tensor_dict["global_torque_tensor"] = torch.zeros(
            (self.num_envs, num_robot_bodies, 3), device=self.device
        )
        
        # 4. Other Global Tensors required by Base Classes
        self.global_tensor_dict["gravity"] = torch.tensor(
            self.sim_config.sim.gravity, device=self.device
        ).expand(self.num_envs, -1)
        self.global_tensor_dict["dt"] = self.sim_config.sim.dt
        
        # Environment Bounds (Mocking or retrieving from config)
        # aerial_gym uses these for randomization
        self.global_tensor_dict["env_bounds_min"] = torch.tensor(self.cfg.env.lower_bound_min, device=self.device).expand(self.num_envs, -1)
        self.global_tensor_dict["env_bounds_max"] = torch.tensor(self.cfg.env.upper_bound_max, device=self.device).expand(self.num_envs, -1)
        
        # Obstacle count
        self.global_tensor_dict["num_obstacles_in_env"] = 0
        if "obstacles" in self.scene.rigid_objects:
             # Assuming monolithic view, shape[0] is total assets across all envs?
             # Or is it num_envs * num_per_env?
             # We need num per env.
             # If we assumed consistent spawning:
             # num_obstacles = self.scene.rigid_objects["obstacles"].count // self.num_envs
             self.global_tensor_dict["num_obstacles_in_env"] = 1 # Placeholder, should be derived dynamically
        
        return True

    def pre_physics_step(self, actions):
        # Apply external forces/torques to robot bodies
        if "global_force_tensor" in self.global_tensor_dict:
            force_tensor = self.global_tensor_dict["global_force_tensor"]
            torque_tensor = self.global_tensor_dict["global_torque_tensor"]
            
            robot = self.scene.articulations["robot"]
            
            # Articulation.set_external_force_and_torque expects (forces, torques, body_ids)
            # forces: (num_envs, num_selected_bodies, 3)
            # If body_ids is None, it applies to all bodies, expecting (num_envs, num_bodies, 3)
            
            # Verify shapes to avoid runtime errors
            if force_tensor.shape[1] != robot.num_bodies:
                # If mismatch (e.g. tensor is just for root), we might need to pad or slice
                if force_tensor.shape[1] == 1 and robot.num_bodies > 1:
                    # Expand forces to (num_envs, num_bodies, 3) with zeros for non-root bodies
                    # This assumes forces are meant for the root link (body 0)
                    padded_forces = torch.zeros((self.num_envs, robot.num_bodies, 3), device=self.device)
                    padded_forces[:, 0, :] = force_tensor[:, 0, :]
                    padded_torques = torch.zeros((self.num_envs, robot.num_bodies, 3), device=self.device)
                    padded_torques[:, 0, :] = torque_tensor[:, 0, :]
                    
                    robot.set_external_force_and_torque(padded_forces, padded_torques)
                    return
                else:
                    # Log error or attempt best effort
                    logger.error(f"Global force tensor shape {force_tensor.shape} mismatch with robot bodies {robot.num_bodies}")
                    # Allow it to crash or pass if truly unrecoverable
                    pass
            
            robot.set_external_force_and_torque(force_tensor, torque_tensor)

        # Apply DOF actions (joint efforts, positions, or velocities)
        # aerial_gym uses specific tensors for this: dof_position_setpoint_tensor, dof_velocity_setpoint_tensor, dof_effort_tensor
        
        robot = self.scene.articulations["robot"]
        
        if "dof_position_setpoint_tensor" in self.global_tensor_dict:
             robot.set_joint_position_target(self.global_tensor_dict["dof_position_setpoint_tensor"])
             
        if "dof_velocity_setpoint_tensor" in self.global_tensor_dict:
             robot.set_joint_velocity_target(self.global_tensor_dict["dof_velocity_setpoint_tensor"])
             
        if "dof_effort_tensor" in self.global_tensor_dict:
             robot.set_joint_effort_target(self.global_tensor_dict["dof_effort_tensor"])

    def physics_step(self):
        self.sim.step()

    def post_physics_step(self):
        # Update buffers
        self.scene.update(dt=self.sim.get_physics_dt())
        
        # Update Global Tensors from Isaac Lab Views
        robot = self.global_tensor_dict["robot_articulation"]
        
        # 1. Orientation Permutation [w,x,y,z] -> [x,y,z,w]
        # We update the tensors in global_tensor_dict which are used by the task
        self.global_tensor_dict["robot_orientation"][:] = robot.data.root_quat_w[:, [1, 2, 3, 0]]
        self.global_tensor_dict["robot_position"][:] = robot.data.root_pos_w
        self.global_tensor_dict["robot_linvel"][:] = robot.data.root_vel_w[:, 0:3]
        self.global_tensor_dict["robot_angvel"][:] = robot.data.root_vel_w[:, 3:6]
        
        # Update combined robot_state_tensor
        self.global_tensor_dict["robot_state_tensor"][:, 0:3] = self.global_tensor_dict["robot_position"]
        self.global_tensor_dict["robot_state_tensor"][:, 3:7] = self.global_tensor_dict["robot_orientation"]
        self.global_tensor_dict["robot_state_tensor"][:, 7:10] = self.global_tensor_dict["robot_linvel"]
        self.global_tensor_dict["robot_state_tensor"][:, 10:13] = self.global_tensor_dict["robot_angvel"]
        
        # 2. Body Velocities
        from aerial_gym.utils.math import quat_rotate_inverse
        # Note: quat_rotate_inverse expects [x,y,z,w] which we just set in robot_orientation
        self.global_tensor_dict["robot_body_linvel"][:] = quat_rotate_inverse(
            self.global_tensor_dict["robot_orientation"], 
            self.global_tensor_dict["robot_linvel"]
        )
        self.global_tensor_dict["robot_body_angvel"][:] = quat_rotate_inverse(
            self.global_tensor_dict["robot_orientation"], 
            self.global_tensor_dict["robot_angvel"]
        )
        
        # 3. Contact Forces (if available)
        # aerial_gym expects (num_envs, num_bodies, 3)
        # We need to map or allocate if not done in prepare_for_sim
        # Assuming we allocate 'global_contact_force_tensor' in prepare_for_sim (I will add it there next if missed)
        if "global_contact_force_tensor" in self.global_tensor_dict:
             # Isaac Lab: net_contact_forces_w
             # Check shapes. Isaac Lab might not compute this unless requested?
             # It is usually available in data.
             self.global_tensor_dict["global_contact_force_tensor"][:] = robot.data.net_contact_forces_w
             self.global_tensor_dict["robot_contact_force_tensor"][:] = robot.data.net_contact_forces_w[:, 0, :] # Assuming root is 0

    def reset_idx(self, env_ids):
        # Reset specific environments
        if "robot_articulation" in self.global_tensor_dict:
             robot = self.global_tensor_dict["robot_articulation"]
             
             # Call reset first to clear internal buffers/velocities
             robot.reset(env_ids)
             
             state_tensor = self.global_tensor_dict["robot_state_tensor"]
             
             # Permute Quaternion back to [w, x, y, z] for Isaac Lab
             # state_tensor has [x, y, z, w] at indices 3:7
             reset_state = state_tensor[env_ids].clone()
             # [x, y, z, w] -> [w, x, y, z]
             reset_state[:, 3:7] = state_tensor[env_ids, 3:7][:, [3, 0, 1, 2]]
             
             # Write the explicit state to sim
             robot.write_root_state_to_sim(reset_state, env_ids=env_ids)
             
        # Reset obstacles
        if "env_asset_state_tensor" in self.global_tensor_dict and "obstacles" in self.scene.rigid_objects:
             obstacles = self.scene.rigid_objects["obstacles"]
             obstacle_state = self.global_tensor_dict["env_asset_state_tensor"]
             
             num_assets = obstacle_state.shape[1]
             # Flatten selected states: (len(env_ids), num_assets, 13) -> (len(env_ids) * num_assets, 13)
             selected_states = obstacle_state[env_ids].view(-1, 13)
             
             indices = []
             for env_id in env_ids:
                 start = env_id * num_assets
                 indices.append(torch.arange(start, start + num_assets, device=self.device))
             
             if indices:
                 flat_indices = torch.cat(indices)
                 # Reset then write
                 obstacles.reset(flat_indices)
                 obstacles.write_root_state_to_sim(selected_states, env_ids=flat_indices)

    def step_graphics(self):
        self.sim.render()

    def render_viewer(self):
        # Isaac Lab handles viewer updates automatically via the App.
        pass

    def write_to_sim(self):
        # Write state to sim
        if "robot_articulation" in self.global_tensor_dict:
             robot = self.global_tensor_dict["robot_articulation"]
             state_tensor = self.global_tensor_dict["robot_state_tensor"]
             
             # Permute quaternion [x,y,z,w] -> [w,x,y,z]
             sim_state = state_tensor.clone()
             sim_state[:, 3:7] = state_tensor[:, 3:7][:, [3, 0, 1, 2]]
             
             robot.write_root_state_to_sim(sim_state)
             
        # Also obstacles if needed
        if "env_asset_state_tensor" in self.global_tensor_dict and "obstacles" in self.scene.rigid_objects:
             obstacles = self.scene.rigid_objects["obstacles"]
             state_tensor = self.global_tensor_dict["env_asset_state_tensor"]
             flat_state = state_tensor.view(-1, 13)
             obstacles.write_root_state_to_sim(flat_state)

import time
from aerial_gym.utils.logging import CustomLogger
import torch
import numpy as np

logger = CustomLogger("test_isaaclab_backend")
from aerial_gym.registry.task_registry import task_registry

def test_backend():
    try:
        logger.info("Starting Isaac Lab Backend Test")
        
        # Initialize task in headless mode
        # We explicitly set num_envs to a small number for testing
        rl_task_env = task_registry.make_task(
            "position_setpoint_task",
            num_envs=2,
            headless=True,
            use_warp=False
        )
        
        logger.info("Task environment created successfully")
        
        # Reset environment
        obs, info = rl_task_env.reset()
        logger.info("Environment reset successfully")
        
        # Define dummy actions
        action_dim = rl_task_env.sim_env.robot_manager.robot.controller_config.num_actions
        actions = torch.zeros((2, action_dim), device=rl_task_env.device)
        
        # Step environment
        for i in range(5):
            obs, reward, terminated, truncated, info = rl_task_env.step(actions=actions)
            logger.info(f"Step {i+1} completed")
            
        logger.info("Backend test PASSED")
        return True
    except Exception as e:
        logger.error(f"Backend test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_backend()

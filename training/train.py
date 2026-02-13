#!/usr/bin/env python3
"""Main training script for quadruped locomotion using PPO."""

import os
import sys
import argparse
import yaml
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.utils import set_random_seed

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from envs.quadruped_env import QuadrupedEnv
from envs.terrain_generator import TerrainGenerator
from training.callbacks import VideoRecorderCallback, RewardLoggingCallback, CurriculumCallback


def make_env(rank, seed, config, terrain_xml=None):
    """Create a closure for environment creation."""
    def _init():
        env_cfg = config["env"]
        env = QuadrupedEnv(
            max_episode_steps=env_cfg["max_episode_steps"],
            action_scale=env_cfg["action_scale"],
            ctrl_cost_weight=env_cfg["rewards"]["ctrl_cost_weight"],
            alive_bonus=env_cfg["rewards"]["alive_bonus"],
            velocity_tracking_weight=env_cfg["rewards"]["velocity_tracking_weight"],
            fall_penalty=env_cfg["rewards"]["fall_penalty"],
            smoothness_weight=env_cfg["rewards"]["smoothness_weight"],
            foot_slip_weight=env_cfg["rewards"]["foot_slip_weight"],
            orientation_weight=env_cfg["rewards"]["orientation_weight"],
            height_weight=env_cfg["rewards"]["height_weight"],
            cmd_vx=env_cfg["command"]["vx"],
            cmd_vy=env_cfg["command"]["vy"],
            cmd_yaw_rate=env_cfg["command"]["yaw_rate"],
            domain_randomize=env_cfg.get("domain_randomize", False),
            terrain_xml=terrain_xml,
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def make_eval_env(config, terrain_xml=None):
    """Create a single evaluation env with rendering."""
    env_cfg = config["env"]
    env = QuadrupedEnv(
        render_mode="rgb_array",
        max_episode_steps=env_cfg["max_episode_steps"],
        action_scale=env_cfg["action_scale"],
        ctrl_cost_weight=env_cfg["rewards"]["ctrl_cost_weight"],
        alive_bonus=env_cfg["rewards"]["alive_bonus"],
        velocity_tracking_weight=env_cfg["rewards"]["velocity_tracking_weight"],
        fall_penalty=env_cfg["rewards"]["fall_penalty"],
        smoothness_weight=env_cfg["rewards"]["smoothness_weight"],
        foot_slip_weight=env_cfg["rewards"]["foot_slip_weight"],
        orientation_weight=env_cfg["rewards"]["orientation_weight"],
        height_weight=env_cfg["rewards"]["height_weight"],
        cmd_vx=env_cfg["command"]["vx"],
        cmd_vy=env_cfg["command"]["vy"],
        cmd_yaw_rate=env_cfg["command"]["yaw_rate"],
        domain_randomize=False,
        terrain_xml=terrain_xml,
    )
    return env


def get_activation_fn(name):
    """Get activation function by name."""
    activations = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "elu": torch.nn.ELU,
    }
    return activations.get(name, torch.nn.ELU)


def train(config_path, resume_from=None):
    """Run PPO training."""
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    project_root = os.path.dirname(os.path.dirname(__file__))
    training_cfg = config["training"]
    ppo_cfg = config["ppo"]
    policy_cfg = config["policy"]
    paths_cfg = config["paths"]

    # Paths
    model_dir = os.path.join(project_root, paths_cfg["model_dir"])
    log_dir = os.path.join(project_root, paths_cfg["log_dir"])
    video_dir = os.path.join(project_root, paths_cfg["video_dir"])
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    seed = training_cfg["seed"]
    n_envs = training_cfg["n_envs"]

    # Generate terrains
    terrain_gen = TerrainGenerator()
    terrain_paths = terrain_gen.generate_all(seed=seed)
    print(f"Generated terrains: {list(terrain_paths.keys())}")

    # Use flat terrain by default, or curriculum terrain
    terrain_xml = terrain_paths["flat"]

    # Create vectorized training environments
    env_fns = [make_env(i, seed, config, terrain_xml) for i in range(n_envs)]
    if n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    # Wrap with observation/reward normalization
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Create eval env
    eval_env_raw = make_eval_env(config, terrain_xml)

    # Build PPO model
    policy_kwargs = {
        "net_arch": policy_cfg["net_arch"],
        "activation_fn": get_activation_fn(policy_cfg["activation_fn"]),
    }

    if resume_from:
        print(f"Resuming from: {resume_from}")
        model = PPO.load(resume_from, env=vec_env)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=ppo_cfg["learning_rate"],
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            n_epochs=ppo_cfg["n_epochs"],
            gamma=ppo_cfg["gamma"],
            gae_lambda=ppo_cfg["gae_lambda"],
            clip_range=ppo_cfg["clip_range"],
            ent_coef=ppo_cfg["ent_coef"],
            vf_coef=ppo_cfg["vf_coef"],
            max_grad_norm=ppo_cfg["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            verbose=1,
            seed=seed,
        )

    print(f"\nModel architecture:\n{model.policy}")
    print(f"\nTraining for {training_cfg['total_timesteps']:,} timesteps with {n_envs} envs")

    # Setup callbacks
    callbacks = []

    # Checkpoint saving
    checkpoint_cb = CheckpointCallback(
        save_freq=max(training_cfg["save_freq"] // n_envs, 1),
        save_path=model_dir,
        name_prefix="go1_ppo",
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_cb)

    # Reward logging
    callbacks.append(RewardLoggingCallback())

    # Video recording
    video_cb = VideoRecorderCallback(
        eval_env=eval_env_raw,
        video_dir=video_dir,
        video_freq=max(training_cfg["video_freq"] // n_envs, 1),
        video_length=training_cfg["video_length"],
    )
    callbacks.append(video_cb)

    # Curriculum learning
    curriculum_cfg = config.get("curriculum", {})
    if curriculum_cfg.get("enabled", False):
        curriculum_cb = CurriculumCallback(curriculum_cfg["stages"])
        callbacks.append(curriculum_cb)

    callback_list = CallbackList(callbacks)

    # Train
    model.learn(
        total_timesteps=training_cfg["total_timesteps"],
        callback=callback_list,
        log_interval=training_cfg["log_interval"],
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(model_dir, "go1_ppo_final")
    model.save(final_path)
    vec_env.save(os.path.join(model_dir, "vec_normalize.pkl"))
    print(f"\nTraining complete! Model saved to: {final_path}")

    vec_env.close()
    eval_env_raw.close()

    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train quadruped locomotion with PPO")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to config YAML",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    train(args.config, resume_from=args.resume)

#!/usr/bin/env python3
"""Generate demo videos and training analysis plots."""

import os
import sys
import argparse
import json
import numpy as np
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from envs.quadruped_env import QuadrupedEnv
from envs.terrain_generator import TerrainGenerator


def record_video(model, env, output_path, n_steps=500, fps=50):
    """Record a video of the policy running."""
    frames = []
    obs, _ = env.reset()

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            obs, _ = env.reset()

    if frames:
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Saved video: {output_path} ({len(frames)} frames)")
    return len(frames)


def record_all_terrains(model_path, video_dir, n_steps=500):
    """Record demo videos on all terrain types."""
    model = PPO.load(model_path)
    terrain_gen = TerrainGenerator()
    terrain_paths = terrain_gen.generate_all()
    os.makedirs(video_dir, exist_ok=True)

    for terrain_name, terrain_xml in terrain_paths.items():
        print(f"Recording on terrain: {terrain_name}")
        env = QuadrupedEnv(
            terrain_xml=terrain_xml,
            render_mode="rgb_array",
            max_episode_steps=n_steps,
        )
        output_path = os.path.join(video_dir, f"demo_{terrain_name}.mp4")
        record_video(model, env, output_path, n_steps=n_steps)
        env.close()


def record_random_vs_trained(model_path, video_dir, n_steps=300):
    """Record side-by-side comparison of untrained vs trained agent."""
    model = PPO.load(model_path)
    os.makedirs(video_dir, exist_ok=True)

    # Random policy
    env = QuadrupedEnv(render_mode="rgb_array", max_episode_steps=n_steps)
    frames_random = []
    obs, _ = env.reset()
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            frames_random.append(frame)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

    # Trained policy
    env = QuadrupedEnv(render_mode="rgb_array", max_episode_steps=n_steps)
    frames_trained = []
    obs, _ = env.reset()
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            frames_trained.append(frame)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

    # Combine side by side
    min_frames = min(len(frames_random), len(frames_trained))
    if min_frames > 0:
        combined = []
        for i in range(min_frames):
            fr = frames_random[i]
            ft = frames_trained[i]
            # Resize to same height if needed
            h = min(fr.shape[0], ft.shape[0])
            fr = fr[:h, :, :]
            ft = ft[:h, :, :]
            # Add labels
            combined_frame = np.concatenate([fr, ft], axis=1)
            combined.append(combined_frame)

        path = os.path.join(video_dir, "comparison_random_vs_trained.mp4")
        imageio.mimsave(path, combined, fps=50)
        print(f"Saved comparison video: {path}")


def plot_training_curves(log_dir, output_dir):
    """Plot training curves from TensorBoard logs using monitor CSVs."""
    os.makedirs(output_dir, exist_ok=True)

    # Try to load tensorboard data
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        # Find the latest run directory
        run_dirs = []
        for entry in os.listdir(log_dir):
            path = os.path.join(log_dir, entry)
            if os.path.isdir(path):
                run_dirs.append(path)

        if not run_dirs:
            print("No TensorBoard log directories found.")
            return

        latest_run = max(run_dirs, key=os.path.getmtime)
        ea = EventAccumulator(latest_run)
        ea.Reload()

        tags = ea.Tags().get("scalars", [])
        if not tags:
            print("No scalar data found in TensorBoard logs.")
            return

        # Plot reward curves
        reward_tags = [t for t in tags if "reward" in t.lower() or "ep_rew" in t.lower()]
        if reward_tags:
            fig, ax = plt.subplots(figsize=(10, 6))
            for tag in reward_tags[:5]:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                ax.plot(steps, values, label=tag, alpha=0.8)
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Reward")
            ax.set_title("Training Reward Curves")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.savefig(os.path.join(output_dir, "reward_curves.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved reward curves plot")

        # Plot episode length
        length_tags = [t for t in tags if "len" in t.lower() or "ep_len" in t.lower()]
        if length_tags:
            fig, ax = plt.subplots(figsize=(10, 6))
            for tag in length_tags[:3]:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                ax.plot(steps, values, label=tag, alpha=0.8)
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Episode Length")
            ax.set_title("Episode Length Over Training")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.savefig(os.path.join(output_dir, "episode_length.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved episode length plot")

        # Plot loss curves
        loss_tags = [t for t in tags if "loss" in t.lower()]
        if loss_tags:
            fig, axes = plt.subplots(1, min(len(loss_tags), 3), figsize=(15, 5))
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            for ax, tag in zip(axes, loss_tags[:3]):
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                ax.plot(steps, values, alpha=0.8)
                ax.set_title(tag)
                ax.set_xlabel("Timesteps")
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "losses.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved loss curves plot")

    except ImportError:
        print("TensorBoard not available. Skipping curve plots.")
    except Exception as e:
        print(f"Error plotting training curves: {e}")


def plot_evaluation_results(results_path, output_dir):
    """Plot evaluation results across terrains."""
    os.makedirs(output_dir, exist_ok=True)

    with open(results_path, "r") as f:
        results = json.load(f)

    terrains = list(results.keys())
    rewards = [results[t]["mean_reward"] for t in terrains]
    survival = [results[t]["survival_rate"] * 100 for t in terrains]
    distances = [results[t]["mean_distance"] for t in terrains]
    velocities = [results[t]["mean_velocity"] for t in terrains]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Reward comparison
    axes[0, 0].bar(terrains, rewards, color="steelblue")
    axes[0, 0].set_title("Mean Reward by Terrain")
    axes[0, 0].set_ylabel("Mean Reward")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # Survival rate
    axes[0, 1].bar(terrains, survival, color="forestgreen")
    axes[0, 1].set_title("Survival Rate by Terrain")
    axes[0, 1].set_ylabel("Survival Rate (%)")
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Distance traveled
    axes[1, 0].bar(terrains, distances, color="coral")
    axes[1, 0].set_title("Mean Distance Traveled")
    axes[1, 0].set_ylabel("Distance (m)")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Velocity
    axes[1, 1].bar(terrains, velocities, color="mediumpurple")
    axes[1, 1].set_title("Mean Velocity")
    axes[1, 1].set_ylabel("Velocity (m/s)")
    axes[1, 1].tick_params(axis="x", rotation=45)

    fig.suptitle("Quadruped Evaluation Across Terrains", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "evaluation_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Saved evaluation comparison plot")


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations and demo videos")
    subparsers = parser.add_subparsers(dest="command")

    # Record videos
    video_parser = subparsers.add_parser("videos", help="Record demo videos")
    video_parser.add_argument("model_path", type=str)
    video_parser.add_argument("--output-dir", type=str, default="videos")
    video_parser.add_argument("--n-steps", type=int, default=500)
    video_parser.add_argument("--comparison", action="store_true",
                              help="Also record random vs trained comparison")

    # Plot training
    plot_parser = subparsers.add_parser("plots", help="Generate training plots")
    plot_parser.add_argument("--log-dir", type=str, default="logs")
    plot_parser.add_argument("--output-dir", type=str, default="plots")

    # Plot evaluation
    eval_parser = subparsers.add_parser("eval-plots", help="Plot evaluation results")
    eval_parser.add_argument("results_path", type=str)
    eval_parser.add_argument("--output-dir", type=str, default="plots")

    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(__file__))

    if args.command == "videos":
        output_dir = os.path.join(project_root, args.output_dir)
        record_all_terrains(args.model_path, output_dir, n_steps=args.n_steps)
        if args.comparison:
            record_random_vs_trained(args.model_path, output_dir)

    elif args.command == "plots":
        log_dir = os.path.join(project_root, args.log_dir)
        output_dir = os.path.join(project_root, args.output_dir)
        plot_training_curves(log_dir, output_dir)

    elif args.command == "eval-plots":
        output_dir = os.path.join(project_root, args.output_dir)
        plot_evaluation_results(args.results_path, output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

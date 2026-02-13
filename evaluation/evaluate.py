#!/usr/bin/env python3
"""Evaluate trained quadruped policy on various terrains."""

import os
import sys
import argparse
import json
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from envs.quadruped_env import QuadrupedEnv
from envs.terrain_generator import TerrainGenerator


def evaluate_policy(model, env, n_episodes=20):
    """Evaluate policy and collect metrics."""
    metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "distances_traveled": [],
        "max_heights": [],
        "avg_velocities": [],
        "falls": 0,
        "reward_components": {},
    }

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        start_x = 0.0
        reward_components = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            # Accumulate reward components
            for key, value in info.items():
                if key.startswith("r_"):
                    if key not in reward_components:
                        reward_components[key] = []
                    reward_components[key].append(value)

            if terminated:
                metrics["falls"] += 1

        metrics["episode_rewards"].append(total_reward)
        metrics["episode_lengths"].append(steps)
        metrics["distances_traveled"].append(info.get("trunk_x", 0.0))
        metrics["avg_velocities"].append(
            info.get("trunk_x", 0.0) / (steps * 0.02) if steps > 0 else 0
        )

        # Average reward components for this episode
        for key, values in reward_components.items():
            if key not in metrics["reward_components"]:
                metrics["reward_components"][key] = []
            metrics["reward_components"][key].append(np.mean(values))

    # Aggregate
    results = {
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(metrics["episode_rewards"])),
        "std_reward": float(np.std(metrics["episode_rewards"])),
        "mean_length": float(np.mean(metrics["episode_lengths"])),
        "mean_distance": float(np.mean(metrics["distances_traveled"])),
        "mean_velocity": float(np.mean(metrics["avg_velocities"])),
        "fall_rate": metrics["falls"] / n_episodes,
        "survival_rate": 1.0 - metrics["falls"] / n_episodes,
    }

    # Average reward components
    for key, values in metrics["reward_components"].items():
        results[f"mean_{key}"] = float(np.mean(values))

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained quadruped policy")
    parser.add_argument("model_path", type=str, help="Path to trained model .zip")
    parser.add_argument(
        "--vec-normalize",
        type=str,
        default=None,
        help="Path to VecNormalize stats file",
    )
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--terrain", type=str, default="all",
                        choices=["all"] + TerrainGenerator.TERRAIN_TYPES)
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(__file__))

    # Generate terrains
    terrain_gen = TerrainGenerator()
    terrain_paths = terrain_gen.generate_all()

    if args.terrain == "all":
        terrains_to_test = terrain_paths
    else:
        terrains_to_test = {args.terrain: terrain_paths[args.terrain]}

    # Load model
    model = PPO.load(args.model_path)

    all_results = {}
    for terrain_name, terrain_xml in terrains_to_test.items():
        print(f"\n{'='*60}")
        print(f"Evaluating on terrain: {terrain_name}")
        print(f"{'='*60}")

        env = QuadrupedEnv(
            terrain_xml=terrain_xml,
            max_episode_steps=1000,
        )

        results = evaluate_policy(model, env, n_episodes=args.n_episodes)
        all_results[terrain_name] = results

        print(f"  Mean reward:     {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Mean length:     {results['mean_length']:.0f} steps")
        print(f"  Mean distance:   {results['mean_distance']:.2f} m")
        print(f"  Mean velocity:   {results['mean_velocity']:.3f} m/s")
        print(f"  Survival rate:   {results['survival_rate']*100:.1f}%")

        env.close()

    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(project_root, "evaluation_results.json")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

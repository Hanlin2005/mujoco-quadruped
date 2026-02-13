"""Custom callbacks for training monitoring, video recording, and checkpointing."""

import os
import numpy as np
import imageio
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


class VideoRecorderCallback(BaseCallback):
    """Records videos of the agent at regular intervals during training."""

    def __init__(
        self,
        eval_env,
        video_dir,
        video_freq=200_000,
        video_length=500,
        verbose=1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.video_dir = video_dir
        self.video_freq = video_freq
        self.video_length = video_length
        os.makedirs(video_dir, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps % self.video_freq == 0:
            self._record_video()
        return True

    def _record_video(self):
        """Record a video of the current policy."""
        frames = []
        obs, _ = self.eval_env.reset()
        for _ in range(self.video_length):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            frame = self.eval_env.render()
            if frame is not None:
                frames.append(frame)
            if terminated or truncated:
                obs, _ = self.eval_env.reset()

        if frames:
            step_str = f"{self.num_timesteps:08d}"
            video_path = os.path.join(self.video_dir, f"policy_{step_str}.mp4")
            imageio.mimsave(video_path, frames, fps=50)
            if self.verbose:
                print(f"[VideoRecorder] Saved video: {video_path} ({len(frames)} frames)")


class RewardLoggingCallback(BaseCallback):
    """Logs detailed reward components to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._reward_components = {}

    def _on_step(self):
        # Collect reward info from all envs
        infos = self.locals.get("infos", [])
        for info in infos:
            for key, value in info.items():
                if key.startswith("r_"):
                    if key not in self._reward_components:
                        self._reward_components[key] = []
                    self._reward_components[key].append(value)

        # Log every 2048 steps (one PPO rollout)
        if self.num_timesteps % 2048 == 0 and self._reward_components:
            for key, values in self._reward_components.items():
                self.logger.record(f"reward/{key}", np.mean(values))
            self._reward_components = {}

        return True


class CurriculumCallback(BaseCallback):
    """Implements curriculum learning by adjusting environment difficulty."""

    def __init__(self, curriculum_stages, verbose=1):
        super().__init__(verbose)
        self.stages = curriculum_stages
        self.current_stage = 0
        self._cumulative_timesteps = 0
        self._stage_boundaries = []
        total = 0
        for stage in self.stages:
            total += stage["timesteps"]
            self._stage_boundaries.append(total)

    def _on_step(self):
        # Check if we need to advance curriculum stage
        if self.current_stage < len(self.stages) - 1:
            if self.num_timesteps >= self._stage_boundaries[self.current_stage]:
                self.current_stage += 1
                stage = self.stages[self.current_stage]
                if self.verbose:
                    print(
                        f"[Curriculum] Advancing to stage {self.current_stage}: "
                        f"{stage['name']} at step {self.num_timesteps}"
                    )
                self._apply_stage(stage)

        self.logger.record("curriculum/stage", self.current_stage)
        return True

    def _apply_stage(self, stage):
        """Apply curriculum stage settings to training environments."""
        env = self.training_env
        cmd_range = stage.get("cmd_vx_range", [0.5, 1.0])

        for i in range(env.num_envs):
            sub_env = env.envs[i].unwrapped
            sub_env.domain_randomize = stage.get("domain_randomize", False)
            sub_env.cmd_vx = np.random.uniform(cmd_range[0], cmd_range[1])

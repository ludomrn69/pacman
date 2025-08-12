#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Play a trained PPO agent on ALE/MsPacman-v5 with on-screen rendering.
Usage:
  python play_mspacman.py --model-path runs/ppo_mspacman_opt/best_model.zip --episodes 3 --fps 60
"""

import argparse
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import NoopResetEnv
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
import ale_py

# Register ALE environments (Gymnasium v5 namespace)
gym.register_envs(ale_py)

def make_play_vecenv(env_id: str, seed: int | None, full_action_space: bool):
    """Single-env VecEnv with same preprocessing as eval, but render_mode='human'."""
    def _thunk():
        env = gym.make(env_id, render_mode="human", full_action_space=full_action_space)
        env = NoopResetEnv(env, noop_max=30)
        # Important: resize then grayscale to keep shape (84,84,1)
        env = ResizeObservation(env, (84, 84))
        env = GrayscaleObservation(env, keep_dim=True)
        if seed is not None:
            env.reset(seed=seed)
        return env
    venv = DummyVecEnv([_thunk])
    venv = VecFrameStack(venv, n_stack=4)
    venv = VecTransposeImage(venv)  # (H,W,C)->(C,H,W) for SB3 CNN
    return venv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default="runs/ppo_mspacman_opt/best_model.zip")
    p.add_argument("--env-id", type=str, default="ALE/MsPacman-v5")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--full-action-space", action="store_true")
    args = p.parse_args()

    print(f"Loading model: {args.model_path}")
    model = PPO.load(args.model_path, device="auto")
    # Match env action space with the model expectation (auto-detect)
    expected_n = getattr(getattr(model, "action_space", None), "n", None)

    env = make_play_vecenv(args.env_id, args.seed, args.full_action_space)
    # If the model was trained with a different action space (full vs minimal), rebuild env accordingly
    try:
      actual_n = env.action_space.n
    except Exception:
      actual_n = None
    if expected_n is not None and actual_n is not None and expected_n != actual_n:
      want_full = expected_n > actual_n
      if want_full != args.full_action_space:
        print(f"[info] Model expects {expected_n} actions but env has {actual_n}. Rebuilding with full_action_space={want_full}.")
        env.close()
        env = make_play_vecenv(args.env_id, args.seed, want_full)

    sleep_dt = 1.0 / max(1, args.fps)
    for ep in range(args.episodes):
        try:
            if args.seed is not None:
                env.seed(args.seed + ep)
        except Exception:
            pass

        obs = env.reset()
        done = [False]
        ep_return = 0.0
        steps = 0

        # render first frame (some setups like it)
        try:
            env.envs[0].render()
        except Exception:
            pass

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, done, infos = env.step(action)
            ep_return += float(rewards[0])
            steps += 1
            try:
                env.envs[0].render()
            except Exception:
                pass
            time.sleep(sleep_dt)

        print(f"[Episode {ep+1}/{args.episodes}] return={ep_return:.1f} steps={steps}")

    env.close()

if __name__ == "__main__":
    main()
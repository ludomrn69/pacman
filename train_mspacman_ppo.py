#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a PPO agent on ALE/MsPacman-v5 (Gymnasium + Stable-Baselines3)
- Clean wrappers for ALE v5 (avoid double frameskip)
- Vectorized envs, frame stacking, grayscale+resize
- Separate eval env (no EpisodicLife), multi-seed support
- Early stopping on no improvement + checkpointing + TensorBoard logs
Usage:
    pip install "stable-baselines3>=2.2.0" "gymnasium[atari,accept-rom-license]" ale-py tensorboard
    python train_mspacman_ppo.py --total-timesteps 10000000 --n-envs 8 --save-dir runs/ppo_mspacman --seed 0
"""
import argparse
import os
import random
from pathlib import Path
from typing import Callable


import numpy as np
import torch as th
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement, CallbackList, ProgressBarCallback
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv, NoopResetEnv
import ale_py


gym.register_envs(ale_py)

def make_train_wrapper(noop_max: int = 30, terminal_on_life_loss: bool = True) -> Callable[[gym.Env], gym.Env]:
    """
    Training-time wrapper: No-op reset, episodic life, grayscale+resize. No reward clipping (we care about raw score).
    We avoid any extra frame-skip here because ALE v5 already uses frameskip=4 and sticky actions by default.
    """
    def _wrap(env: gym.Env) -> gym.Env:
        env = NoopResetEnv(env, noop_max=noop_max)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        env = ResizeObservation(env, (84, 84))          # (84, 84, 3)
        env = GrayscaleObservation(env, keep_dim=True)  # (84, 84, 1)
        return env
    return _wrap


def make_eval_wrapper(noop_max: int = 30) -> Callable[[gym.Env], gym.Env]:
    """
    Eval-time wrapper: No-op reset, NO episodic life (true game over), grayscale+resize.
    """
    def _wrap(env: gym.Env) -> gym.Env:
        env = NoopResetEnv(env, noop_max=noop_max)
        env = ResizeObservation(env, (84, 84))
        env = GrayscaleObservation(env, keep_dim=True)
        return env
    return _wrap


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)


def build_venv(env_id: str,
               n_envs: int,
               seed: int,
               wrapper_fn: Callable[[gym.Env], gym.Env],
               full_action_space: bool = False,
               vec_env_cls=SubprocVecEnv):
    """
    Create a vectorized ALE v5 env with our custom wrappers.
    We apply VecFrameStack(4) and VecTransposeImage to get (C,H,W) for PyTorch CNNs.
    """
    env_kwargs = dict(full_action_space=full_action_space)
    venv = make_vec_env(env_id,
                        n_envs=n_envs,
                        seed=seed,
                        wrapper_class=wrapper_fn,
                        env_kwargs=env_kwargs,
                        vec_env_cls=vec_env_cls)
    venv = VecFrameStack(venv, n_stack=4)
    venv = VecTransposeImage(venv)  # (H,W,C) -> (C,H,W)
    return venv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="ALE/MsPacman-v5")
    parser.add_argument("--save-dir", type=str, default="runs/ppo_mspacman")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=100_000,
                        help="Evaluate every N steps (per environment). Final frequency is eval_freq // n_envs.")
    parser.add_argument("--checkpoint-freq", type=int, default=500_000)
    parser.add_argument("--full-action-space", action="store_true",
                        help="Use the full 18-action space instead of reduced action set.")
    parser.add_argument("--subproc", action="store_true", help="Use SubprocVecEnv (default).")
    parser.add_argument("--dummy", action="store_true", help="Use DummyVecEnv instead of SubprocVecEnv.")
    args = parser.parse_args()

    # Sanity
    os.makedirs(args.save_dir, exist_ok=True)
    tb_log = args.save_dir

    # Seeds
    set_global_seeds(args.seed)

    # Choose VecEnv backend
    vec_cls = DummyVecEnv if args.dummy else SubprocVecEnv

    # Training and evaluation environments
    train_env = build_venv(args.env_id, args.n_envs, args.seed,
                           make_train_wrapper(), args.full_action_space, vec_env_cls=vec_cls)

    # Note: eval env without EpisodicLife, deterministic policy
    eval_env = build_venv(args.env_id, 1, args.seed + 10,
                          make_eval_wrapper(), args.full_action_space, vec_env_cls=DummyVecEnv)

    # Auto-adjust batch size for single-env debug runs (keeps it a factor of n_steps * n_envs)
    _batch_size = 256 if args.n_envs == 1 else 512
    # PPO hyperparameters (solid defaults for Atari)
    model = PPO(
        "CnnPolicy",
        train_env,
        n_steps=256,
        batch_size=_batch_size,
        n_epochs=4,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,
        tensorboard_log=tb_log,
        verbose=1,
        seed=args.seed,
    )

    # Callbacks: Eval + early stop on plateau + checkpoints + progress bar
    eval_freq = max(args.eval_freq // args.n_envs, 1)
    early_stop = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,  # stop after 10 evals without improvement
        min_evals=5,                  # give it at least 5 evals before starting to count
        verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.save_dir,
        log_path=args.save_dir,
        eval_freq=eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        callback_after_eval=early_stop  # check plateau after each evaluation
    )
    checkpoint_callback = CheckpointCallback(save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                                             save_path=args.save_dir, name_prefix="ckpt")
    callbacks = CallbackList([eval_callback, checkpoint_callback, ProgressBarCallback()])

    # Train
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    model.save(os.path.join(args.save_dir, "final_model"))

    # Final evaluation across multiple seeds
    print("Evaluating best model (if saved) or final model...")
    try:
        best_path = os.path.join(args.save_dir, "best_model.zip")
        if os.path.exists(best_path):
            model = PPO.load(best_path, env=eval_env, device="auto")
        else:
            print("No best_model.zip found, using final model.")
            model.set_env(eval_env)
    except Exception as e:
        print(f"Could not load best model, using current model. Reason: {e}")
        model.set_env(eval_env)

    all_returns = []
    for s in args.eval_seeds:
        obs, info = eval_env.reset(seed=s)
        ep_returns = []
        ep_return = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_return += float(reward)
            if terminated or truncated:
                ep_returns.append(ep_return)
                all_returns.append(ep_return)
                print(f"[seed {s}] Episode return: {ep_return:.1f}")
                break

    if len(all_returns) > 0:
        mean_r = float(np.mean(all_returns))
        std_r = float(np.std(all_returns))
        print(f"Mean eval return over {len(all_returns)} episodes: {mean_r:.1f} ± {std_r:.1f}")

    # Clean up
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()

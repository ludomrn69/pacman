#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a QRDQN agent on ALE/MsPacman-v5 (Gymnasium + SB3-Contrib).
- Clean Atari wrappers: NoopReset, Resize(84x84), Grayscale, FrameStack(4)
- Full vs minimal action space switch
- Separate eval env (no EpisodicLife)
- EvalCallback + best_model saving + TensorBoard logs

Install:
  pip install "gymnasium[atari,accept-rom-license]" ale-py "stable-baselines3>=2.2.0" "sb3-contrib>=2.2.0" tensorboard
"""

import argparse, os, random
from typing import Callable
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage, VecMonitor
from stable_baselines3.common.atari_wrappers import NoopResetEnv, EpisodicLifeEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold, CallbackList, ProgressBarCallback
from sb3_contrib import QRDQN
import ale_py

# Register ALE v5 envs in Gymnasium
gym.register_envs(ale_py)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def make_train_wrapper(noop_max: int = 30, episodic_life: bool = True) -> Callable[[gym.Env], gym.Env]:
    def _wrap(env: gym.Env) -> gym.Env:
        env = NoopResetEnv(env, noop_max=noop_max)
        if episodic_life:
            env = EpisodicLifeEnv(env)
        env = ResizeObservation(env, (84, 84))
        env = GrayscaleObservation(env, keep_dim=True)  # (84,84,1)
        return env
    return _wrap


def make_eval_wrapper(noop_max: int = 30) -> Callable[[gym.Env], gym.Env]:
    def _wrap(env: gym.Env) -> gym.Env:
        env = NoopResetEnv(env, noop_max=noop_max)
        env = ResizeObservation(env, (84, 84))
        env = GrayscaleObservation(env, keep_dim=True)
        return env
    return _wrap


def build_vec_env(env_id: str, seed: int, full_action_space: bool, wrapper: Callable[[gym.Env], gym.Env]) -> DummyVecEnv:
    def make_thunk():
        def _thunk():
            env = gym.make(env_id, full_action_space=full_action_space)
            env = wrapper(env)
            return env
        return _thunk
    venv = DummyVecEnv([make_thunk()])
    venv = VecFrameStack(venv, n_stack=4)
    venv = VecTransposeImage(venv)  # (H,W,C)->(C,H,W) for CNN policy
    venv = VecMonitor(venv)
    return venv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="ALE/MsPacman-v5")
    p.add_argument("--save-dir", type=str, default="runs/qrdqn_mspacman")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-timesteps", type=int, default=15_000_000)

    # Atari control
    p.add_argument("--full-action-space", action="store_true", help="Use full 18-action set (recommended).")
    p.add_argument("--no-episodic-life", action="store_true", help="Disable EpisodicLife during training.")

    # QRDQN hyperparams (good Atari defaults)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--learning-starts", type=int, default=80_000)
    p.add_argument("--train-freq", type=int, default=4)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--target-update", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--exploration-fraction", type=float, default=0.1)
    p.add_argument("--exploration-final-eps", type=float, default=0.01)
    p.add_argument("--n-quantiles", type=int, default=100)

    # Eval + logging
    p.add_argument("--eval-freq", type=int, default=800_000, help="Per-env steps; effective freq is eval-freq.")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--checkpoint-freq", type=int, default=1_000_000)
    p.add_argument("--target-eval", type=float, default=1500.0)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    train_env = build_vec_env(
        args.env_id,
        seed=args.seed,
        full_action_space=args.full_action_space,
        wrapper=make_train_wrapper(episodic_life=not args.no_episodic_life),
    )
    eval_env = build_vec_env(
        args.env_id,
        seed=args.seed + 123,
        full_action_space=args.full_action_space,
        wrapper=make_eval_wrapper(),
    )

    policy_kwargs = dict(n_quantiles=args.n_quantiles)

    model = QRDQN(
        "CnnPolicy",
        train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,              # every 4 env steps
        gradient_steps=args.gradient_steps,      # 1 gradient step
        target_update_interval=args.target_update,
        batch_size=args.batch_size,
        gamma=args.gamma,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        tensorboard_log=args.save_dir,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
    )

    # Callbacks: eval + early stop on plateau + optional target score + checkpoints
    plateau = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=5, verbose=1)
    thresh = StopTrainingOnRewardThreshold(reward_threshold=args.target_eval, verbose=1) if args.target_eval > 0 else None
    after_eval = plateau if thresh is None else CallbackList([plateau, thresh])

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.save_dir,
        log_path=args.save_dir,
        eval_freq=args.eval_freq,         # for 1 env, no division
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        callback_after_eval=after_eval,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(save_freq=args.checkpoint_freq, save_path=args.save_dir, name_prefix="ckpt")

    model.learn(total_timesteps=args.total_timesteps, callback=CallbackList([eval_cb, ckpt_cb, ProgressBarCallback()]), log_interval=100)

    model.save(os.path.join(args.save_dir, "final_model"))
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
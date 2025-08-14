#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script optimisé pour accélérer l'entraînement d'un agent QRDQN sur ALE/MsPacman-v5.

Principaux changements vs version de base :
- **Parallélisation** via SubprocVecEnv (n_envs configurables) pour accélérer la collecte d'expériences.
- **Frame skipping** (MaxAndSkipEnv) pour réduire le coût de rendu et d'inférence.
- **Images plus petites** (par défaut 80x80) pour une CNN plus légère.
- Hyperparamètres ajustables pour une convergence plus rapide (batch_size, learning_rate, n_quantiles, etc.).
- Entraînement avec EpisodicLife + ClipReward ; évaluation sans ces biais.
- Espace d'actions COMPLET (recommandé sur Atari).

Usage (exemples) :
    python train_mspacman_qrdqn_fast.py \
        --save-dir ../runs/qrdqn_fast \
        --n-envs 8 \
        --frame-size 80 \
        --skip 4 \
        --learning-rate 2.5e-4 \
        --batch-size 64 \
        --n-quantiles 50 \
        --total-timesteps 5000000

Remarque : plus n_envs est grand, plus le GPU est utilisé efficacement, mais la RAM/CPU augmente aussi.
"""

import argparse, os, random
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage, VecMonitor
)
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv, EpisodicLifeEnv, ClipRewardEnv, MaxAndSkipEnv
)
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList
)
from sb3_contrib import QRDQN
import ale_py

# Enregistrer les envs ALE v5 dans Gymnasium
# (utile si l'environnement n'est pas auto-registré par défaut)
gym.register_envs(ale_py)

ENV_ID = "ALE/MsPacman-v5"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def make_train_wrapper(noop_max: int = 30, frame_size: int = 80, skip: int = 4):
    def _wrap(env: gym.Env) -> gym.Env:
        env = NoopResetEnv(env, noop_max=noop_max)
        if skip and skip > 1:
            env = MaxAndSkipEnv(env, skip=skip)
        env = EpisodicLifeEnv(env)                 # episodic life ONLY à l'entraînement
        env = ResizeObservation(env, (frame_size, frame_size))
        env = GrayscaleObservation(env, keep_dim=True)
        env = ClipRewardEnv(env)                    # clip {-1,0,1} (stabilise DQN)
        return env
    return _wrap


def make_eval_wrapper(noop_max: int = 30, frame_size: int = 80, skip: int = 4):
    def _wrap(env: gym.Env) -> gym.Env:
        env = NoopResetEnv(env, noop_max=noop_max)
        if skip and skip > 1:
            env = MaxAndSkipEnv(env, skip=skip)
        env = ResizeObservation(env, (frame_size, frame_size))
        env = GrayscaleObservation(env, keep_dim=True)
        # Pas d'EpisodicLife ni ClipReward en évaluation
        return env
    return _wrap


def build_vec_env(wrapper, n_envs: int, seed: int):
    def make_thunk(rank: int):
        def _thunk():
            env = gym.make(ENV_ID, full_action_space=True)
            env = wrapper(env)
            # Seed déterministe par environnement
            env.reset(seed=seed + rank)
            return env
        return _thunk

    if n_envs <= 1:
        venv = DummyVecEnv([make_thunk(0)])
    else:
        venv = SubprocVecEnv([make_thunk(i) for i in range(n_envs)])

    # Empilement de frames et passage en channels-first pour la CNN
    venv = VecFrameStack(venv, n_stack=4)
    venv = VecTransposeImage(venv)  # (H,W,C)->(C,H,W)
    venv = VecMonitor(venv)
    return venv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save-dir", type=str, default="../runs/qrdqn_mspacman_fast")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    p.add_argument("--eval-freq", type=int, default=100_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--checkpoint-freq", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--frame-size", type=int, default=80)
    p.add_argument("--skip", type=int, default=4, help="Frame skip (>=1). 4 recommandé")
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--n-quantiles", type=int, default=50)
    p.add_argument("--exploration-fraction", type=float, default=0.05)
    p.add_argument("--exploration-final-eps", type=float, default=0.01)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    train_env = build_vec_env(
        make_train_wrapper(frame_size=args.frame_size, skip=args.skip),
        n_envs=args.n_envs,
        seed=args.seed,
    )
    # Pour l'évaluation, 1 env suffit pour des mesures stables
    eval_env = build_vec_env(
        make_eval_wrapper(frame_size=args.frame_size, skip=args.skip),
        n_envs=1,
        seed=args.seed + 10_000,
    )

    policy_kwargs = dict(n_quantiles=args.n_quantiles)

    model = QRDQN(
        "CnnPolicy",
        train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=20_000,          # peut être augmenté si n_envs est grand
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10_000,
        batch_size=args.batch_size,
        gamma=0.99,
        optimize_memory_usage=False,     # compat sb3_contrib (évite conflits)
        max_grad_norm=10.0,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        tensorboard_log=args.save_dir,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.save_dir,
        log_path=args.save_dir,
        eval_freq=max(10_000, args.eval_freq // max(1, args.n_envs)),  # éval plus régulière quand n_envs grand
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(save_freq=args.checkpoint_freq, save_path=args.save_dir, name_prefix="ckpt")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList([eval_cb, ckpt_cb]),
        progress_bar=True,
        log_interval=100,
    )

    model.save(os.path.join(args.save_dir, "fast_model"))
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
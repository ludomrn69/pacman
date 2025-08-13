#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script minimal et optimisé pour entraîner un agent QRDQN sur ALE/MsPacman-v5
avec un objectif de score moyen >= 1500 en évaluation.

Choix clés (simple et efficaces) :
- Wrappers Atari propres : NoopReset(30), Resize(84x84), Grayscale, FrameStack(4)
- Entraînement avec EpisodicLife + ClipReward; Évaluation sans ces biais
- Espace d’actions COMPLET (recommandé sur Atari)
- Meilleur modèle sauvegardé + TensorBoard
- Pas de PER, pas de NoisyNet (inutile/fragile ici) et AUCUN paramètre de timeout
"""

import argparse, os, random
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from stable_baselines3.common.vec_env import (
    DummyVecEnv, VecFrameStack, VecTransposeImage, VecMonitor
)
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv, EpisodicLifeEnv, ClipRewardEnv
)
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList
)
from sb3_contrib import QRDQN
import ale_py

# Enregistrer les envs ALE v5 dans Gymnasium
gym.register_envs(ale_py)

ENV_ID = "ALE/MsPacman-v5"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def make_train_wrapper(noop_max: int = 30):
    def _wrap(env: gym.Env) -> gym.Env:
        env = NoopResetEnv(env, noop_max=noop_max)
        env = EpisodicLifeEnv(env)                 # episodic life ONLY à l'entraînement
        env = ResizeObservation(env, (84, 84))
        env = GrayscaleObservation(env, keep_dim=True)
        env = ClipRewardEnv(env)                    # clip {-1,0,1} (stabilise DQN)
        return env
    return _wrap


def make_eval_wrapper(noop_max: int = 30):
    def _wrap(env: gym.Env) -> gym.Env:
        env = NoopResetEnv(env, noop_max=noop_max)
        env = ResizeObservation(env, (84, 84))
        env = GrayscaleObservation(env, keep_dim=True)
        # Pas d'EpisodicLife ni ClipReward en évaluation
        return env
    return _wrap


def build_vec_env(wrapper):
    def make_thunk():
        def _thunk():
            env = gym.make(ENV_ID, full_action_space=True)
            env = wrapper(env)
            return env
        return _thunk
    venv = DummyVecEnv([make_thunk()])
    venv = VecFrameStack(venv, n_stack=4)
    venv = VecTransposeImage(venv)  # (H,W,C)->(C,H,W) pour la CNN
    venv = VecMonitor(venv)
    return venv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save-dir", type=str, default="../runs/qrdqn_mspacman")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    p.add_argument("--eval-freq", type=int, default=100_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--checkpoint-freq", type=int, default=2_000_000)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    train_env = build_vec_env(make_train_wrapper())
    eval_env = build_vec_env(make_eval_wrapper())

    policy_kwargs = dict(n_quantiles=100)

    model = QRDQN(
        "CnnPolicy",
        train_env,
        learning_rate=1e-4,
        buffer_size=1_000_000,
        learning_starts=20_000,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10_000,
        batch_size=32,
        gamma=0.99,
        optimize_memory_usage=False,  # compat sb3_contrib: pas de param handle_timeout_termination, éviter conflit ReplayBuffer
        max_grad_norm=10.0,
        exploration_fraction=0.02,
        exploration_final_eps=0.01,
        tensorboard_log=args.save_dir,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        device="auto",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.save_dir,
        log_path=args.save_dir,
        eval_freq=args.eval_freq,
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

    model.save(os.path.join(args.save_dir, "final_model"))
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
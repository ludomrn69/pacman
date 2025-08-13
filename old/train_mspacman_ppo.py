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
import time
import random
from pathlib import Path
from typing import Callable


import numpy as np
import torch as th
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold,
    CallbackList, ProgressBarCallback, BaseCallback
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv, NoopResetEnv
import ale_py


gym.register_envs(ale_py)

class RandomActionWrapper(gym.Wrapper):
    """With probability eps, replace the agent action by a random action (training only).
    Helps escape local traps/corners and improves exploration."""
    def __init__(self, env: gym.Env, eps: float = 0.0):
        super().__init__(env)
        self.eps = eps

    def step(self, action):
        if self.eps > 0.0 and np.random.rand() < self.eps:
            action = self.action_space.sample()
        return self.env.step(action)

def make_train_wrapper(noop_max: int = 30, terminal_on_life_loss: bool = True, random_action_eps: float = 0.0) -> Callable[[gym.Env], gym.Env]:
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
        env = RandomActionWrapper(env, eps=random_action_eps)
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


def linear_schedule(start: float, end: float):
    """SB3-style linear schedule from `start` to `end`.
    progress_remaining goes from 1 (start) to 0 (end)."""
    def func(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return func


def _choose_batch_size(buffer_size: int) -> int:
    """Pick a batch size that divides the rollout buffer size (n_steps * n_envs)."""
    for bs in (1024, 512, 256):
        if buffer_size % bs == 0:
            return bs
    return 256


class RandomActionAnnealCallback(BaseCallback):
    """Linearly anneal RandomActionWrapper.eps from start to end over a fraction of training."""
    def __init__(self, venv, start_eps: float, end_eps: float, frac: float, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.venv = venv
        self.start_eps = float(start_eps)
        self.end_eps = float(end_eps)
        self.anneal_steps = max(1, int(total_timesteps * max(0.0, min(1.0, frac))))

    def _set_eps_on_envs(self, eps: float):
        try:
            envs = self.venv.envs
        except Exception:
            envs = []
        for e in envs:
            cur = e
            # Traverse wrapper chain to find RandomActionWrapper
            while hasattr(cur, 'env'):
                if isinstance(cur, RandomActionWrapper):
                    cur.eps = eps
                    break
                cur = cur.env

    def _on_training_start(self) -> None:
        self._set_eps_on_envs(self.start_eps)

    def _on_step(self) -> bool:
        t = self.model.num_timesteps
        if t <= self.anneal_steps:
            alpha = 1.0 - (t / self.anneal_steps)
            eps = self.end_eps + (self.start_eps - self.end_eps) * alpha
        else:
            eps = self.end_eps
        self._set_eps_on_envs(eps)
        return True


class EntropyCoefAnnealCallback(BaseCallback):
    """Linearly anneal model.ent_coef from start to end over total_timesteps."""
    def __init__(self, start: float, end: float, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.start = float(start)
        self.end = float(end)
        self.total = int(max(1, total_timesteps))

    def _on_training_start(self) -> None:
        # initialize to start value
        self.model.ent_coef = float(self.start)

    def _on_step(self) -> bool:
        t = self.model.num_timesteps
        if t <= self.total:
            progress_remaining = 1.0 - (t / self.total)
            coef = self.end + (self.start - self.end) * progress_remaining
        else:
            coef = self.end
        self.model.ent_coef = float(coef)
        return True


# ---- NEW: TimeLimitCallback ----
class TimeLimitCallback(BaseCallback):
    """Stop training after a given wall-clock time (in seconds)."""
    def __init__(self, time_limit_seconds: int, verbose: int = 0):
        super().__init__(verbose)
        self.time_limit_seconds = int(time_limit_seconds)
        self._start = None

    def _on_training_start(self) -> None:
        self._start = time.time()

    def _on_step(self) -> bool:
        if self._start is None:
            return True
        elapsed = time.time() - self._start
        if elapsed >= self.time_limit_seconds:
            if self.verbose > 0:
                print(f"Time limit reached ({self.time_limit_seconds}s). Stopping training.")
            return False
        return True



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
    parser.add_argument("--eval-freq", type=int, default=1_600_000,
                        help="Evaluate every N steps (per environment). Final frequency is eval_freq // n_envs.")
    parser.add_argument("--no-episodic-life", action="store_true", help="Disable EpisodicLife during training (use true game over).")
    parser.add_argument("--checkpoint-freq", type=int, default=500_000)
    parser.add_argument("--full-action-space", action="store_true",
                        help="Use the full 18-action space instead of reduced action set.")
    parser.add_argument("--subproc", action="store_true", help="Use SubprocVecEnv (default).")
    parser.add_argument("--dummy", action="store_true", help="Use DummyVecEnv instead of SubprocVecEnv.")
    parser.add_argument("--anneal-lr", action="store_true", help="Linearly anneal learning rate from lr-start to lr-end.")
    parser.add_argument("--lr-start", type=float, default=3e-4)
    parser.add_argument("--lr-end", type=float, default=5e-5)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--target-kl", type=float, default=0.04)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--clip-range-vf", type=float, default=0.2, help="Clipping range for the value function.")
    parser.add_argument("--pi-sizes", type=str, default="1024,512", help="Comma-separated hidden sizes for the policy MLP head.")
    parser.add_argument("--vf-sizes", type=str, default="1024,512", help="Comma-separated hidden sizes for the value MLP head.")
    parser.add_argument("--random-action-eps", type=float, default=0.03, help="Probability to take a random action during training (exploration boost).")
    parser.add_argument("--random-action-eps-end", type=float, default=0.0, help="Final random action probability after annealing.")
    parser.add_argument("--random-action-anneal-frac", type=float, default=0.8, help="Fraction of total timesteps over which to anneal random-action epsilon.")
    parser.add_argument("--anneal-ent", action="store_true", help="Linearly anneal entropy coefficient from ent-start to ent-end.")
    parser.add_argument("--ent-start", type=float, default=0.06)
    parser.add_argument("--ent-end", type=float, default=0.005)
    parser.add_argument("--target-eval", type=float, default=1500.0, help="Stop early if eval/mean_reward >= this threshold (0=disabled).")
    args = parser.parse_args()

    # Sanity
    os.makedirs(args.save_dir, exist_ok=True)
    tb_log = args.save_dir

    # Seeds
    set_global_seeds(args.seed)

    # Choose VecEnv backend
    vec_cls = DummyVecEnv if args.dummy else SubprocVecEnv

    # Training and evaluation environments
    train_env = build_venv(
        args.env_id,
        args.n_envs,
        args.seed,
        make_train_wrapper(
            terminal_on_life_loss=(not args.no_episodic_life),
            random_action_eps=args.random_action_eps,
        ),
        args.full_action_space,
        vec_env_cls=vec_cls,
    )

    # Note: eval env without EpisodicLife, deterministic policy
    eval_env = build_venv(args.env_id, 1, args.seed + 10,
                          make_eval_wrapper(), args.full_action_space, vec_env_cls=DummyVecEnv)

    # Auto-pick a batch size that divides the buffer size
    _buffer_size = args.n_steps * args.n_envs
    _batch_size = _choose_batch_size(_buffer_size)
    # Build policy kwargs with larger MLP heads for Atari (helps break plateaus)
    pi_sizes = tuple(int(x) for x in args.pi_sizes.split(",") if x)
    vf_sizes = tuple(int(x) for x in args.vf_sizes.split(",") if x)
    policy_kwargs = dict(net_arch=[dict(pi=list(pi_sizes), vf=list(vf_sizes))])
    # PPO hyperparameters (optimized defaults for Atari stability)
    model = PPO(
        "CnnPolicy",
        train_env,
        n_steps=args.n_steps,
        batch_size=_batch_size,
        n_epochs=args.n_epochs,
        learning_rate=linear_schedule(args.lr_start, args.lr_end) if args.anneal_lr else args.lr_start,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=args.clip_range,
        clip_range_vf=args.clip_range_vf,
        # Start with ent_start if we plan to anneal via callback; otherwise use fixed ent_coef
        ent_coef=(args.ent_start if args.anneal_ent else args.ent_coef),
        vf_coef=0.5,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        policy_kwargs=policy_kwargs,
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
    threshold_cb = None
    if args.target_eval and args.target_eval > 0:
        threshold_cb = StopTrainingOnRewardThreshold(reward_threshold=args.target_eval, verbose=1)
    after_eval_cb = early_stop if threshold_cb is None else CallbackList([early_stop, threshold_cb])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.save_dir,
        log_path=args.save_dir,
        eval_freq=eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        callback_after_eval=after_eval_cb,  # plateau + optional reward threshold
        verbose=1
    )
    checkpoint_callback = CheckpointCallback(save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                                             save_path=args.save_dir, name_prefix="ckpt")
    callbacks_list = [eval_callback, checkpoint_callback, ProgressBarCallback()]
    if args.random_action_eps > args.random_action_eps_end:
        callbacks_list.append(RandomActionAnnealCallback(train_env, args.random_action_eps, args.random_action_eps_end, args.random_action_anneal_frac, args.total_timesteps))
    if args.anneal_ent:
        callbacks_list.append(EntropyCoefAnnealCallback(args.ent_start, args.ent_end, args.total_timesteps))
    callbacks = CallbackList(callbacks_list)

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
        # Seed the vectorized eval env
        try:
            eval_env.seed(s)
        except Exception:
            pass
        obs = eval_env.reset()
        ep_return = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_env.step(action)
            ep_return += float(rewards[0])
            if dones[0]:
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

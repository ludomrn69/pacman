import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

import ale_py


gym.register_envs(ale_py)


# --- Traitement de l'image (comme dans train_target.py) ---
def transform_img(obs):
    # Prend les 170 premières lignes et canal rouge
    return obs[:170, :, 0].astype(np.float32)


# --- Réseau de neurones convolutionnel ---
class DQNCNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        c = 4  # We are stacking 4 frames
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, input_shape[0], input_shape[1])
            n_flatten = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = (x / 128.0) - 1.0  # Normalisation
        return self.fc(self.conv(x))


# --- Experience Replay ---
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)


# --- Environnement + Entraînement ---
def train():
    env = gym.make("ALE/MsPacman-v5", render_mode=None)
    obs, info = env.reset()
    print(f"[DEBUG] shape obs au reset: {obs.shape}")
    obs = transform_img(obs)
    obs_shape = obs.shape
    num_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Réseaux
    policy_net = DQNCNN(obs_shape, num_actions).to(device)
    target_net = DQNCNN(obs_shape, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    replay = ReplayBuffer(100_000)

    # Hyperparamètres
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    min_epsilon = 0.1
    decay = 0.9995
    total_steps = 500_000
    target_update_freq = 1
    frame_stack = deque(maxlen=4)

    obs, _ = env.reset()
    obs = transform_img(obs)
    frame_stack.extend([obs] * 4)
    state = np.stack(frame_stack, axis=0)

    print("[DEBUG] Début de la boucle principale...")
    for step in range(total_steps):
        print(f"[DEBUG] step={step}")
        # Choix de l'action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state).to(device).unsqueeze(0)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()
        print(f"[DEBUG] action choisie: {action}")

        # Action
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(f"[DEBUG] reward: {reward}, terminated: {terminated}, truncated: {truncated}")
        print(f"[DEBUG] shape next_obs: {next_obs.shape}")
        done = terminated or truncated
        next_obs = transform_img(next_obs)
        frame_stack.append(next_obs)
        if len(frame_stack) != 4:
            print(f"[ERREUR] frame_stack a une mauvaise taille: {len(frame_stack)}")
        next_state = np.stack(frame_stack, axis=0)

        replay.add((state, action, reward, next_state, done))
        state = next_state

        if done:
            obs, _ = env.reset()
            print(f"[DEBUG] obs reset shape: {obs.shape}")
            obs = transform_img(obs)
            frame_stack.extend([obs] * 4)
            state = np.stack(frame_stack, axis=0)

        # Apprentissage
        if len(replay) >= batch_size:
            states, actions, rewards, next_states, dones = replay.sample(batch_size)
            states = torch.tensor(states, dtype=torch.float32).to(device)
            actions = torch.tensor(actions).unsqueeze(1).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).to(device)

            q_vals = policy_net(states).gather(1, actions).squeeze()
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0]
                target = rewards + gamma * max_next_q * (1 - dones)

            loss = nn.functional.mse_loss(q_vals, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target network
        if step % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * decay)

        if step % 5000 == 0:
            print(f"Étape: {step}, ε={epsilon:.3f}, Dernière récompense: {reward}")

    torch.save(policy_net.state_dict(), "dqn_pacman_final.pt")
    env.close()


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"[ERREUR] Exception détectée : {e}")
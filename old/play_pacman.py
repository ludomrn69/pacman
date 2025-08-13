import gymnasium as gym
from stable_baselines3 import PPO
import ale_py


gym.register_envs(ale_py)

# Charge le modèle entraîné
model = PPO.load("logs/best_model")

# Environnement avec rendu visuel
env = gym.make("ALE/MsPacman-v5", render_mode="human")

obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

env.close()
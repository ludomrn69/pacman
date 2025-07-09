import gymnasium as gym
from stable_baselines3 import PPO
import ale_py
import time


gym.register_envs(ale_py)

# Crée l’environnement (MsPacman est un peu spécial, on va simplifier)
env = gym.make("ALE/MsPacman-v5", render_mode=None)  # pas de fenêtre pour l’entraînement

# Crée l’agent
model = PPO(
    "CnnPolicy",
    env,
    n_steps=128,
    batch_size=32,
    ent_coef=0.01,
    clip_range=0.1
)


# Entraîne l’agent pendant 10000 étapes
print("🧠 Début de l'entraînement de l'agent...")

def training_callback(locals_, globals_):
    n_steps = locals_["self"].num_timesteps
    if n_steps % 1000 == 0:
        print(f"🔄 Étapes effectuées: {n_steps}")
    return True

model.learn(total_timesteps=10000, callback=training_callback)
print("✅ Entraînement terminé.")

# Sauvegarde le modèle
model.save("ppo_pacman")
print("💾 Modèle sauvegardé sous 'ppo_pacman.zip'")

env.close()
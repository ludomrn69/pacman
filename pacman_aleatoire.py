import gymnasium as gym
import ale_py


gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5", render_mode="human")
obs, info = env.reset()
done = False

total_reward = 0  # 🔹 Ajouté

while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    total_reward += reward  # 🔹 Ajouté
    done = terminated or truncated

env.close()
print("Score final :", total_reward)  # 🔹 Ajouté
import gymnasium as gym
import numpy as np
import tensorflow as tf
import time
import cv2
import ale_py


gym.register_envs(ale_py)
from keras.saving import register_keras_serializable
from keras import layers

@register_keras_serializable()
class NoisyDense(layers.Layer):
    def __init__(self, units, activation=None, sigma_init=0.017, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.activation = layers.Activation(activation) if activation else None
        self.sigma_init = sigma_init

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.mu_w = self.add_weight(name="mu_w", shape=[input_dim, self.units],
                                    initializer=tf.keras.initializers.RandomUniform(-1/np.sqrt(input_dim), 1/np.sqrt(input_dim)))
        self.sigma_w = self.add_weight(name="sigma_w", shape=[input_dim, self.units],
                                       initializer=tf.keras.initializers.Constant(self.sigma_init))
        self.mu_b = self.add_weight(name="mu_b", shape=[self.units],
                                    initializer=tf.keras.initializers.RandomUniform(-1/np.sqrt(input_dim), 1/np.sqrt(input_dim)))
        self.sigma_b = self.add_weight(name="sigma_b", shape=[self.units],
                                       initializer=tf.keras.initializers.Constant(self.sigma_init))

    def call(self, x):
        epsilon_w = tf.random.normal(self.mu_w.shape)
        epsilon_b = tf.random.normal(self.mu_b.shape)
        w = self.mu_w + self.sigma_w * epsilon_w
        b = self.mu_b + self.sigma_b * epsilon_b
        x = tf.matmul(x, w) + b
        if self.activation:
            x = self.activation(x)
        return x

# Charger le modèle entraîné
model = tf.keras.models.load_model("my_model_target.keras", custom_objects={"NoisyDense": NoisyDense})

# Paramètres de traitement d'image (doivent être cohérents avec l'entraînement)
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized / 255.0

def stack_frames(frames, state, is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode:
        frames = [frame] * 6
    else:
        frames.append(frame)
        frames = frames[-6:]
    stacked_state = np.stack(frames, axis=-1)
    return frames, stacked_state

# Lancer l'environnement
env = gym.make("ALE/MsPacman-v5", render_mode="human")
num_episodes = 3

print("Actions disponibles :", env.unwrapped.get_action_meanings())
print("Nombre d'actions :", env.action_space.n)

for episode in range(num_episodes):
    obs, _ = env.reset()
    frames = []
    frames, state = stack_frames(frames, obs, True)
    done = False
    total_reward = 0

    while not done:
        input_tensor = np.expand_dims(state, axis=0).astype(np.float32)  # batch dimension
        q_values = model.predict(input_tensor, verbose=0)
        # Reconstruction de la valeur espérée depuis la distribution (C51)
        v_min, v_max = -10, 10
        n_atoms = 51
        z = np.linspace(v_min, v_max, n_atoms)  # (51,)

        probs = q_values[0]  # (9, 51)
        q_expected = np.sum(probs * z, axis=1)  # (9,)
        action_index = np.argmax(q_expected)
        print("Q-values shape :", q_values.shape)
        print("Action choisie (via espérance):", action_index)
        action = action_index

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        frames, state = stack_frames(frames, obs, False)
        total_reward += reward
        time.sleep(0.01)

    print(f"Épisode {episode+1} terminé avec un score de {total_reward}")

env.close()
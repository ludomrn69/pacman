import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tqdm import trange
from collections import deque, namedtuple

import gymnasium as gym
import cv2
import sys
import tensorflow as tf
from keras import models, layers
from keras import backend as K
from keras.saving import register_keras_serializable
import numpy as np
import time
import matplotlib.pyplot as plot
import ale_py


gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
print("Liste des actions", env.unwrapped.get_action_meanings())
nbr_action = env.action_space.n

file_model='my_model_target'
file_stats='tab_score_target'

gamma=tf.constant(0.99)
tau = 0.01
n_step = 5
gamma_n = gamma ** n_step
n_step_buffer = deque(maxlen=n_step)

n_atoms = 51
v_min = -10
v_max = 10
delta_z = (v_max - v_min) / (n_atoms - 1)
z = tf.cast(tf.linspace(v_min, v_max, n_atoms), tf.float32)

class LazyFrames:
  def __init__(self, frames):
    self._frames = frames

  def __array__(self, dtype=None):
    out = np.concatenate(self._frames, axis=-1)
    if dtype is not None:
      out = out.astype(dtype)
    return out

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
    epsilon_w = tf.random.normal([tf.shape(x)[-1], self.units])
    epsilon_b = tf.random.normal([self.units])
    w = self.mu_w + self.sigma_w * epsilon_w
    b = self.mu_b + self.sigma_b * epsilon_b
    out = tf.matmul(x, w) + b
    return self.activation(out) if self.activation else out

  def get_config(self):
      config = super().get_config()
      config.update({
          "units": self.units,
          "activation": self.activation.activation if self.activation else None,
          "sigma_init": self.sigma_init
      })
      return config

n_epochs=150
decalage_debut=90
taille_sequence=6
games_per_epoch=300
pourcentage_batch=0.20
best_score=-np.inf

epsilon = 1.0  # Starting value
epsilon_min = 0.03  # Minimum value
epsilon_decay_steps = 20_000  # Number of steps for linear decay
epsilon_decay_rate = (epsilon - epsilon_min) / epsilon_decay_steps

tab_s = deque(maxlen=200)

class PrioritizedReplayBuffer:
  def __init__(self, capacity=100_000, alpha=0.6):
    self.capacity = capacity
    self.alpha = alpha
    self.buffer = deque(maxlen=capacity)
    self.priorities = deque(maxlen=capacity)

  def add(self, experience, td_error=1.0):
    self.buffer.append(experience)
    self.priorities.append(abs(td_error) + 1e-6)

  def sample(self, batch_size=16, beta=0.4):
    scaled_priorities = np.array(self.priorities) ** self.alpha
    probs = scaled_priorities / scaled_priorities.sum()
    indices = np.random.choice(len(self.buffer), batch_size, p=probs)
    weights = (len(self.buffer) * probs[indices]) ** (-beta)
    weights /= weights.max()
    samples = [self.buffer[i] for i in indices]
    return samples, indices, weights

  def update_priorities(self, indices, td_errors):
    for i, err in zip(indices, td_errors):
      self.priorities[i] = abs(err) + 1e-6

replay_buffer = PrioritizedReplayBuffer(capacity=50_000)
batch_size = 16

def model(nbr_cc=32):
  entree = layers.Input(shape=(84, 84, taille_sequence), dtype='float32')
  x = layers.Conv2D(32, 8, strides=4, activation='relu')(entree)
  x = layers.Conv2D(64, 4, strides=2, activation='relu')(x)
  x = layers.Conv2D(64, 3, strides=1, activation='relu')(x)
  x = layers.Flatten()(x)
  
  # C51 output: n_actions x n_atoms
  output = NoisyDense(nbr_action * n_atoms)(x)
  output = layers.Reshape((nbr_action, n_atoms))(output)
  output = layers.Softmax(axis=-1)(output)  # Probabilités par action
  return models.Model(inputs=entree, outputs=output)

def transform_img(image):
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
  return np.expand_dims(resized, axis=-1).astype(np.float32) / 255.0

def simulation(epsilon, debug=False):
  global tab_s
  if debug:
    start_time=time.time()

  tab_observations=[]
  tab_rewards=[]
  tab_actions=[]
  tab_next_observations=[]
  tab_done=[]
  
  ######
  observations, _ = env.reset()
  vie=3
  for i in range(decalage_debut-taille_sequence):
    observation, reward, terminated, truncated, info = env.step(0)
    done = terminated or truncated
  tab_sequence=[]
  for i in range(taille_sequence):
    observation, reward, terminated, truncated, info = env.step(0)
    done = terminated or truncated
    img=transform_img(observation)
    tab_sequence.append(img)
  tab_sequence=LazyFrames(tab_sequence)
  ######

  score=0
  while True:
    if np.random.random()>epsilon:
      dist=model_primaire(np.expand_dims(np.array(tab_sequence), axis=0))
      q_values = tf.reduce_sum(z * dist, axis=2)
      action=int(tf.argmax(q_values[0], axis=-1))
    else:
      action=np.random.randint(0, nbr_action)

    h=np.random.randint(10)
    if h==0:
      tab_observations.append(np.array(tab_sequence))
      tab_actions.append(action)
    n_step_buffer.append((np.array(tab_sequence), action, reward))
    score+=reward
    if 'ale.lives' in info and info['ale.lives'] < vie:
      reward=-50.
      vie=info['ale.lives']
      if h==0:
        tab_done.append(True)
    else:
      if h==0:
        tab_done.append(done)
    if h==0:
      tab_rewards.append(reward)
    if done:
      tab_s.append(score)
      if h==0:
        tab_sequence._frames[:-1] = tab_sequence._frames[1:]
        tab_sequence._frames[taille_sequence - 1] = img
        tab_next_observations.append(np.array(tab_sequence))
      tab_done=np.array(tab_done, dtype=np.float32)
      tab_observations=np.array(tab_observations, dtype=np.float32)
      tab_next_observations=np.array(tab_next_observations, dtype=np.float32)
      tab_rewards=np.array(tab_rewards, dtype=np.float32)
      tab_rewards = np.clip(tab_rewards, -1, 1)
      tab_rewards += 0.3
      tab_rewards = tab_rewards.reshape(-1)
      tab_actions=np.array(tab_actions, dtype=np.int32)
      if debug:
            print("  Creation observations {:5.3f} seconde(s)".format(float(time.time()-start_time)))
            print("     score:{:5d}   batch:{:4d}".format(int(score), len(tab_done)))
      for o, r, a, no, d in zip(tab_observations, tab_rewards, tab_actions, tab_next_observations, tab_done):
        replay_buffer.add((o, r, a, no, d))
      if len(n_step_buffer) == n_step:
        obs_n, act_n, _ = n_step_buffer[0]
        _, _, reward_n = n_step_buffer[-1]
        R = sum([r * (gamma ** i) for i, (_, _, r) in enumerate(n_step_buffer)])
        replay_buffer.add((np.array(obs_n), R, act_n, np.array(tab_sequence), done))
      n_step_buffer.clear()
      return
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    img=transform_img(observation)
    tab_sequence._frames[:-1]=tab_sequence._frames[1:]
    tab_sequence._frames[taille_sequence-1]=img
    if h==0:
      tab_next_observations.append(np.array(tab_sequence))
    if len(n_step_buffer) == n_step:
      obs_n, act_n, _ = n_step_buffer[0]
      _, _, reward_n = n_step_buffer[-1]
      R = sum([r * (gamma ** i) for i, (_, _, r) in enumerate(n_step_buffer)])
      replay_buffer.add((np.array(obs_n), R, act_n, np.array(tab_sequence), done))

def my_loss(y, q):
  return tf.reduce_mean(tf.keras.losses.huber(y, q))

def train_step():
  samples, indices, weights = replay_buffer.sample(batch_size)
  obs, rew, act, next_obs, done = zip(*samples)
  # Ensure arrays and correct shapes
  obs = tf.convert_to_tensor(np.array(obs), dtype=tf.float32)
  rew = tf.convert_to_tensor(np.array(rew), dtype=tf.float32)
  act = tf.convert_to_tensor(np.array(act), dtype=tf.int32)
  next_obs = tf.convert_to_tensor(np.array(next_obs), dtype=tf.float32)
  done = tf.convert_to_tensor(np.array(done), dtype=tf.float32)
  weights = tf.convert_to_tensor(weights, dtype=tf.float32)

  next_dist = model_cible(next_obs)  # (batch_size, n_actions, n_atoms)
  next_q = tf.reduce_sum(z * next_dist, axis=2)  # Espérance
  best_next_actions = tf.argmax(next_q, axis=1)

  target_dist = tf.zeros((batch_size, nbr_action, n_atoms), dtype=tf.float32)
  for i in range(batch_size):
      proj_dist = tf.zeros((n_atoms,), dtype=tf.float32)
      for j in range(n_atoms):
          tzj = tf.clip_by_value(rew[i] + (1 - done[i]) * gamma * z[j], v_min, v_max)
          bj = (tzj - v_min) / delta_z
          l = tf.cast(tf.math.floor(bj), tf.int32)
          u = tf.cast(tf.math.ceil(bj), tf.int32)
          if l == u:
              proj_dist = proj_dist + next_dist[i, best_next_actions[i], j] * tf.one_hot(l, n_atoms)
          else:
              proj_dist = proj_dist + next_dist[i, best_next_actions[i], j] * (
                  (tf.cast(u, tf.float32) - bj) * tf.one_hot(l, n_atoms) + (bj - tf.cast(l, tf.float32)) * tf.one_hot(u, n_atoms)
              )
      target_dist = tf.tensor_scatter_nd_update(target_dist, [[i, act[i]]], [proj_dist])

  with tf.GradientTape() as tape:
      all_dists = model_primaire(obs)
      log_pred = tf.math.log(tf.clip_by_value(all_dists, 1e-6, 1.0))
      loss = tf.keras.losses.categorical_crossentropy(target_dist, log_pred)
      loss = tf.reduce_sum(loss * tf.expand_dims(weights, axis=1), axis=1)
      loss = tf.reduce_mean(loss)
  gradients=tape.gradient(loss, model_primaire.trainable_variables)
  gradients = [tf.clip_by_norm(g, 10.0) for g in gradients]
  optimizer.apply_gradients(zip(gradients, model_primaire.trainable_variables))
  train_loss(loss)
  loss_history.append(loss.numpy())
  td_errors = tf.abs(tf.reduce_sum(target_dist * z, axis=2) - tf.reduce_sum(all_dists * z, axis=2))
  td_errors_np = tf.stop_gradient(tf.reduce_max(td_errors, axis=1)).numpy().flatten()
  replay_buffer.update_priorities(indices, td_errors_np)
  
def train(debug=False):
  global epsilon, best_score
  tab_s.clear()
  total_steps = 0
  for e in trange(n_epochs, desc="Training Epochs"):
    for i in trange(games_per_epoch, desc=f"Epoch {e:04d}", leave=False):
      print("Epoch {:04d}/{:05d} | Jeu {:03d}/{:03d} | epsilon={:05.3f}".format(e, n_epochs, i, games_per_epoch, epsilon))
      simulation(epsilon, debug=True)
      if debug:
        start_time=time.time()
      if len(replay_buffer.buffer) > 5000:
        train_step()
      if debug:
        print("  Entrainement {:5.3f} seconde(s)".format(float(time.time()-start_time)))
        print("     loss: {:6.4f}".format(train_loss.result()))
      total_steps += 1
      if epsilon > epsilon_min:
          epsilon -= epsilon_decay_rate
          epsilon = max(epsilon, epsilon_min)
      train_loss.reset_state()

    print("Copie des poids primaire -> cible")
    for a, b in zip(model_cible.variables, model_primaire.variables):
      a.assign(tau * b + (1 - tau) * a)
    
    np.save(file_stats, tab_s)
    tab_s_list = list(tab_s)
    recent_mean_score = np.mean(tab_s)
    if recent_mean_score > best_score:
      print("Sauvegarde du modele")
      model_cible.save(file_model + ".keras")
      best_score = recent_mean_score

  from scipy.ndimage import uniform_filter1d

  tab_s_list = list(tab_s)
  moving_avg = uniform_filter1d(tab_s_list, size=20)
  max_scores = [max(tab_s_list[max(0, i - games_per_epoch):i + 1]) for i in range(len(tab_s_list))]
  cumulative_rewards = np.cumsum(tab_s_list) / (np.arange(len(tab_s_list)) + 1)
  epsilons = [max(1.0 - (i * epsilon_decay_rate), epsilon_min) for i in range(len(tab_s_list))]

  import matplotlib.pyplot as plt

  plt.figure(figsize=(15, 10))

  plt.subplot(2, 2, 1)
  plt.plot(tab_s_list, label="Score brut")
  plt.plot(moving_avg, label="Moyenne glissante (20 jeux)")
  plt.plot(max_scores, label="Score max/epoch")
  plt.legend()
  plt.title("Scores au fil des jeux")
  plt.xlabel("Jeu")
  plt.ylabel("Score")

  plt.subplot(2, 2, 2)
  plt.plot(epsilons)
  plt.title("Évolution de l'epsilon")
  plt.xlabel("Jeu")
  plt.ylabel("Epsilon")

  plt.subplot(2, 2, 3)
  plt.plot(cumulative_rewards)
  plt.title("Reward cumulé moyen")
  plt.xlabel("Jeu")
  plt.ylabel("Reward moyen")

  plt.subplot(2, 2, 4)
  plt.plot(loss_history)
  plt.title("Courbe de perte (loss)")
  plt.xlabel("Itération d'entraînement")
  plt.ylabel("Loss")

  plt.tight_layout()
  plt.show()

loss_history = []

model_primaire=model(16)
model_primaire.summary()
model_cible=tf.keras.models.clone_model(model_primaire)
for a, b in zip(model_cible.variables, model_primaire.variables):
  a.assign(b)

optimizer = tf.keras.optimizers.Adam(learning_rate=2E-4)
train_loss=tf.keras.metrics.Mean()
train(debug=True)

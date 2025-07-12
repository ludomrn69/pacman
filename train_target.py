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
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import time
import matplotlib.pyplot as plt
import ale_py

# Experience tuple for n-step learning
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# For manual frame stacking in simulation
class LazyFrames:
    def __init__(self, frames):
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=-1)
        if dtype is not None:
            out = out.astype(dtype)
        return out

gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5")
# standard Atari preprocessing: grayscale, resize, frame-skip, reward clipping
env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=True)
# stack 4 frames
env = FrameStack(env, num_stack=4)
print("Liste des actions", env.env.unwrapped.get_action_meanings())
nbr_action = env.action_space.n

file_model = 'my_model_target'
file_stats = 'tab_score_target'

# Hyperparameters - Version corrigée
gamma = tf.constant(0.99)
tau = 0.005  # Soft update plus conservateur
n_step = 3   # N-step réduit pour plus de stabilité
gamma_n = gamma ** n_step

# Distributional RL parameters
n_atoms = 51
v_min = -10
v_max = 10
delta_z = (v_max - v_min) / (n_atoms - 1)
z = tf.cast(tf.linspace(v_min, v_max, n_atoms), tf.float32)

# Training parameters - Version corrigée
n_epochs = 300
games_per_epoch = 100  # Réduit pour feedback plus rapide
batch_size = 32
update_frequency = 4   # Update tous les 4 jeux
target_update_frequency = 1000  # Update target moins fréquent
best_score = -np.inf

# Epsilon parameters - Version corrigée
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay_steps = 50_000  # Decay plus lent
epsilon_decay_rate = (epsilon - epsilon_min) / epsilon_decay_steps

# Prioritized Replay parameters
beta_start = 0.4
beta_frames = n_epochs * games_per_epoch
alpha = 0.6

tab_s = deque(maxlen=1000)  # Garde plus d'historique

@register_keras_serializable()
class NoisyDense(layers.Layer):
    def __init__(self, units, activation=None, sigma_init=0.5, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.activation = layers.Activation(activation) if activation else None
        self.sigma_init = sigma_init

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        # Factorized noise
        self.mu_w = self.add_weight(
            name="mu_w", shape=[input_dim, self.units],
            initializer=tf.keras.initializers.RandomUniform(-1/np.sqrt(input_dim), 1/np.sqrt(input_dim))
        )
        self.sigma_w = self.add_weight(
            name="sigma_w", shape=[input_dim, self.units],
            initializer=tf.keras.initializers.Constant(self.sigma_init/np.sqrt(input_dim))
        )
        self.mu_b = self.add_weight(
            name="mu_b", shape=[self.units],
            initializer=tf.keras.initializers.Zeros()
        )
        self.sigma_b = self.add_weight(
            name="sigma_b", shape=[self.units],
            initializer=tf.keras.initializers.Constant(self.sigma_init/np.sqrt(input_dim))
        )

    def call(self, x):
        # Factorized Gaussian noise
        input_size = tf.shape(x)[-1]
        epsilon_input = tf.random.normal([input_size])
        epsilon_output = tf.random.normal([self.units])
        
        # f(x) = sgn(x) * sqrt(|x|)
        f_epsilon_input = tf.sign(epsilon_input) * tf.sqrt(tf.abs(epsilon_input))
        f_epsilon_output = tf.sign(epsilon_output) * tf.sqrt(tf.abs(epsilon_output))
        
        # Outer product for weight noise
        epsilon_w = tf.expand_dims(f_epsilon_input, 1) * tf.expand_dims(f_epsilon_output, 0)
        epsilon_b = f_epsilon_output
        
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

class PrioritizedReplayBuffer:
    def __init__(self, capacity=100_000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def add(self, experience):
        self.buffer.append(experience)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size=32, beta=0.4):
        if len(self.buffer) < batch_size:
            return None, None, None
            
        scaled_priorities = np.array(self.priorities) ** self.alpha
        probs = scaled_priorities / scaled_priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        samples = [self.buffer[i] for i in indices]
        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for i, error in zip(indices, td_errors):
            priority = abs(error) + 1e-6
            self.priorities[i] = priority
            self.max_priority = max(self.max_priority, priority)

replay_buffer = PrioritizedReplayBuffer(capacity=100_000, alpha=alpha)

def create_model():
    entree = layers.Input(shape=(84, 84, 4), dtype='float32')
    
    # Convolutional layers
    x = layers.Conv2D(32, 8, strides=4, activation='relu')(entree)
    x = layers.Conv2D(64, 4, strides=2, activation='relu')(x)
    x = layers.Conv2D(64, 3, strides=1, activation='relu')(x)
    x = layers.Flatten()(x)
    
    # Dueling architecture
    # Value stream
    v = layers.Dense(512, activation='relu')(x)
    v = NoisyDense(n_atoms)(v)
    v = layers.Reshape((1, n_atoms))(v)
    
    # Advantage stream
    a = layers.Dense(512, activation='relu')(x)
    a = NoisyDense(nbr_action * n_atoms)(a)
    a = layers.Reshape((nbr_action, n_atoms))(a)
    
    # Combine streams into Q-distribution
    q_atoms = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
    out = layers.Softmax(axis=-1)(q_atoms)
    
    return models.Model(inputs=entree, outputs=out)

def transform_img(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return np.expand_dims(resized, axis=-1).astype(np.float32) / 255.0

class NStepBuffer:
    def __init__(self, n_step, gamma):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)
        
    def add(self, experience):
        self.buffer.append(experience)
        
    def get_n_step_experience(self):
        if len(self.buffer) < self.n_step:
            return None
            
        # Calculate n-step return
        n_step_return = 0
        for i, exp in enumerate(self.buffer):
            n_step_return += (self.gamma ** i) * exp.reward
            
        # Get first and last experience
        first_exp = self.buffer[0]
        last_exp = self.buffer[-1]
        
        return Experience(
            state=first_exp.state,
            action=first_exp.action,
            reward=n_step_return,
            next_state=last_exp.next_state,
            done=last_exp.done
        )
    
    def clear(self):
        self.buffer.clear()

def simulation(epsilon, debug=False):
    global tab_s
    
    if debug:
        start_time = time.time()
    
    n_step_buffer = NStepBuffer(n_step, gamma.numpy())
    
    observations, _ = env.reset()
    score = 0
    step_count = 0
    
    # Skip initial frames
    for _ in range(90):
        observation, reward, terminated, truncated, info = env.step(0)
        done = terminated or truncated
        if done:
            break
    
    # Initialize frame sequence
    tab_sequence = []
    for _ in range(4):  # 4 frames pour le stacking
        observation, reward, terminated, truncated, info = env.step(0)
        done = terminated or truncated
        if done:
            break
        img = transform_img(observation)
        tab_sequence.append(img)
    
    if done:
        tab_s.append(0)
        return
        
    tab_sequence = LazyFrames(tab_sequence)
    
    while True:
        # Action selection with noisy networks (no epsilon needed)
        dist = model_primaire(np.expand_dims(np.array(tab_sequence), axis=0))
        q_values = tf.reduce_sum(z * dist, axis=2)
        action = int(tf.argmax(q_values[0], axis=-1))
        
        # Store current state
        current_state = np.array(tab_sequence)
        
        # Execute action
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Process reward - Version corrigée
        original_reward = reward
        reward = np.clip(reward, -1, 1)  # Reward clipping
        
        # Gestion des vies - Version corrigée
        life_lost = False
        if 'ale.lives' in info:
            if not hasattr(simulation, 'previous_lives'):
                simulation.previous_lives = info['ale.lives']
            
            if info['ale.lives'] < simulation.previous_lives:
                life_lost = True
                reward = -1.0  # Pénalité raisonnable au lieu de -50
                simulation.previous_lives = info['ale.lives']
        
        score += original_reward
        
        # Update frame sequence
        img = transform_img(observation)
        tab_sequence._frames[:-1] = tab_sequence._frames[1:]
        tab_sequence._frames[-1] = img
        next_state = np.array(tab_sequence)
        
        # Create experience
        experience = Experience(
            state=current_state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done or life_lost
        )
        
        # Add to n-step buffer
        n_step_buffer.add(experience)
        
        # Add regular experience to replay buffer
        replay_buffer.add(experience)
        
        # Add n-step experience if available
        n_step_exp = n_step_buffer.get_n_step_experience()
        if n_step_exp is not None:
            replay_buffer.add(n_step_exp)
        
        step_count += 1
        
        if done:
            # Clear n-step buffer and add remaining experiences
            n_step_buffer.clear()
            
            tab_s.append(score)
            
            if debug:
                print(f"  Création observations {time.time()-start_time:.3f} seconde(s)")
                print(f"     score: {int(score):5d}   steps: {step_count:4d}")
            
            # Reset previous lives counter
            if hasattr(simulation, 'previous_lives'):
                delattr(simulation, 'previous_lives')
            
            return

@tf.function
def train_step(samples, weights, beta):
    obs, actions, rewards, next_obs, dones = zip(*samples)
    
    obs = tf.convert_to_tensor(np.array(obs), dtype=tf.float32)
    actions = tf.convert_to_tensor(np.array(actions), dtype=tf.int32)
    rewards = tf.convert_to_tensor(np.array(rewards), dtype=tf.float32)
    next_obs = tf.convert_to_tensor(np.array(next_obs), dtype=tf.float32)
    dones = tf.convert_to_tensor(np.array(dones), dtype=tf.float32)
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    
    # Double DQN: use online network to select actions
    next_dist_online = model_primaire(next_obs)
    next_q_online = tf.reduce_sum(z * next_dist_online, axis=2)
    next_actions = tf.argmax(next_q_online, axis=1)
    
    # Use target network to evaluate actions
    next_dist_target = model_cible(next_obs)
    
    # Distributional Bellman operator
    batch_size = tf.shape(obs)[0]
    target_dist = tf.zeros((batch_size, nbr_action, n_atoms), dtype=tf.float32)
    
    for i in range(batch_size):
        # Project the distribution
        m = tf.zeros((n_atoms,), dtype=tf.float32)
        
        for j in range(n_atoms):
            # Bellman update
            tzj = tf.clip_by_value(
                rewards[i] + (1 - dones[i]) * gamma * z[j], 
                v_min, v_max
            )
            bj = (tzj - v_min) / delta_z
            l = tf.cast(tf.floor(bj), tf.int32)
            u = tf.cast(tf.ceil(bj), tf.int32)
            
            # Distribute probability mass
            prob = next_dist_target[i, next_actions[i], j]
            if l == u:
                indices = tf.stack([l])
                updates = tf.stack([prob])
            else:
                l_weight = tf.cast(u, tf.float32) - bj
                u_weight = bj - tf.cast(l, tf.float32)
                indices = tf.stack([l, u])
                updates = tf.stack([prob * l_weight, prob * u_weight])
            
            # Ensure indices are within bounds
            indices = tf.clip_by_value(indices, 0, n_atoms - 1)
            m = tf.tensor_scatter_nd_add(m, tf.expand_dims(indices, 1), updates)
        
        # Update target distribution
        target_dist = tf.tensor_scatter_nd_update(
            target_dist, [[i, actions[i]]], [m]
        )
    
    with tf.GradientTape() as tape:
        # Get current distributions
        current_dist = model_primaire(obs)
        
        # Extract distributions for taken actions
        indices = tf.stack([tf.range(batch_size), actions], axis=1)
        current_action_dist = tf.gather_nd(current_dist, indices)
        target_action_dist = tf.gather_nd(target_dist, indices)
        
        # Cross-entropy loss
        log_probs = tf.math.log(tf.clip_by_value(current_action_dist, 1e-8, 1.0))
        loss = -tf.reduce_sum(target_action_dist * log_probs, axis=1)
        
        # Apply importance sampling weights
        weighted_loss = loss * weights
        total_loss = tf.reduce_mean(weighted_loss)
    
    # Compute gradients and update
    gradients = tape.gradient(total_loss, model_primaire.trainable_variables)
    gradients = [tf.clip_by_norm(g, 10.0) for g in gradients]
    optimizer.apply_gradients(zip(gradients, model_primaire.trainable_variables))
    
    # Compute TD errors for priority update
    current_q = tf.reduce_sum(z * current_action_dist, axis=1)
    target_q = tf.reduce_sum(z * target_action_dist, axis=1)
    td_errors = tf.abs(current_q - target_q)
    
    return total_loss, td_errors

def train(debug=False):
    global epsilon, best_score
    
    tab_s.clear()
    total_steps = 0
    loss_history = []
    
    for epoch in trange(n_epochs, desc="Training Epochs"):
        epoch_scores = []
        
        for game in trange(games_per_epoch, desc=f"Epoch {epoch+1:03d}", leave=False):
            if debug:
                print(f"Epoch {epoch+1:03d}/{n_epochs:03d} | Jeu {game+1:03d}/{games_per_epoch:03d}")
            
            # Run simulation
            simulation(epsilon, debug=debug)
            
            # Training step
            if len(replay_buffer.buffer) > 1000:
                if total_steps % update_frequency == 0:
                    # Compute beta for prioritized replay
                    beta = min(1.0, beta_start + total_steps * (1.0 - beta_start) / beta_frames)
                    
                    # Sample from replay buffer
                    samples, indices, weights = replay_buffer.sample(batch_size, beta)
                    
                    if samples is not None:
                        if debug:
                            start_time = time.time()
                        
                        # Train step
                        loss, td_errors = train_step(samples, weights, beta)
                        loss_history.append(loss.numpy())
                        
                        # Update priorities
                        replay_buffer.update_priorities(indices, td_errors.numpy())
                        
                        if debug:
                            print(f"  Entraînement {time.time()-start_time:.3f} seconde(s)")
                            print(f"     loss: {loss:.4f}")
                
                # Soft update target network
                if total_steps % target_update_frequency == 0:
                    if debug:
                        print("Mise à jour soft du réseau cible")
                    for target_param, param in zip(model_cible.variables, model_primaire.variables):
                        target_param.assign(tau * param + (1 - tau) * target_param)
            
            total_steps += 1
            
            # Update epsilon (même si on utilise noisy networks)
            if epsilon > epsilon_min:
                epsilon -= epsilon_decay_rate
                epsilon = max(epsilon, epsilon_min)
        
        # Epoch summary
        if tab_s:
            recent_scores = list(tab_s)[-games_per_epoch:]
            mean_score = np.mean(recent_scores)
            max_score = np.max(recent_scores)
            
            print(f"Epoch {epoch+1:03d} - Score moyen: {mean_score:.1f}, Max: {max_score:.0f}")
            
            # Save best model
            if mean_score > best_score:
                print("Sauvegarde du meilleur modèle")
                model_primaire.save(f"{file_model}.keras")
                best_score = mean_score
            
            # Save stats
            np.save(file_stats, list(tab_s))
    
    # Final plotting
    plot_results(list(tab_s), loss_history)

def plot_results(scores, loss_history):
    from scipy.ndimage import uniform_filter1d
    
    if len(scores) < 20:
        return
        
    # Calculate metrics
    moving_avg = uniform_filter1d(scores, size=min(50, len(scores)//4))
    
    plt.figure(figsize=(15, 10))
    
    # Scores
    plt.subplot(2, 2, 1)
    plt.plot(scores, alpha=0.6, label="Score brut")
    plt.plot(moving_avg, label=f"Moyenne mobile ({min(50, len(scores)//4)})")
    plt.legend()
    plt.title("Évolution des scores")
    plt.xlabel("Jeu")
    plt.ylabel("Score")
    
    # Loss
    plt.subplot(2, 2, 2)
    if loss_history:
        plt.plot(loss_history)
        plt.title("Évolution de la perte")
        plt.xlabel("Étape d'entraînement")
        plt.ylabel("Loss")
    
    # Score distribution
    plt.subplot(2, 2, 3)
    plt.hist(scores, bins=50, alpha=0.7)
    plt.title("Distribution des scores")
    plt.xlabel("Score")
    plt.ylabel("Fréquence")
    
    # Recent performance
    plt.subplot(2, 2, 4)
    if len(scores) > 100:
        recent_scores = scores[-100:]
        plt.plot(recent_scores)
        plt.title("100 derniers scores")
        plt.xlabel("Jeu")
        plt.ylabel("Score")
    
    plt.tight_layout()
    plt.show()

# Initialize models
print("Initialisation des modèles...")
model_primaire = create_model()
model_primaire.summary()

model_cible = tf.keras.models.clone_model(model_primaire)
for target_param, param in zip(model_cible.variables, model_primaire.variables):
    target_param.assign(param)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=6.25e-5)  # Plus petit LR

print("Début de l'entraînement...")
train(debug=True)
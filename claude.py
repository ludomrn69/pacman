import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import cv2

gym.register_envs(ale_py)

class PacmanQLearning:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.alpha = alpha          # Taux d'apprentissage
        self.gamma = gamma          # Facteur de discount
        self.epsilon = epsilon      # Exploration (epsilon-greedy)
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Table Q sous forme de dictionnaire (état -> action -> valeur)
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        
        # Statistiques
        self.scores = []
        self.epsilons = []
        self.steps_per_episode = []
        
    def preprocess_state(self, state):
        """
        Simplifie l'état pour réduire l'espace d'états
        (sinon on aurait 210x160x3 = 100,800 états possibles!)
        """
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        
        # Réduire la taille (downsampling)
        small = cv2.resize(gray, (84, 84))
        
        # Binariser (0 ou 1) pour simplifier encore plus
        binary = (small > 128).astype(np.uint8)
        
        # Convertir en tuple pour pouvoir l'utiliser comme clé de dictionnaire
        return tuple(binary.flatten())
    
    def choose_action(self, state):
        """
        Stratégie epsilon-greedy comme dans tes cours
        """
        if np.random.random() < self.epsilon:
            # Exploration : action aléatoire
            return self.env.action_space.sample()
        else:
            # Exploitation : meilleure action selon Q-table
            state_key = self.preprocess_state(state)
            return np.argmax(self.q_table[state_key])
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Mise à jour Q-learning (équation de Bellman)
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        """
        state_key = self.preprocess_state(state)
        next_state_key = self.preprocess_state(next_state)
        
        # Valeur actuelle
        current_q = self.q_table[state_key][action]
        
        # Valeur maximale pour l'état suivant
        max_next_q = np.max(self.q_table[next_state_key])
        
        # Équation de Bellman
        target = reward + self.gamma * max_next_q
        
        # Mise à jour
        self.q_table[state_key][action] = current_q + self.alpha * (target - current_q)
    
    def train(self, episodes=1000):
        """
        Entraînement avec Q-learning
        """
        print("🧠 Début de l'entraînement Q-learning...")
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                # Choisir une action (epsilon-greedy)
                action = self.choose_action(state)
                
                # Exécuter l'action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                # Mise à jour Q-learning
                self.update_q_value(state, action, reward, next_state)
                
                # Mise à jour des variables
                state = next_state
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            # Décroissance epsilon (moins d'exploration au fil du temps)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Statistiques
            self.scores.append(total_reward)
            self.epsilons.append(self.epsilon)
            self.steps_per_episode.append(steps)
            
            # Affichage périodique
            if episode % 100 == 0:
                avg_score = np.mean(self.scores[-100:])
                print(f"Episode {episode}: Score moyen = {avg_score:.2f}, Epsilon = {self.epsilon:.3f}")
        
        print("✅ Entraînement terminé!")
    
    def test(self, episodes=5, render=True):
        """
        Test de l'agent entraîné
        """
        print("🎮 Test de l'agent entraîné...")
        
        # Créer un environnement avec rendu visuel
        if render:
            test_env = gym.make("ALE/MsPacman-v5", render_mode="human")
        else:
            test_env = self.env
        
        test_scores = []
        old_epsilon = self.epsilon
        self.epsilon = 0  # Pas d'exploration pendant le test
        
        for episode in range(episodes):
            state, _ = test_env.reset()
            total_reward = 0
            
            while True:
                action = self.choose_action(state)
                state, reward, terminated, truncated, _ = test_env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            test_scores.append(total_reward)
            print(f"Episode test {episode + 1}: Score = {total_reward}")
        
        self.epsilon = old_epsilon  # Restaurer epsilon
        
        if render:
            test_env.close()
        
        print(f"📊 Score moyen sur {episodes} tests: {np.mean(test_scores):.2f}")
        return test_scores
    
    def plot_results(self):
        """
        Graphiques des résultats (comme dans tes cours)
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Scores par épisode
        axes[0, 0].plot(self.scores)
        axes[0, 0].set_title('Scores par épisode')
        axes[0, 0].set_xlabel('Épisode')
        axes[0, 0].set_ylabel('Score')
        
        # Moyenne mobile des scores
        window = 100
        if len(self.scores) > window:
            moving_avg = np.convolve(self.scores, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title(f'Moyenne mobile (fenêtre {window})')
            axes[0, 1].set_xlabel('Épisode')
            axes[0, 1].set_ylabel('Score moyen')
        
        # Évolution d'epsilon
        axes[1, 0].plot(self.epsilons)
        axes[1, 0].set_title('Évolution d\'epsilon (exploration)')
        axes[1, 0].set_xlabel('Épisode')
        axes[1, 0].set_ylabel('Epsilon')
        
        # Durée des épisodes
        axes[1, 1].plot(self.steps_per_episode)
        axes[1, 1].set_title('Durée des épisodes')
        axes[1, 1].set_xlabel('Épisode')
        axes[1, 1].set_ylabel('Nombre d\'étapes')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filename='qlearning_pacman.pkl'):
        """
        Sauvegarde de la Q-table
        """
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"💾 Modèle sauvegardé: {filename}")
    
    def load_model(self, filename='qlearning_pacman.pkl'):
        """
        Chargement de la Q-table
        """
        try:
            with open(filename, 'rb') as f:
                loaded_q_table = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n))
                self.q_table.update(loaded_q_table)
            print(f"📂 Modèle chargé: {filename}")
        except FileNotFoundError:
            print(f"❌ Fichier {filename} non trouvé")

# Utilisation
if __name__ == "__main__":
    # Créer l'environnement
    env = gym.make("ALE/MsPacman-v5", render_mode=None)
    
    # Créer l'agent Q-learning
    agent = PacmanQLearning(env, alpha=0.1, gamma=0.99, epsilon=1.0)
    
    # Entraînement
    agent.train(episodes=500)  # Commence avec peu d'épisodes
    
    # Graphiques des résultats
    agent.plot_results()
    
    # Sauvegarder le modèle
    agent.save_model()
    
    # Test avec affichage
    agent.test(episodes=3, render=True)
    
    env.close()
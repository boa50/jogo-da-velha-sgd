# Link: https://github.com/dennybritz/reinforcement-learning

import gym
import itertools
import sys
import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

matplotlib.style.use('ggplot')


def rbf_projection_idea_main():
  # Exemplo clássico de utilizar kernel RBF para aumentar a dimensionalidade dos dados (similar à SVM)
  # Retirado da página do sklearn

  from sklearn.linear_model import SGDClassifier
  X = [[0, 0], [1, 1], [1, 0], [0, 1]]
  y = [0, 0, 1, 1]
  rbf_feature = RBFSampler(gamma=1, random_state=1)
  X_features = rbf_feature.fit_transform(X)
  clf = SGDClassifier(max_iter=5, tol=1e-3)
  clf.fit(X_features, y)
  SGDClassifier(max_iter=5)
  print('Score:', clf.score(X_features, y))


# Código para os plots do código principal
EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])


def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):
    """
    Responsável por plotar a superficie das funções de valor
    """
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Posição')
    ax.set_ylabel('Velocidade')
    ax.set_zlabel('Valor')
    ax.set_title("Função de custo do cenário")
    fig.colorbar(surf)
    plt.show()


def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plota o comprimento do episódio no tempo
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episódio")
    plt.ylabel("Comprimento do episódio")
    plt.title("Comprimento do episódio no tempo")
    if noshow:
        plt.close(fig1)
    else:
        plt.show()

    # Plota a recompensa do episódio no tempo
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa do Episódio (Suavizada)")
    plt.title("Recompensa do Episódio no Tempo (Janela de suavização {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show()

    # Plota o número de passos no episódio por unidade de tempo
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Passos no tempo")
    plt.ylabel("Episódio")
    plt.title("Episódios por passo temporal")
    if noshow:
        plt.close(fig3)
    else:
        plt.show()

    return fig1, fig2, fig3


# Código para o environment
class Estimator:
  """
  O aproximador da função de valor
  """

  def __init__(self, env):
    # Pré-processamento: padronização (média zero e variância unitária)
    # Utiliza alguns exemplos de observações para isso
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    self.scaler = sklearn.preprocessing.StandardScaler()
    self.scaler.fit(observation_examples)

    # O estado é convertido para uma representação de mais alta dimensionalidade via kernels RBF (a estilo da SVM)
    # Uma combinação de diferentes kernels é utilizada, com diferentes variâncias
    self.featurizer = sklearn.pipeline.FeatureUnion([
      ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
      ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
      ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
      ("rbf4", RBFSampler(gamma=0.5, n_components=100))
    ])
    self.featurizer.fit(self.scaler.transform(observation_examples))

    # Um modelo separado é criado para cada ação no espaço de ações
    self.models = []
    for _ in range(env.action_space.n):
      model = SGDRegressor(learning_rate="constant")
      # O método partial_fit é utilizado para inicializar o modelo
      model.partial_fit([self.featurize_state(env.reset())], [0])
      self.models.append(model)

  def featurize_state(self, state):
    """
    Retorna a representação de um estado no espaço de características
    """
    scaled = self.scaler.transform([state])
    featurized = self.featurizer.transform(scaled)
    return featurized[0]

  def predict(self, s, a=None):
    """
    Faz predições da função de valor.

    Se uma ação a foi passada, retorna um único número de predição
    Se nenhuma ação foi passada, retorna um vetor para todas as ações naquele estado

    """
    features = self.featurize_state(s)
    if not a:
      return np.array([m.predict([features])[0] for m in self.models])
    else:
      return self.models[a].predict([features])[0]

  def update(self, s, a, y):
    """
    Realiza o update dos parâmetros do estimador para um dado (estado, ação) com relação a y
    """
    features = self.featurize_state(s)
    self.models[a].partial_fit([features], [y])


def make_epsilon_greedy_policy(estimator, epsilon, nA):
  """
  Cria uma política epsilon-greedy baseado em uma função de valor Q aproximada e um epsilon. Retorna as
  probabilidades de cada ação
  """

  def policy_fn(observation):
    A = np.ones(nA, dtype=float) * epsilon / nA
    q_values = estimator.predict(observation)
    best_action = np.argmax(q_values)
    A[best_action] += (1.0 - epsilon)
    return A

  return policy_fn


def q_learning(env, estimator, num_episodes, gamma=1.0, epsilon=0.1, epsilon_decay=1.0):
  """
  Algoritmo Q-learning via aproximador com uma política epsilon-greedy
  """

  # Mantém as estatísticas
  stats = EpisodeStats(
    episode_lengths=np.zeros(num_episodes),
    episode_rewards=np.zeros(num_episodes))

  for i_episode in range(num_episodes):

    # A política greedy com base no estado atual do estimador
    policy = make_epsilon_greedy_policy(
      estimator, epsilon * epsilon_decay ** i_episode, env.action_space.n)

    # Printa o número do episódio e a recompensa para o último episódio
    last_reward = stats.episode_rewards[i_episode - 1]
    sys.stdout.flush()

    # Inicializa o environment
    state = env.reset()

    # Os passos no ambiente
    for t in itertools.count():
      action_probs = policy(state)
      action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

      # Realiza uma ação
      next_state, reward, done, _ = env.step(action)
      #env.render()

      # Atualiza as estatísticas
      stats.episode_rewards[i_episode] += reward
      stats.episode_lengths[i_episode] = t

      # Update do TD
      q_values_next = estimator.predict(next_state)

      # Q-value para o TD Target
      td_target = reward + gamma * np.max(q_values_next)

      # Atualiza o aproximador usando o td_target
      estimator.update(state, action, td_target)

      print("\rPasso {} @ Episódio {}/{} ({})".format(t, i_episode + 1, num_episodes, last_reward), end="")

      if done:
        break

      state = next_state

  return stats


def run_episode(env, estimator):
  done = False
  state = env.reset()
  env.render()
  while not done:
    q_values = estimator.predict(state)
    action = np.argmax(q_values)
    next_state, reward, done, _ = env.step(action)
    env.render()
    state = next_state
  env.close()


def main_q_learning_run():
  # Observação: [posição do carro, velocidade do carro]
  # Recompensa: -1 para qualquer transição e 0 quando alcança o resultado
  # Ação: [esquerda, neutro, direita]
  env = gym.envs.make("MountainCar-v0")
  estimator = Estimator(env)

  stats = q_learning(env, estimator, 100, epsilon=0.1)

  plot_cost_to_go_mountain_car(env, estimator)
  plot_episode_stats(stats, smoothing_window=25)

  run_episode(env, estimator)


if __name__ == "__main__":
  # Ideia dos kernels RBF (similar à SVM)
  # rbf_projection_idea_main()

  # Código principal do algoritmo Q-learning via método SGD com representação por kernels RBF
  main_q_learning_run()

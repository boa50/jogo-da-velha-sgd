# Link: https://michaeloneill.github.io/RL-tutorial.html

import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
import time
import timeit
from collections import namedtuple
from tile_coding import IHT, tiles
from matplotlib import pyplot as plt
from matplotlib import cm
matplotlib.style.use('ggplot')


RunStats = namedtuple('RunStats', ['steps', 'returns'])


def basic_test_main():
  # Ações básicas para exemplificar o cenário
  env = gym.envs.make("MountainCar-v0")
  env.reset()
  env.render(mode='rgb_array')
  for i in range(500):
    print('Movimento #%d' % (i))
    state, reward, done, _ = env.step(0)
    env.render(mode='rgb_array')
  env.close()


def trying_the_objective_main():
  # Exemplo explorando outras ações. Não é fácil fazer isso
  env = gym.envs.make("MountainCar-v0")
  env.reset()
  env.render(mode='rgb_array')

  for i in range(50):
    print('Movimento #%d' % (i))
    state, reward, done, _ = env.step(0)
    env.render(mode='rgb_array')

    if done:
      break

  for i in range(10):
    print('Movimento #%d' % (i))
    state, reward, done, _ = env.step(1)
    env.render(mode='rgb_array')

    if done:
      break

  if not done:
    for i in range(150):
      print('Movimento #%d' % (i))
      state, reward, done, _ = env.step(2)
      env.render(mode='rgb_array')

      if done:
        break

  env.close()


class QEstimator:
    """
    Q-value parametrizado por um aproximador linear para método de semi-gradiente com
    features geradas por tile-coding.
    """
    def __init__(self, env, step_size, num_tilings=8, max_size=4096):
        self.max_size = max_size
        self.num_tilings = num_tilings
        self.tiling_dim = num_tilings
        self.env = env

        # O tamanho do passo é a fração a considerar em direção ao target. Para computar
        # a learning rate alpha, é escalado pelo número de tiles
        self.alpha = step_size / num_tilings

        # A inicialização atribui um único index para cada tile, limitado a max_size tiles.
        # É necessário que max_size >= número total de tiles (numero de tiles x dimensão tiles ** 2)
        self.iht = IHT(max_size)

        # Inicializa os pesos (correspondentes ao número de tiles)
        self.weights = np.zeros(max_size)

        # Cada tile terá um número de buckets, determinado por tiling_dim. É necessário
        # calcular a escala das features com relação às tiles.
        self.position_scale = self.tiling_dim / \
                              (self.env.observation_space.high[0] - self.env.observation_space.low[0])
        self.velocity_scale = self.tiling_dim / \
                              (self.env.observation_space.high[1] - self.env.observation_space.low[1])

    def featurize_state_action(self, state, action):
        """
        Retorna a representação em features de um par stado-ação
        """
        featurized = tiles(self.iht, self.num_tilings,
                           [self.position_scale * state[0],
                            self.velocity_scale * state[1]],
                           [action])
        return featurized

    def predict(self, s):
        """
        Prediz a q-value(s) usando uma função linear.
        """
        features = [self.featurize_state_action(s, i) for
                    i in range(self.env.action_space.n)]
        return [np.sum(self.weights[f]) for f in features]

    def update(self, s, a, target):
        """
        Faz o update dos parâmetros do estimador, que é representado por um vetor w de pesos para cada
        feature do tile-coding
        """
        features = self.featurize_state_action(s, a)
        estimation = np.sum(self.weights[features])  # Linear FA
        delta = (target - estimation)
        self.weights[features] += self.alpha * delta

    def reset(self):
        """
        Reseta o vetor de pesos
        """
        self.weights = np.zeros(self.max_size)


def make_epsilon_greedy_policy(estimator, epsilon, num_actions):
    """
    Cria uma política epsilon-greedy baseado na q-value aproximada pelo estimator.
    """
    def policy_fn(observation):
        action_probs = np.ones(num_actions, dtype=float) * epsilon / num_actions
        q_values = estimator.predict(observation)
        best_action_idx = np.argmax(q_values)
        action_probs[best_action_idx] += (1.0 - epsilon)
        return action_probs
    return policy_fn


def sarsa_n(n, env, estimator, gamma=1.0, epsilon=0):
    """
    Algoritmo Sarsa de n passos via método de semi-gradientes
    para encontrar q e pi via aproximação de funções lineares
    """

    # Cria política epsilon greedy
    policy = make_epsilon_greedy_policy(estimator, epsilon, env.action_space.n)

    # Inicia o environment e escolhe uma ação
    state = env.reset()
    action_probs = policy(state)
    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

    # Armazena informações
    states = [state]
    actions = [action]
    rewards = [0.0]

    # Passa pelo episódio
    for t in itertools.count():
        env.render()
        next_state, reward, done, _ = env.step(action)
        states.append(next_state)
        rewards.append(reward)

        if done:
            break
        else:
            # Toma o próximo passo
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            actions.append(next_action)

        # Especifica o estado a ser realizado update
        update_time = t + 1 - n
        if update_time >= 0:
            target = 0
            for i in range(update_time + 1, update_time + n + 1):
                target += np.power(gamma, i - update_time - 1) * rewards[i]
            q_values_next = estimator.predict(states[update_time + n])
            target += q_values_next[actions[update_time + n]]
            estimator.update(states[update_time], actions[update_time], target)

        action = next_action

    ret = np.sum(rewards)
    return t, ret


def plot_cost_to_go(env, estimator, num_partitions=50):
    """
    Plota Q(s, a_max) para cada estado s=(posição, velocidade), onde a_max
    é a ação que maximiza o valor a partir de s, de acordo com o estimador. O espaço
    de estados é contínuo. Portanto, há uma discretização por num_partitions em cada dimensão.
    """
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_partitions)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_partitions)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda obs: -np.max(estimator.predict(obs)), 2, np.stack([X, Y], axis=2))

    fig, ax = plt.subplots(figsize=(10, 5))
    p = ax.pcolor(X, Y, Z, cmap=cm.RdBu, vmin=0, vmax=200)

    ax.set_xlabel('Posição')
    ax.set_ylabel('Velocidade')
    ax.set_title("\"Cost To Go\" Function (Função de estado-ação - max)")
    fig.colorbar(p)
    plt.show()


def generate_greedy_policy_animation(env, estimator):
    """
    Segue deterministicamente a greedy policy com respeito à função de valor (q-value)
    """

    # Coloca o epsilon como 0 para seguir uma política completamente greedy
    policy = make_epsilon_greedy_policy(estimator=estimator, epsilon=0, num_actions=env.action_space.n)
    state = env.reset()
    for t in itertools.count():
        env.render()
        time.sleep(0.01)
        action_probs = policy(state)  # Calcula as probabilidades de ação no estado atual
        [action] = np.nonzero(action_probs)[0]  # Seleciona a greedy action
        state, _, done, _ = env.step(action)
        if done:
            print('Resolvido em {} passos'.format(t))
            break
    env.close()


def plot_learning_curves(stats, smoothing_window=10):
    """
    Plota o número de passos dado pelo agente para resolver a tarefa como uma função
    do número do episódio, com uma janela de suavização (média móvel)
    """

    plt.figure(figsize=(10, 5))
    steps_per_episode = pd.Series(stats.steps).rolling(smoothing_window).mean()
    plt.plot(np.arange(len(steps_per_episode)), steps_per_episode)
    plt.xlabel("Episódio")
    plt.ylabel("Passos")
    plt.title("Passos por Episódio")
    plt.legend()
    plt.show()


def run(env, estimator, num_episodes=500, n=1, gamma=1.0, epsilon=0):
    """
    Roda o algoritmo por múltiplos episódios e faz logs de cada retorno completo (G_t) e o
    número de passos tomados
    """
    stats = RunStats(
        steps=np.zeros(num_episodes),
        returns=np.zeros(num_episodes))

    for i in range(num_episodes):
        episode_steps, episode_return = sarsa_n(n, env, estimator, gamma, epsilon)
        stats.steps[i] = episode_steps
        stats.returns[i] = episode_return
        sys.stdout.flush()
        print("\rEpisódio {}/{} Retorno {}".format(
            i + 1, num_episodes, episode_return), end="")
    return stats


def main():
    step_size = 0.5
    n = 1
    num_episodes = 500

    env = gym.make("MountainCar-v0")
    env._max_episode_steps = 3000  # Aumenta o limite de tempo do environment
    np.random.seed(6)  # Seta o seed para reproduzir o experimento (posição inicial é aleatória)

    estimator = QEstimator(env=env, step_size=step_size)

    start_time = timeit.default_timer()
    run_stats_n = run(env, estimator, num_episodes=500, n=1, gamma=1.0, epsilon=0)
    elapsed_time = timeit.default_timer() - start_time

    plot_cost_to_go(env, estimator)
    plot_learning_curves(run_stats_n)
    print('{} episódios completos em {:.2f}s'.format(num_episodes, elapsed_time))
    generate_greedy_policy_animation(env, estimator)


if __name__ == '__main__':
    # Teste básico
    #basic_test_main()

    # Tentando fazer algo mais interessante
    #trying_the_objective_main()

    # código principal
    main()
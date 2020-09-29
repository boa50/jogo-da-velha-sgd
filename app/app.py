#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)           #
# 2016 Jan Hakenberg(jan.hakenberg@gmail.com)                         #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS


class State:
    def __init__(self):
        # o tabuleiro é representado por um array n * n
        # 1 representa o símbolo do jogador que joga primeiro
        # -1 é o símbolo do outro jogador
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.winner = None
        self.end = None

    # Checa se alguém venceu o jogo ou se é um empate
    def is_end(self):
        if self.end is not None:
            return self.end
        results = []
        # Checa as linhas
        for i in range(BOARD_ROWS):
            results.append(np.sum(self.data[i, :]))
        # Checa as colunas
        for i in range(BOARD_COLS):
            results.append(np.sum(self.data[:, i]))

        # Checa as diagonais
        trace = 0
        reverse_trace = 0
        for i in range(BOARD_ROWS):
            trace += self.data[i, i]
            reverse_trace += self.data[i, BOARD_ROWS - 1 - i]
        results.append(trace)
        results.append(reverse_trace)

        for result in results:
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end
            if result == -3:
                self.winner = -1
                self.end = True
                return self.end

        # Verifica se é um empate
        sum_values = np.sum(np.abs(self.data))
        if sum_values == BOARD_SIZE:
            self.winner = 0
            self.end = True
            return self.end

        # O jogo ainda continua
        self.end = False
        return self.end

    # Coloca o símbolo symbol na posição (i, j)
    def next_state(self, i, j, symbol):
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol
        return new_state

    # Printa o tabuleiro
    def print_state(self):
        for i in range(BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(BOARD_COLS):
                if self.data[i, j] == 1:
                    token = '*'
                elif self.data[i, j] == -1:
                    token = 'x'
                else:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-------------')

def get_random_state():
    state = State()
    symbol = np.random.choice([-1, 1])

    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            escolha = np.random.choice([0, symbol])
            state = state.next_state(i, j, escolha)
            
            if escolha == symbol:
                symbol = -symbol

    return state


class Estimator:
  def __init__(self, symbol):
    observations = [get_random_state().data.reshape(-1) for _ in range(10000)]
    self.symbol = symbol
    # self.scaler = sklearn.preprocessing.StandardScaler()
    # self.scaler.fit(observations)

    self.featurizer = sklearn.pipeline.FeatureUnion([
      ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
      ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
      ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
      ("rbf4", RBFSampler(gamma=0.5, n_components=100))
    ], n_jobs=-1)
    self.featurizer.fit(observations)
    # self.featurizer.fit(self.scaler.transform(observations))

    self.models = dict()
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([self.featurize_state(State())], [0])
            self.models[(i,j)] = model

  def featurize_state(self, state):
    # scaled = self.scaler.transform([state.data.reshape(-1)])
    # featurized = self.featurizer.transform(scaled)
    featurized = self.featurizer.transform([state.data.reshape(-1)])
    return featurized[0]
    # return state.data.reshape(-1)

  def predict(self, s, a=None):
    features = self.featurize_state(s)
    if not a:
        return np.array([m.predict([features])[0] for m in self.models.values()])
    else:
        return self.models[a].predict([features])[0]

  def update(self, s, a, y):
    features = self.featurize_state(s)
    self.models[a].partial_fit([features], [y])


def make_epsilon_greedy_policy(estimator, epsilon):
  def policy_fn(observation):
    values = observation.data.reshape(-1)
    disponiveis = len(values) - np.count_nonzero(values)
    q_values = estimator.predict(observation)

    A = []
    for i in range(len(values)):
      if values[i] == 0:
        A.append(epsilon / disponiveis)
      else:
        A.append(0)
        q_values[i] = -np.inf

    best_value_action = np.max(q_values)
    best_idxs = np.where(q_values == best_value_action)
    best_action = np.random.choice(best_idxs[0])
    A[best_action] += (1.0 - epsilon)
    return A

  return policy_fn

class Player:
    def __init__(self, step_size=0.1, epsilon=0.1, symbol=0):
        self.estimations = dict()
        self.step_size = step_size
        self.epsilon = epsilon
        self.previous_state = State()
        self.state = None
        self.symbol = symbol

        self.estimator = Estimator(self.symbol)
        self.policy = make_epsilon_greedy_policy(self.estimator, self.epsilon)
        self.action = (0,0)

        self.actions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.actions.append((i,j))

    # Adiciona informação do novo estado
    def set_state(self, state):
        if self.state != None:
            self.previous_state.data = np.copy(self.state.data)
        self.state = state

    def set_symbol(self, symbol):        
        self.symbol = symbol

    def update_epsilon(self, epsilon):
        epsilon = max(epsilon, 0.01)
        self.policy = make_epsilon_greedy_policy(self.estimator, epsilon)

    # Faz o update da estimação
    def backup(self, next_state, other=False):
        is_end = next_state.is_end()
        reward = 0
        if is_end:
            if next_state.winner == self.symbol:
                reward = 0.5
            elif next_state.winner == -self.symbol:
                reward = -1
            else:
                reward = 0

        if other:
            next_state.data = np.copy(self.state.data)
            self.state = self.previous_state

        # Update do TD
        q_values_next = self.estimator.predict(next_state)

        # Q-value para o TD Target
        if is_end:
            td_target = reward
        else:
            gamma = 1
            td_target = reward + gamma * np.max(q_values_next)

        # Atualiza o aproximador usando o td_target
        self.estimator.update(self.state, self.action, td_target)

    # Escolhe uma ação baseada no estado
    def act(self):
        action_probs = self.policy(self.state)
        action_idx = np.random.choice(np.arange(len(self.actions)), p=action_probs)
        self.action = self.actions[action_idx]

        next_state = self.state.next_state(self.action[0], self.action[1], self.symbol)
        is_end = next_state.is_end()

        self.backup(next_state)

        return next_state, is_end

    def save_policy(self, epoch):
        with open('app/policy_%s_%d.bin' % (('first' if self.symbol == 1 else 'second'), epoch), 'wb') as f:
            pickle.dump(self.estimator, f)

    def load_policy(self, epoch):
        with open('app/policy_%s_%d.bin' % (('first' if self.symbol == 1 else 'second'), epoch), 'rb') as f:
            self.estimator = pickle.load(f)
            self.policy = make_epsilon_greedy_policy(self.estimator, self.epsilon)


class Judger:
    def __init__(self, player1, player2):
        self.p1 = player1
        self.p2 = player2
        self.current_player = None

    def alternate(self):
        while True:
            yield self.p1
            yield self.p2

    def play(self, train=False, print_state=False):
        alternator = self.alternate()
        current_state = State()
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)
        if print_state:
            current_state.print_state()
        while True:
            player = next(alternator)
            current_state, is_end = player.act()
            
            if print_state:
                current_state.print_state()
            if is_end:
                if train:
                    player = next(alternator)
                    player.backup(current_state, True)
                return current_state.winner

            self.p1.set_state(current_state)
            self.p2.set_state(current_state)


# human interface
# input a number to put a chessman
# | q | w | e |
# | a | s | d |
# | z | x | c |
class HumanPlayer:
    def __init__(self, **kwargs):
        self.symbol = 1
        self.keys = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
        self.state = None

    def reset(self):
        pass

    def set_state(self, state):
        self.state = state

    def set_symbol(self, symbol):
        self.symbol = symbol

    def act(self):
        self.state.print_state()
        key = input("Input your position:")
        data = self.keys.index(key)
        i = data // BOARD_COLS
        j = data % BOARD_COLS

        next_state = self.state.next_state(i, j, self.symbol)
        is_end = next_state.is_end()

        return next_state, is_end

def train(epochs, print_every_n=500):
    epsilon = 1
    epsilon_decay = 0.999
    epsilon_min = 0.01

    player1 = Player(epsilon=epsilon, symbol=1)
    player2 = Player(epsilon=epsilon, symbol=-1)
    judger = Judger(player1, player2)
    player1_win = 0.0
    player2_win = 0.0
    for i in range(1, epochs + 1):
        winner = judger.play(train=True, print_state=False)
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        if i % print_every_n == 0:
            print('Epoch %d, player 1 winrate: %.02f, player 2 winrate: %.02f' % (i, player1_win / i, player2_win / i))

            player1.save_policy(i)
            player2.save_policy(i)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        player1.update_epsilon(epsilon)
        player2.update_epsilon(epsilon)


def compete(turns, policy_number):
    player1 = Player(epsilon=0, symbol=1)
    player2 = Player(epsilon=0, symbol=-1)
    judger = Judger(player1, player2)
    player1.load_policy(policy_number)
    player2.load_policy(policy_number)
    player1_win = 0.0
    player2_win = 0.0
    for _ in range(turns):
        winner = judger.play()
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
    print('%d turns, player 1 win %.02f, player 2 win %.02f' % (turns, player1_win / turns, player2_win / turns))


def play(policy_number):
    while True:
        player1 = HumanPlayer()
        player2 = Player(epsilon=0, symbol=-1)
        judger = Judger(player1, player2)
        player2.load_policy(policy_number)
        winner = judger.play()
        if winner == player2.symbol:
            print("You lose!")
        elif winner == player1.symbol:
            print("You win!")
        else:
            print("It is a tie!")

if __name__ == '__main__':
    train(int(1e5), print_every_n=1000)
    # compete(int(1e3))
    # play(30000)

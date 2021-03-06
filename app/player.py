import numpy as np
import pickle
import os
import csv

from env import State
from estimator import Estimator
from utils import make_epsilon_greedy_policy


BOARD_ROWS = 3
BOARD_COLS = 3

class Player:
    def __init__(self, step_size=0.1, epsilon=0.1, symbol=0):
        self.step_size = step_size
        self.epsilon = epsilon
        self.previous_state = State()
        self.state = None
        self.symbol = symbol
        self.td_errors = []

        self.estimator = Estimator()
        self.policy = make_epsilon_greedy_policy(self.estimator)
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

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    # Faz o update da estimação
    def backup(self, next_state, other=False):
        is_end = next_state.is_end()
        reward = 0
        if is_end:
            if next_state.winner == self.symbol:
                reward = 1
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

        # Cálculo do TD error
        td = self.estimator.predict(self.state, self.action)
        td_error = np.abs(td_target - td)
        self.td_errors.append(td_error)

        # Atualiza o aproximador usando o td_target
        self.estimator.update(self.state, self.action, td_target)

    # Escolhe uma ação baseada no estado
    def act(self):
        action_probs = self.policy(self.state, self.epsilon)
        action_idx = np.random.choice(np.arange(len(self.actions)), p=action_probs)
        self.action = self.actions[action_idx]

        next_state = self.state.next_state(self.action[0], self.action[1], self.symbol)
        is_end = next_state.is_end()

        self.backup(next_state)

        return next_state, is_end

    def save_policy(self, epoch):
        with open('app/saves/policy_%s_%d.bin' % (('first' if self.symbol == 1 else 'second'), epoch), 'wb') as f:
            pickle.dump(self.estimator, f)

        path = 'app/saves/metrics_%s.csv' % ('first' if self.symbol == 1 else 'second')
        metrics_file = open(path, "a")
        with metrics_file:
            writer = csv.writer(metrics_file)
            for td_error in self.td_errors : writer.writerow ([td_error])

        self.td_errors.clear()

    def load_policy(self, epoch):
        with open('app/saves/policy_%s_%d.bin' % (('first' if self.symbol == 1 else 'second'), epoch), 'rb') as f:
            self.estimator = pickle.load(f)
            self.policy = make_epsilon_greedy_policy(self.estimator)

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
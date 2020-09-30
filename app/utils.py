import numpy as np

from env import State

BOARD_ROWS = 3
BOARD_COLS = 3

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
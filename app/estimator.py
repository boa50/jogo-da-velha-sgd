import numpy as np

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

from utils import get_random_state
from env import State

BOARD_ROWS = 3
BOARD_COLS = 3

class Estimator:
  def __init__(self):
    observations = [get_random_state().data.reshape(-1) for _ in range(10000)]
    self.scaler = sklearn.preprocessing.StandardScaler()
    self.scaler.fit(observations)

    self.featurizer = sklearn.pipeline.FeatureUnion([
      ("rbf1", RBFSampler(gamma=5.0, n_components=150)),
      ("rbf2", RBFSampler(gamma=2.0, n_components=150)),
      ("rbf3", RBFSampler(gamma=1.0, n_components=150)),
      ("rbf4", RBFSampler(gamma=0.5, n_components=150))
    ], n_jobs=1)
    # self.featurizer.fit(observations)
    self.featurizer.fit(self.scaler.transform(observations))

    self.models = dict()
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([self.featurize_state(State())], [0])
            self.models[(i,j)] = model

  def featurize_state(self, state):
    scaled = self.scaler.transform([state.data.reshape(-1)])
    featurized = self.featurizer.transform(scaled)
    # featurized = self.featurizer.transform([state.data.reshape(-1)])
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
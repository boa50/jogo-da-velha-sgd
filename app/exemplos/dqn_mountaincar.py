# Link: https://github.com/pylSER/Deep-Reinforcement-learning-Mountain-Car/blob/master/MountainCarV2.py

import gym
from keras import models
from keras import layers
from keras.optimizers import Adam
from collections import deque
import random
import numpy as np


class MountainCarTrain:
  def __init__(self, env):
    self.env = env
    self.gamma = 0.99
    self.epsilon = 1
    self.epsilon_decay = 0.05
    self.epsilon_min = 0.01
    self.learningRate = 0.001
    self.replayBuffer = deque(maxlen=20000)
    self.trainNetwork = self.createNetwork()
    self.episodeNum = 5
    self.iterationNum = 201  # máximo é 200
    self.numPickFromBuffer = 32
    self.targetNetwork = self.createNetwork()
    self.targetNetwork.set_weights(self.trainNetwork.get_weights())

  def createNetwork(self):
    model = models.Sequential()
    state_shape = self.env.observation_space.shape
    model.add(layers.Dense(24, activation='relu', input_shape=state_shape))
    model.add(layers.Dense(48, activation='relu'))
    model.add(layers.Dense(self.env.action_space.n, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=self.learningRate))
    return model

  def getBestAction(self, state):
    self.epsilon = max(self.epsilon_min, self.epsilon)
    if np.random.rand(1) < self.epsilon:
      action = np.random.randint(0, 3)
    else:
      action = np.argmax(self.trainNetwork.predict(state)[0])
    return action

  def trainFromBuffer_Boost(self):
    if len(self.replayBuffer) < self.numPickFromBuffer:
      return
    samples = random.sample(self.replayBuffer, self.numPickFromBuffer)
    npsamples = np.array(samples)
    states_temp, actions_temp, rewards_temp, newstates_temp, dones_temp = np.hsplit(npsamples, 5)
    states = np.concatenate((np.squeeze(states_temp[:])), axis=0)
    rewards = rewards_temp.reshape(self.numPickFromBuffer, ).astype(float)
    targets = self.trainNetwork.predict(states)
    newstates = np.concatenate(np.concatenate(newstates_temp))
    dones = np.concatenate(dones_temp).astype(bool)
    notdones = ~dones
    notdones = notdones.astype(float)
    dones = dones.astype(float)
    Q_futures = self.targetNetwork.predict(newstates).max(axis=1)
    targets[(np.arange(self.numPickFromBuffer),
             actions_temp.reshape(self.numPickFromBuffer, ).astype(int))] = rewards * dones + (
              rewards + Q_futures * self.gamma) * notdones
    self.trainNetwork.fit(states, targets, epochs=1, verbose=0)

  def trainFromBuffer(self):
    if len(self.replayBuffer) < self.numPickFromBuffer:
      return

    samples = random.sample(self.replayBuffer, self.numPickFromBuffer)

    states = []
    newStates = []
    for sample in samples:
      state, action, reward, new_state, done = sample
      states.append(state)
      newStates.append(new_state)

    newArray = np.array(states)
    states = newArray.reshape(self.numPickFromBuffer, 2)

    newArray2 = np.array(newStates)
    newStates = newArray2.reshape(self.numPickFromBuffer, 2)

    targets = self.trainNetwork.predict(states)
    new_state_targets = self.targetNetwork.predict(newStates)

    i = 0
    for sample in samples:
      state, action, reward, new_state, done = sample
      target = targets[i]
      if done:
        target[action] = reward
      else:
        Q_future = max(new_state_targets[i])
        target[action] = reward + Q_future * self.gamma
      i += 1

    self.trainNetwork.fit(states, targets, epochs=1, verbose=0)

  def orginalTry(self, currentState, eps):
    rewardSum = 0
    max_position = -99

    for i in range(self.iterationNum):
      bestAction = self.getBestAction(currentState)

      # if eps % 50 == 0:
      #   env.render()

      new_state, reward, done, _ = env.step(bestAction)

      new_state = new_state.reshape(1, 2)

      # # Verifica a posição máxima
      if new_state[0][0] > max_position:
        max_position = new_state[0][0]

      # Ajusta a recompnesa caso complete a tarefa
      if new_state[0][0] >= 0.5:
        reward += 10

      self.replayBuffer.append([currentState, bestAction, reward, new_state, done])

      self.trainFromBuffer_Boost()
      #self.trainFromBuffer()

      rewardSum += reward

      currentState = new_state

      if done:
        break

    if i >= 199:
      print("Falhou em finalizar a tarefa no episódio {}".format(eps))
    else:
      print("Sucesso no episódio {}, usou {} iterações!".format(eps, i))
      self.trainNetwork.save('./trainNetworkInEPS{}.h5'.format(eps))

    # Sync
    self.targetNetwork.set_weights(self.trainNetwork.get_weights())

    print("Epsilon = {}, a recompensa é {} a posição máxima é {}".format(max(self.epsilon_min, self.epsilon), rewardSum,
                                                                             max_position))
    self.epsilon -= self.epsilon_decay

  def start(self):
    for eps in range(self.episodeNum):
      currentState = env.reset().reshape(1, 2)
      self.orginalTry(currentState, eps)


def run_episode(env, dqn):
  state = env.reset()
  env.render()
  done = False
  dqn.epsilon = 0
  while not done:
    action = dqn.getBestAction(state.reshape(1, -1))
    state, reward, done, _ = env.step(action)
    env.render()
    if done:
      break
  env.close()


if __name__ == '__main__':
  env = gym.make('MountainCar-v0')
  dqn = MountainCarTrain(env=env)
  dqn.start()
  run_episode(env, dqn)

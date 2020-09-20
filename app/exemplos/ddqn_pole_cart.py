# Link: https://github.com/Parsa33033/Deep-Reinforcement-Learning-DQN/blob/master/Dueling-DQN.py

import numpy as np
import gym
import keras
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D
import random
from collections import deque
import time
from pathlib import Path
import itertools


class DQN():
    def __init__(self, gym_game, epsilon=1, epsilon_decay=0.995, epsilon_min=0.01, batch_size=32,
                 discount_factor=0.9, num_of_episodes=500, load_path=None):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.num_of_episodes = num_of_episodes
        self.game = gym_game
        self.environment = gym.make(gym_game)

        try:
            shape = self.environment.observation_space.shape
            self.state_size = (shape[0], shape[1], shape[2])
            self.s = shape[0] * shape[1] * shape[2]
            self.state_mode = "observation"
        except:
            self.state_size = self.environment.observation_space.shape[0]
            self.s = self.state_size
            self.state_mode = "information"

        try:
            self.action_size = self.environment.action_space.n
        except:
            self.action_size = self.environment.action_space.shape[0]
        self.a = self.action_size

        print("state size is: ",self.state_size)
        print("action size is: ", self.action_size)
        self.memory = deque(maxlen=20000)
        if load_path:
            self.model = keras.models.load_model(load_path)
        else:
            self.model = self.create_model()
        self.alternate_model = self.model
        print(self.model.summary())

    def create_model(self):
        input = Input(shape=(self.state_size))

        if self.state_mode == "information":
            # o estado não é uma imagem
            out = Dense(24, activation="relu")(input)
            out = Dense(24, activation="relu")(out)
            out = Dense(self.action_size, activation="linear")(out)
        elif self.state_mode == "observation":
            # o estado é uma imagem
            out = Conv2D(128, kernel_size=(5,5), padding="same", activation="relu")(input)
            out = MaxPooling2D()(out)
            out = Conv2D(128, kernel_size=(3,3), padding="same", activation="relu")(out)
            out = MaxPooling2D()(out)
            out = Flatten()(out)
            out = Dense(24, activation="relu")(out)
            out = Dense(24, activation="relu")(out)
            out = Dense(self.action_size, activation="linear")(out)

        model = Model(inputs=input, outputs=out)
        model.compile(optimizer="adam", loss="mse")
        return model

    def state_reshape(self, state):
        shape = state.shape
        if self.state_mode == "observation":
            return np.reshape(state, [1, shape[0], shape[1], shape[2]])
        elif self.state_mode == "information":
            return np.reshape(state, [1, shape[0]])

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, next_state, action, reward, done):
        self.memory.append((state, next_state, action, reward, done))

    def replay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        batch = random.choices(self.memory, k=self.batch_size)
        for state, next_state, action, reward, done in batch:
            target = reward
            if not done:
                greedy_action = np.argmax(self.model.predict(next_state)[0])
                target = reward + self.discount_factor * self.alternate_model.predict(next_state)[0][greedy_action]
            final_target = self.model.predict(state)
            final_target[0][action] = target
            self.model.fit(state,final_target,verbose=0)

    def play(self):
        for episode in range(1, self.num_of_episodes + 1):
            state = self.environment.reset()
            state = self.state_reshape(state)
            self.alternate_model = self.model
            for t in itertools.count(0, 1):
                action = self.act(state)
                next_state, reward, done, _ = self.environment.step(action)
                self.environment.render()
                next_state = self.state_reshape(next_state)
                self.remember(state, next_state, action, reward, done)
                state = next_state
                if done:
                    print("episode number: ", episode, "time score: ", t, "epsilon", self.epsilon)
                    break
            self.replay()

            if episode % 100 == 0:
                path = Path('ddqn_save')
                if not path.exists():
                    path.mkdir(parents=True)
                self.model.save(str(path / (self.game + '_episode' + str(episode) + '.h5')))


def evaluate_agent(dqn):
    state = dqn.environment.reset()
    done = False
    while not done:
        action = np.argmax(dqn.model(np.expand_dims(state, 0)))
        dqn.environment.render()
        state, reward, done, _ = dqn.environment.step(action)
        time.sleep(0.1)
        if done:
            break


if __name__ == '__main__':
    game = "Pong-v0"
    #game = "CartPole-v1"
    #load_path = 'ddqn_save/CartPole-v1_episode1000.h5'
    load_path = None
    dqn = DQN(game, num_of_episodes=1000, load_path=load_path)
    dqn.play()
    evaluate_agent(dqn)

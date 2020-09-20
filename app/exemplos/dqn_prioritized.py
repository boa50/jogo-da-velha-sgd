# Link: https://github.com/Parsa33033/Deep-Reinforcement-Learning-DQN/blob/master/DQN-with-Prioritized-Experience-Replay.py

import numpy as np
import gym
import tensorflow as tf
import random
from collections import deque
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
        self.state_size = self.environment.observation_space.shape[0]
        self.action_size = self.environment.action_space.n
        self.load_path = load_path

        print("state size is: ",self.state_size)
        print("action size is: ", self.action_size)
        self.memory = deque(maxlen=20000)
        self.priority = deque(maxlen=20000)
        self.create_model()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def create_model(self):
        self.input = tf.placeholder(dtype=tf.float32, shape=(None, self.state_size))
        self.target = tf.placeholder(dtype=tf.float32, shape=(None, self.action_size))
        self.importance = tf.placeholder(dtype=tf.float32, shape=(None))
        out = tf.layers.dense(inputs=self.input, units=24, activation="relu")
        out = tf.layers.dense(inputs=out, units=24, activation="relu")
        self.output = tf.layers.dense(inputs=out, units=self.action_size, activation="linear")
        loss = tf.reduce_mean(tf.multiply(tf.square(self.output - self.target), self.importance))
        self.optimizer = tf.train.AdamOptimizer().minimize(loss)
        self.saver = tf.train.Saver()

    def predict(self, input):
        return self.sess.run(self.output, feed_dict={self.input: input})

    def fit(self, input, target, importance):
        self.sess.run(self.optimizer, feed_dict={self.input: input, self.target: target, self.importance: importance})

    def state_reshape(self, state):
        return np.expand_dims(state, 0)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.predict(state)[0])

    def remember(self, state, next_state, action, reward, done):
        self.prioritize(state, next_state, action, reward, done)

    def prioritize(self, state, next_state, action, reward, done, alpha=0.6):
        q_next = reward + self.discount_factor * np.max(self.predict(next_state)[0])
        q = self.predict(state)[0][action]
        p = (np.abs(q_next - q) + (np.e ** -10)) ** alpha
        self.priority.append(p)
        self.memory.append((state, next_state, action, reward, done))

    def get_priority_experience_batch(self):
        p_sum = np.sum(self.priority)
        prob = self.priority / p_sum
        sample_indices = random.choices(range(len(prob)), k=self.batch_size, weights=prob)
        importance = (1/prob) * (1/len(self.priority))
        importance = np.array(importance)[sample_indices]
        samples = np.array(self.memory)[sample_indices]
        return samples, importance

    def replay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        batch, importance = self.get_priority_experience_batch()
        for b, i in zip(batch, importance):
            state, next_state, action, reward, done = b
            target = reward
            if not done:
                target = reward + self.discount_factor * np.max(self.predict(next_state)[0])
            final_target = self.predict(state)
            final_target[0][action] = target
            imp = i ** (1 - self.epsilon) # Quão maior o epsilon, mais imp -> 0 (mais explorador é o agente)
            imp = np.reshape(imp, 1)
            self.fit(state, final_target, imp)

    def play(self):
        for episode in range(1, self.num_of_episodes + 1):
            state = self.environment.reset()
            state = self.state_reshape(state)
            for t in itertools.count(1, 1):
                action = self.act(state)
                next_state, reward, done, _ = self.environment.step(action)
                next_state = self.state_reshape(next_state)
                self.remember(state, next_state, action, reward, done)
                state = next_state
                if done:
                    print("episode number: ", episode, "episode time: ", t, "epsilon", self.epsilon)
                    self.save_info(episode, t)
                    break
            self.replay()

            if episode % 500 == 0:
                self.saver.save(self.sess, "dqn_prioritized/" + self.game + ".ckpt")
                print('Saving ckpt!')

    def save_info(self, episode, time):
        file = open("dqn_prioritized/" + self.game + "-" + str(self.num_of_episodes) + "-episodes-batchsize-"
                    + str(self.batch_size), 'a')
        file.write(str(episode) + " " + str(time) + " \n")
        file.close()

    def evaluate(self):
        self.epsilon = 0
        if self.load_path:
            self.saver.restore(self.sess, self.load_path)
        state = self.environment.reset()
        self.environment.render()
        state = self.state_reshape(state)
        for t in itertools.count(1, 1):
            action = self.act(state)
            next_state, reward, done, _ = self.environment.step(action)
            self.environment.render()
            next_state = self.state_reshape(next_state)
            state = next_state
            if done:
                print('Steps', t)
                break



if __name__ == "__main__":
    #game = "CartPole-v1"
    game= "Acrobot-v1"
    #load_path = None
    #load_path = 'dqn_prioritized/CartPole-v1-2500.ckpt'
    load_path = 'dqn_prioritized/Acrobot-v1-2500.ckpt'
    dqn = DQN(game, num_of_episodes=2500, load_path=load_path)
    dqn.play()
    dqn.evaluate()
# Link: https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/CliffWalk%20REINFORCE%20with%20Baseline%20Solution.ipynb
# Tensorflow 1.x

import itertools
import tensorflow as tf
import collections
from cliff_walking import CliffWalkingEnv
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt


EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])


def plot_episode_stats(stats, smoothing_window=10):
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.show()

    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.show()

    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    plt.show()

    return fig1, fig2, fig3


class PolicyEstimator():
    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], "state")
            self.action = tf.placeholder(dtype=tf.int32, name="action")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            state_one_hot = tf.one_hot(self.state, int(env.observation_space.n))
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=env.action_space.n,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)

            # Loss e train op
            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, {self.state: state})

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target, self.action: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class ValueEstimator():
    def __init__(self, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            state_one_hot = tf.one_hot(self.state, int(env.observation_space.n))
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


def reinforce(env, estimator_policy, estimator_value, num_episodes, gamma=1.0):
    stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    for i_episode in range(num_episodes):
        state = env.reset()
        episode = []
        for t in itertools.count():
            action_probs = estimator_policy.predict(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            episode.append(Transition(state=state, action=action, reward=reward, next_state=next_state, done=done))

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            print("\rStep {} @ Episode {}/{} ({})".format(
                t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")

            if done:
                break

            state = next_state

        # Passa o episódio e calcula o horizonte em cada passo. Com essas recompensas, o horizonte é calculado.
        for t, transition in enumerate(episode):
            # Calcula o horizonte descontado pelo gamma (não usa a fórmula iterativa apresentada em MC)
            reward_horizon = sum(gamma ** i * t.reward for i, t in enumerate(episode[t:]))
            # Calcula o baseline (função de valor) e a advantage (diferença entre o horizonte e a função de valor)
            baseline_value = estimator_value.predict(transition.state)
            advantage = reward_horizon - baseline_value
            # Usa o horizonte para atualizar o aproximador da função de valor (aproximação de MC)
            estimator_value.update(transition.state, reward_horizon)
            # Atualiza o estimador da política
            estimator_policy.update(transition.state, advantage, transition.action)

    return stats


def test_environment_main(env):
    env.reset()
    env.render()
    while True:
        action = np.random.randint(0, 3)
        observation, reward, done, _ = env.step(action)
        env.render()
        if done:
            break


if __name__ == '__main__':
    env = CliffWalkingEnv()

    test_environment_main(env)

    # Para inicializar o grafo de operações
    tf.reset_default_graph()

    global_step = tf.Variable(0, name="global_step", trainable=False)
    policy_estimator = PolicyEstimator()
    value_estimator = ValueEstimator()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        # O número de episódios para encontrar uma boa política pode variar bastante
        stats = reinforce(env, policy_estimator, value_estimator, 2000, gamma=1.0)

    plot_episode_stats(stats, smoothing_window=25)
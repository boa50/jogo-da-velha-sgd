# Link: https://keras.io/examples/rl/ddpg_pendulum/
# Tensorflow 2.x

import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class OUActionNoise:
  def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
    self.theta = theta
    self.mean = mean
    self.std_dev = std_deviation
    self.dt = dt
    self.x_initial = x_initial
    self.reset()

  def __call__(self):
    # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
    x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
    )
    # Store x into x_prev
    # Makes next noise dependent on current one
    self.x_prev = x
    return x

  def reset(self):
    if self.x_initial is not None:
      self.x_prev = self.x_initial
    else:
      self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):

        # Número de experiências ao máximo
        self.buffer_capacity = buffer_capacity
        # Número de tuplas para treinamento
        self.batch_size = batch_size

        # Contador de quantas vezes o método record foi utilizado
        self.buffer_counter = 0

        # Há um array para cada tipo de informação das tuplas de experiência armazenadas
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # A tupla é composta por (s, a, r, s')
    def record(self, obs_tuple):
        # Usa a operação de resto para ir reindexando os valores se a capacidade foi excedida
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Calcula a loss e faz o update dos parâmetros
    def learn(self):
        # Considera a quantidade de elementos atualmente no buffer
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Escolhe índices aleatórios
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Converte para bathes de tensores
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        # Treinando e atualizando as redes de Actor & Critic
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch)
            y = reward_batch + gamma * target_critic([next_state_batch, target_actions])
            critic_value = critic_model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch)
            critic_value = critic_model([state_batch, actions])
            # O negativo é usado pois originalmente busca-se maximizar os valores
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )


# Faz os updates dos parâmetros lentamente
# Baseado em tau, que tem um valor bem menor do que 1.
def update_target(tau):
    new_weights = []
    target_variables = target_critic.weights
    for i, variable in enumerate(critic_model.weights):
        new_weights.append(variable * tau + target_variables[i] * (1 - tau))

    target_critic.set_weights(new_weights)

    new_weights = []
    target_variables = target_actor.weights
    for i, variable in enumerate(actor_model.weights):
        new_weights.append(variable * tau + target_variables[i] * (1 - tau))

    target_actor.set_weights(new_weights)


def get_actor():
  last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

  inputs = layers.Input(shape=(num_states,))
  out = layers.Dense(512, activation="relu")(inputs)
  out = layers.BatchNormalization()(out)
  out = layers.Dense(512, activation="relu")(out)
  out = layers.BatchNormalization()(out)
  outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

  # As ações são valores contínuos no intervalo [-2, 2]
  outputs = outputs * upper_bound
  model = tf.keras.Model(inputs, outputs)
  return model


def get_critic():
  # Branch referente ao estado
  state_input = layers.Input(shape=(num_states))
  state_out = layers.Dense(16, activation="relu")(state_input)
  state_out = layers.BatchNormalization()(state_out)
  state_out = layers.Dense(32, activation="relu")(state_out)
  state_out = layers.BatchNormalization()(state_out)

  # Branch referente à ação
  action_input = layers.Input(shape=(num_actions))
  action_out = layers.Dense(32, activation="relu")(action_input)
  action_out = layers.BatchNormalization()(action_out)

  # Daqui em diante um conjunto comum de layers vai processar os dados
  # O processamento separado anterior é interessante pois os dados de estado e ações são bem diferentes
  concat = layers.Concatenate()([state_out, action_out])

  out = layers.Dense(512, activation="relu")(concat)
  out = layers.BatchNormalization()(out)
  out = layers.Dense(512, activation="relu")(out)
  out = layers.BatchNormalization()(out)
  outputs = layers.Dense(1)(out)

  # A saída é um único elemento representando a action-state value function
  model = tf.keras.Model([state_input, action_input], outputs)

  return model


def policy(state, noise_object):
  sampled_actions = tf.squeeze(actor_model(state))
  noise = noise_object()
  # Adicionando ruído à ação
  sampled_actions = sampled_actions.numpy() + noise

  # É necessário garantir que a ação esteja no intervalo permitido
  legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

  return [np.squeeze(legal_action)]


def run():
  ep_reward_list = []  # História dos rewards para cada episódio
  avg_reward_list = []

  for ep in range(total_episodes):
    prev_state = env.reset()
    episodic_reward = 0

    while True:
      # env.render()

      tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
      action = policy(tf_prev_state, ou_noise)
      state, reward, done, info = env.step(action)
      buffer.record((prev_state, action, reward, state))
      episodic_reward += reward
      buffer.learn()
      update_target(tau)

      if done:
        break

      prev_state = state

    ep_reward_list.append(episodic_reward)

    # Média dos últimos 40 episódios
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

  # Plot do gráfico da recompensa média por episódio
  plt.plot(avg_reward_list)
  plt.xlabel("Episode")
  plt.ylabel("Avg. Epsiodic Reward")
  plt.show()

  # Salvando os modelos
  path = Path('deep_ac')
  if not path.exists():
    path.mkdir(parents=True)

  actor_model.save(str(path / "pendulum_actor.h5"))
  critic_model.save(str(path / "pendulum_critic.h5"))
  target_actor.save(str(path / "pendulum_target_actor.h5"))
  target_critic.save(str(path / "pendulum_target_critic.h5"))


def evaluate():
  pass


if __name__ == '__main__':
  env = gym.make("Pendulum-v0")
  num_states = env.observation_space.shape[0]
  num_actions = env.action_space.shape[0]
  upper_bound = env.action_space.high[0]
  lower_bound = env.action_space.low[0]

  # Regularizador para a ação escolhida
  ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1))

  load_folder = 'deep_ac'
  load_folder = None
  # Redes online
  actor_model = get_actor()
  critic_model = get_critic()

  # Redes target
  target_actor = get_actor()
  target_critic = get_critic()

  # Sync das redes target
  target_actor.set_weights(actor_model.get_weights())
  target_critic.set_weights(critic_model.get_weights())

  # Características do otimizador
  critic_lr = 0.002
  actor_lr = 0.001
  critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
  actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

  # Parâmetros do treinamento
  total_episodes = 100
  gamma = 0.99
  tau = 0.005 # Usado para os updates das redes neurais

  # Replay buffer
  buffer = Buffer(50000, 64)

  # Método principal
  run()


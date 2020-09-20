# Paper: https://arxiv.org/pdf/1312.5602.pdf
# Link: https://github.com/EXJUSTICE/Deep_Q-learning_OpenAI_MissPacman/blob/master/TF2_MsPacMan_GC.ipynb

import numpy as np
import gym
import tensorflow as tf
from collections import deque, Counter
import random
import matplotlib.pyplot as plt
import time
from pathlib import Path


def preprocess_observation(obs):
  img = obs[1:176:2, ::2]
  img = img.mean(axis=2)
  img[img == COLOR] = 0
  img = (img - 128) / 128 - 1
  return img.reshape(88, 80, 1)


def q_network(input_shape, num_actions):
  input_t = tf.keras.layers.Input(shape=input_shape)
  tensor1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(8, 8), strides=4, padding='SAME', activation='relu')(input_t)
  tensor2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, padding='SAME', activation='relu')(tensor1)
  tensor3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='SAME', activation='relu')(tensor2)
  flat_t = tf.keras.layers.Flatten()(tensor3)
  fc = tf.keras.layers.Dense(units=128)(flat_t)
  output = tf.keras.layers.Dense(units=num_actions)(fc)
  model = tf.keras.Model(inputs=input_t, outputs=output)
  return model


def epsilon_greedy(state, step):
  global epsilon
  epsilon = max(epsilon_min, epsilon_max - (epsilon_max - epsilon_min) * step / eps_decay_steps)
  if np.random.rand(1) < epsilon:
    return np.random.randint(NUM_ACTIONS)
  else:
    return np.argmax(main_model(state))


def sample_from_replay_buffer(batch_size):
  replay = random.sample(exp_buffer, batch_size)
  return np.transpose(replay)


def run():
  reward_history = []
  global_step = 0

  save_path = Path('pacman_saves')
  if not save_path.exists():
    save_path.mkdir(parents=True)

  for eps in range(num_episodes):
    done = False
    state = env.reset()
    steps = 0
    episodic_reward = 0
    actions_counter = Counter()

    state = np.expand_dims(preprocess_observation(state), 0)
    while not done:
      action = epsilon_greedy(state, global_step)
      actions_counter[str(action)] += 1
      next_state, reward, done, _ = env.step(action)
      #print('action', action, 'reward', reward)
      #env.render()

      next_state = np.expand_dims(preprocess_observation(next_state), 0)
      exp_buffer.append([state, action, next_state, reward, done])

      if global_step % steps_train == 0 and len(exp_buffer) >= batch_size:
        r_states, r_actions, r_next_states, r_rewards, r_done = sample_from_replay_buffer(batch_size)
        r_states = tf.constant(np.vstack(r_states))
        r_next_states = tf.constant(np.vstack(r_next_states))

        q_prediction = main_model(r_states)
        q_future = target_model(r_next_states)

        update_indexes = tf.stack([tf.range(batch_size), r_actions], axis=-1)
        update_values = r_rewards + gamma * tf.reduce_max(q_future, axis=-1) * (1 - r_done)
        update_target = tf.tensor_scatter_nd_update(q_prediction, update_indexes, update_values)
        main_model.fit(r_states, update_target, epochs=1, verbose=0)

      if global_step % copy_steps == 0:
        target_model.set_weights(main_model.get_weights())

      state = next_state
      steps += 1
      global_step += 1
      episodic_reward += reward
      reward_history.append(episodic_reward)

    print('Passos por episódio', steps, 'Recompensa', episodic_reward, '#Episódio', eps + 1,
          'epsilon', epsilon)
    if eps % 100 == 0:
      main_model.save('./pacman_saves/NetworkInEPS{}.h5'.format(eps))


def evaluate_agent():
  state = env.reset()
  done = False
  while not done:
    state = preprocess_observation(state)
    action = np.argmax(main_model(np.expand_dims(state, 0)))
    env.render()
    state, reward, done, _ = env.step(action)
    time.sleep(0.1)

    if done:
      break


def test_preprocessing():
  env = gym.make("MsPacman-v0")
  n_outputs = env.action_space.n
  print(n_outputs)
  print(env.env.get_action_meanings())

  observation = env.reset()
  for _ in range(100):
    observation, _, _, _ = env.step(1)

  plt.imshow(observation)
  plt.show()

  # Para observar o impacto do pré-processamento
  obs_preprocessed = preprocess_observation(observation)
  plt.imshow(obs_preprocessed[..., 0])
  plt.show()
  print(observation.shape)
  print(obs_preprocessed.shape)

  # Exibindo em escala de cinza
  plt.imshow(obs_preprocessed[..., 0], cmap='gray')
  plt.show()


def play_game():
  from gym.utils import play
  env = gym.make("MsPacman-v0")
  play.play(env, zoom=3)


if __name__ == '__main__':
  #play_game()
  #test_preprocessing()

  epsilon = 0.5
  epsilon_min = 0.1
  epsilon_max = 1.0
  eps_decay_steps = 50000  # O decay do artigo foi com relação a 1M frames no range [1.0; 0.1]
  buffer_len = 50000  # DeepMind usou uma replay memory de 1M amostras
  exp_buffer = deque(maxlen=buffer_len)
  num_episodes = 100  # O treinamento durou por 10 milhões de frames.
  batch_size = 32
  input_shape = (88, 80, 1)
  learning_rate = 0.001
  gamma = 0.9
  global_step = 0
  copy_steps = 250
  steps_train = 4
  start_steps = 10
  load_file = 'pacman_saves/NetworkInEPS100.h5'

  env = gym.make("MsPacman-v0")
  COLOR = np.array([210, 164, 74]).mean()
  NUM_ACTIONS = env.action_space.n
  # NUM_ACTIONS = 5

  if load_file:
    main_model = tf.keras.models.load_model(load_file)
  else:
    main_model = q_network(input_shape, NUM_ACTIONS)
    main_model.compile(tf.keras.optimizers.Adam(learning_rate), 'mse')
  target_model = q_network(input_shape, NUM_ACTIONS)
  run()
  evaluate_agent()
  env.close()

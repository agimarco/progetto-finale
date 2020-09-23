import numpy as np
import tensorflow as tf
from tensorflow import keras

from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper, DiscreteWrapper
from my_utils import EasyObservation, DtRewardWrapper, MyDiscreteWrapperTrain, NoiseWrapper

tf.random.set_seed(42)
np.random.seed(42)

# Create the environment 
env = DuckietownEnv(
    seed=123, # random seed
    map_name="loop_empty",
    max_steps=500001, # we don't want the gym to reset itself
    domain_rand=0,
    camera_width=640,
    camera_height=480,
    accept_start_angle_deg=4, # start close to straight
    full_transparency=True,
    distortion=True
)
# discrete actions, 4 value observation and modified reward
env = NoiseWrapper(env)
env = MyDiscreteWrapperTrain(env)
env = EasyObservation(env)
env = DtRewardWrapper(env)

input_shape = env.observation_space.shape
n_outputs = env.action_space.n

model = keras.models.Sequential([
    keras.layers.Dense(32, activation="elu", input_shape=input_shape),
    keras.layers.Dense(32, activation="elu"),
    keras.layers.Dense(n_outputs)
])

target = keras.models.clone_model(model)
target.set_weights(model.get_weights())

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand()<epsilon:
        action = np.random.randint(n_outputs)
        return action
    else:
        Q_values = model.predict(state[np.newaxis])
        action = np.argmax(Q_values[0])
        return action

# replay buffer

from collections import deque

replay_buffer = deque(maxlen=50000)

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones

def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

batch_size = 32
discount_factor = 0.95
optimizer = keras.optimizers.Adam(lr=1e-3)
loss_fn = keras.losses.mean_squared_error
loss_fn = keras.losses.Huber()

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    # model network for selecting the action
    next_Q_values = model.predict(next_states)
    best_next_actions = np.argmax(next_Q_values, axis=1)
    next_mask = tf.one_hot(best_next_actions, n_outputs).numpy()
    # target network for evaluating the action
    next_best_Q_values = (target.predict(next_states) * next_mask).sum(axis=1)
    target_Q_values = (rewards +
                        (1 - dones) * discount_factor * next_best_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)    
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# training

N_STEPS = 50000

steps_per_episode = 0
episode = 0
rewards = []
best_score = 0


obs = env.reset()
for step in range(N_STEPS):
    epsilon = max(1 - step / 5000, 0.02)
    steps_per_episode += 1
    obs, reward, done, info = play_one_step(env, obs, epsilon)
    if done or step == N_STEPS-1:
        episode += 1
        print("\rEpisode: {}, StepsPerEp: {}, Steps: {}, eps: {:.3f}".format(episode, steps_per_episode, step + 1, epsilon))
        obs = env.reset()
        rewards.append(steps_per_episode)
        if steps_per_episode > best_score:
            best_weights = model.get_weights()
            model.save_weights('weights/test-dqn-duckietown.h5')
            best_score = steps_per_episode
        steps_per_episode = 0
    if step > 1000:
        training_step(batch_size)
    #if step % 500 == 0:
    #    target.set_weights(model.get_weights())
    # Alternatively, you can do soft updates at each step:
    if step > 1000:
        target_weights = target.get_weights()
        online_weights = model.get_weights()
        for index in range(len(target_weights)):
            target_weights[index] = 0.99 * target_weights[index] + 0.01 * online_weights[index]
        target.set_weights(target_weights)

model.set_weights(best_weights)


import matplotlib.pyplot as plt

# graph
plt.figure(figsize=(8,4))
plt.plot(rewards)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Sum of rewards", fontsize=14)
plt.show()

import os 
import numpy as np 
from collections import deque

import tensorflow as tf

# algorithm class
class DDQN():
    def __init__(self, env, batch_size=32, learning_rate=1e-3, loss_fn='hub', activation='elu',
                    discount_factor=0.95, cnn=False, buffer_len=50000,
                    initial_epsilon=1, final_epsilon=0.02, epsilon_decay=1e-4):
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        self.loss_fn = tf.keras.losses.Huber()
        if loss_fn=='mse':
            self.loss_fn = tf.keras.losses.mean_squared_error

        self.env = env
        self.input_shape = env.observation_space.shape
        self.n_outputs = env.action_space.n
        self.model = tf.keras.Sequential()

        # convolutional neural network
        if cnn:
            self.model.add(tf.keras.layers.Dense(32, activation="elu", input_shape=self.input_shape))
            self.model.add(tf.keras.layers.Dense(32, activation="elu"))
            self.model.add(tf.keras.layers.Dense(self.n_outputs))
            
        # multi layered perceptron
        if not cnn:
            self.model.add(tf.keras.layers.Dense(32, activation=activation, input_shape=self.input_shape))
            self.model.add(tf.keras.layers.Dense(32, activation=activation))
            self.model.add(tf.keras.layers.Dense(self.n_outputs))
            
        self.target = tf.keras.models.clone_model(self.model)
        self.target.set_weights(self.model.get_weights())    
        
        self.replay_buffer = deque(maxlen=buffer_len)

        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay

        self.epsilon = self.initial_epsilon

    # chooses an action either randomly or by making a prediction
    def epsilon_greedy_policy(self, state):
        if np.random.rand()<self.epsilon:
            action = np.random.randint(self.n_outputs)
            return action
        else:
            Q_values = self.model.predict(state[np.newaxis])
            action = np.argmax(Q_values[0])
            return action

    # plays one step (following epsilon greedy policy) and saves it in the replay buffer
    def play_one_step(self, state):
        action = self.epsilon_greedy_policy(state)
        next_state, reward, done, info = self.env.step(action)
        self.replay_buffer.append((state, action, reward, next_state, done))
        return next_state, reward, done, info

    # samples a batch of random experiences from the replay buffer 
    def sample_batch(self):
        indices = np.random.randint(len(self.replay_buffer), size=self.batch_size)
        batch = [self.replay_buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = [
            np.array([sample[elem_index] for sample in batch])
            for elem_index in range(5)]
        return states, actions, rewards, next_states, dones
    
    def training_step(self):
        experiences = self.sample_batch()
        states, actions, rewards, next_states, dones = experiences
        # model network for selecting the action
        next_Q_values = self.model.predict(next_states)
        best_next_actions = np.argmax(next_Q_values, axis=1)
        next_mask = tf.one_hot(best_next_actions, self.n_outputs).numpy()
        # target network for evaluating the action
        next_best_Q_values = (self.target.predict(next_states) * next_mask).sum(axis=1)
        target_Q_values = (rewards +
                            (1 - dones) * self.discount_factor * next_best_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)    
        mask = tf.one_hot(actions, self.n_outputs)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    # training
    def learn(self, timesteps=50000, tensorboard_log=True):
        steps_per_episode = 0
        episode = 0
        rewards = []
        best_score = 0

        obs = self.env.reset()
        for step in range(timesteps):
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.final_epsilon)
            obs, reward, done, info = self.play_one_step(obs)
            steps_per_episode += 1
            if done or step == timesteps-1:
                episode += 1
                print(f"\rEpisode: {episode}, StepsPerEp: {steps_per_episode}, Steps: {step + 1}, eps: {self.epsilon:.3f}")
                obs = self.env.reset()
                rewards.append(steps_per_episode)
                if steps_per_episode > best_score:
                    best_weights = self.model.get_weights()
                    best_score = steps_per_episode
                steps_per_episode = 0
            if step > 1000:
                self.training_step()
            #if step % 500 == 0:
            #    target.set_weights(model.get_weights())
            # Alternatively, you can do soft updates at each step:
            if step > 1000:
                target_weights = self.target.get_weights()
                online_weights = self.model.get_weights()
                for index in range(len(target_weights)):
                    target_weights[index] = 0.99 * target_weights[index] + 0.01 * online_weights[index]
                self.target.set_weights(target_weights)

        self.model.set_weights(best_weights)

    def save(self, weights):
        weights_dir = os.path.dirname(weights)
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        self.model.save_weights(weights)

    def load(self, weights):
        self.model.load_weights(weights)

    def predict(self, state):
        Q_values = self.model.predict(state[np.newaxis])
        action = np.argmax(Q_values[0])
        return action


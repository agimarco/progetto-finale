import os 
import numpy as np 
from collections import deque

import tensorflow as tf

# algorithm class
class DDQN():
    def __init__(self, env, batch_size=32, learning_rate=1e-3, loss_fn='hub',
                    activation='elu', cnn=False, mlp_width=32,
                    discount_factor=0.95, buffer_len=50000,
                    initial_epsilon=1, final_epsilon=0.02):
        '''
        Parameters:
        env: the environment to train on
        batch_size: batch size
        learning_rate: larning rate used with Adam optimizer
        loss_fn: loss function used for training, either 'hub' (for Huber function) or 'mse' 
        activation: sets the activation function using keras parameters ('elu', 'relu', etc...)
        cnn: True to use a convolutional neural network 
        mlp_width: number of neurons for each layer of mlp network (32 or 64)
        discount_factor: the gamma parameter in the Q value function
        buffer_len: maximum length used for replay buffer
        initial_epsilon: initial value of epsilon
        final_epsilon: final value of epsilon (always final_epsilon probability of random action)
        '''
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        self.loss_fn = tf.keras.losses.Huber()
        if loss_fn=='mse':
            self.loss_fn = tf.keras.losses.mean_squared_error

        self.env = env
        self.input_shape = env.observation_space.shape
        self.output_shape = env.action_space.n
        self.model = tf.keras.Sequential()

        # convolutional neural network
        if cnn:
            self.model.add(tf.keras.layers.Conv2D(32, kernel_size=8, strides=4, activation=activation, input_shape=self.input_shape))
            self.model.add(tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation=activation))
            self.model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation=activation))
            self.model.add(tf.keras.layers.Flatten())
            self.model.add(tf.keras.layers.Dense(512, activation=activation))
            self.model.add(tf.keras.layers.Dense(self.output_shape))
            
        # multi layered perceptron
        if not cnn:
            self.model.add(tf.keras.layers.Dense(mlp_width, activation=activation, input_shape=self.input_shape))
            self.model.add(tf.keras.layers.Dense(mlp_width, activation=activation))
            self.model.add(tf.keras.layers.Dense(self.output_shape))
            
        self.target = tf.keras.models.clone_model(self.model)
        self.target.set_weights(self.model.get_weights())    
        
        self.reset_weights = self.model.get_weights()

        self.replay_buffer = deque(maxlen=buffer_len)

        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon = self.initial_epsilon

    # chooses an action either randomly or by making a prediction
    def _epsilon_greedy(self, state):
        if np.random.rand()<self.epsilon:
            action = np.random.randint(self.output_shape)
            return action
        else:
            action = self.predict(state)
            return action

    # plays one step (following epsilon greedy policy) and saves it in the replay buffer
    def _one_step(self, state):
        action = self._epsilon_greedy(state)
        next_state, reward, done, info = self.env.step(action)
        self.replay_buffer.append((state, action, reward, next_state, done))
        return next_state, reward, done, info

    # samples a batch of random experiences from the replay buffer 
    def _sample_batch(self):
        indices = np.random.randint(len(self.replay_buffer), size=self.batch_size)
        batch = [self.replay_buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = [
            np.array([sample[elem_index] for sample in batch]) for elem_index in range(5)]
        return states, actions, rewards, next_states, dones
    
    # a training step
    def _training(self):
        # sample a batch of experiences from the replay buffer
        states, actions, rewards, next_states, dones = self._sample_batch()
        # model network for selecting the action
        best_actions = np.argmax(self.model.predict(next_states), axis=1)
        mask = tf.one_hot(best_actions, self.output_shape).numpy()
        # target network for evaluating the action
        target_values = (rewards + (1 - dones) * self.discount_factor * (self.target.predict(next_states) * mask).sum(axis=1))
        target_values = target_values.reshape(-1, 1)    
        mask = tf.one_hot(actions, self.output_shape)
        with tf.GradientTape() as tape:
            values = tf.reduce_sum(self.model(states) * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_values, values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    # training loop
    def learn(self, timesteps=50000, learning_starts=1000, update_freq=1,
                reset_epsilon=True, epsilon_decay=2e-4, tensorboard_log_name="training"):
        '''
        Parameters:
        timesteps: number of training steps it performs
        learning_starts: learning starts 
        update_freq: updates the target network every update_freq steps, if update_freq == 1 then uses soft updates
        reset_epsilon: reset epsilon before training
        epsilon_decay: each step it decreases epsilon by epsilon_decay
        tensorboard_log_name: name used to save tensorboard results
        '''
        # reset epsilon
        self.epsilon = self.initial_epsilon if reset_epsilon else self.epsilon
        steps_per_episode = 0
        episode = 0
        longest_run = 0

        # tensorboard log
        summary_writer = tf.summary.create_file_writer(os.path.join("logs", tensorboard_log_name))

        obs = self.env.reset()
        for step in range(timesteps):
            self.epsilon = max(self.epsilon - epsilon_decay, self.final_epsilon)
            obs, reward, done, info = self._one_step(obs)
            steps_per_episode += 1
            if done or step == timesteps-1:
                episode += 1
                print(f"\rEpisode: {episode}, StepsPerEp: {steps_per_episode}, Steps: {step + 1}, eps: {self.epsilon:.3f}")
                obs = self.env.reset()
                if steps_per_episode > longest_run:
                    best_weights = self.model.get_weights()
                    longest_run = steps_per_episode
                # tensorboard log
                with summary_writer.as_default():
                  tf.summary.scalar('steps_per_episode', steps_per_episode, episode)
                summary_writer.flush()
                steps_per_episode = 0
            if step > learning_starts:
                self._training()
            # target network update
            # soft updates
            if update_freq==1 and step > learning_starts:
                target_weights = self.target.get_weights()
                online_weights = self.model.get_weights()
                for index in range(len(target_weights)):
                    target_weights[index] = 0.99 * target_weights[index] + 0.01 * online_weights[index]
                self.target.set_weights(target_weights)
            # update target every update_freq steps
            if update_freq!=1 and step % update_freq == 0:
                self.target.set_weights(self.model.get_weights())
            
        self.model.set_weights(best_weights)

    def save(self, weights):
        weights_dir = os.path.dirname(weights)
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        self.model.save_weights(weights)

    def load(self, weights):
        self.model.load_weights(weights)

    def reset_model(self):
        self.model.set_weights(self.reset_weights)
        self.target.set_weights(self.reset_weights)

    def predict(self, state):
        Q_values = self.model.predict(state[np.newaxis])
        action = np.argmax(Q_values[0])
        return action


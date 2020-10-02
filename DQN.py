import os 
import numpy as np 
from collections import deque

import tensorflow as tf

# algorithm class
class DQN():
    def __init__(self, env, batch_size=32, learning_rate=1e-3, loss_fn='mse',
                    activation='relu', cnn=False, mlp_width=64,
                    discount_factor=0.99, buffer_len=50000,
                    initial_epsilon=1, final_epsilon=0.02,
                    use_target=True):
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
        use_target: set this to False to not use the target network
        '''
        self.use_target = use_target
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

        self.cnn = cnn

        # convolutional neural network
        if self.cnn:
            self.model.add(tf.keras.layers.Conv2D(32, kernel_size=8, strides=4, activation=activation, input_shape=self.input_shape))
            self.model.add(tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation=activation))
            self.model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation=activation))
            self.model.add(tf.keras.layers.Flatten())
            self.model.add(tf.keras.layers.Dense(512, activation=activation))
            self.model.add(tf.keras.layers.Dense(self.output_shape))
            
        # multi layered perceptron
        if not self.cnn:
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
   
    # plays one step (following epsilon greedy policy) and saves it in the replay buffer
    def _playing_step(self, state):
        # epsilon greedy policy for choosing action (either randomly or by making a prediction)
        action, q_value = self.predict(state)
        if np.random.rand()<self.epsilon:
            action = np.random.randint(self.output_shape)
        # perform action
        next_state, reward, done, info = self.env.step(action)
        # save step in the replay buffer
        self.replay_buffer.append((state, action, reward, next_state, done))
        return next_state, reward, done, info, q_value
    
    # a training step: samples batch, computes q values and target q values then computes loss and applies gradient 
    def _training_step(self):
        # sample a batch of random experiences from the replay buffer
        indices = np.random.randint(len(self.replay_buffer), size=self.batch_size)
        batch = [self.replay_buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = [
            np.array([sample[elem_index] for sample in batch]) for elem_index in range(5)]
        # normalize if using cnn
        if self.cnn:
            obs_lo = self.env.observation_space.low[0, 0, 0]
            obs_hi = self.env.observation_space.high[0, 0, 0]
            states = (states - obs_lo) / (obs_hi - obs_lo)
            next_states = (next_states - obs_lo) / (obs_hi - obs_lo)
        # using a target network or not
        if self.use_target:
            max_next_values = np.max(self.target.predict(next_states), axis=1)
        else:
            max_next_values = np.max(self.model.predict(next_states), axis=1)
        target_values = (rewards +
                       (1 - dones) * self.discount_factor * max_next_values)
        target_values = target_values.reshape(-1, 1)
        mask = tf.one_hot(actions, self.output_shape)
        with tf.GradientTape() as tape:
            values = tf.reduce_sum(self.model(states) * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_values, values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, values

    # training loop
    def learn(self, timesteps=50000, max_ep_steps=0, learning_starts=1000, update_freq=1, tau=0.99,
                reset_epsilon=True, epsilon_decay=2e-4, metric_update=100, tensorboard_log_name="training",
                cool_gpu=0):
        '''
        Parameters:
        timesteps: number of training steps it performs
        max_ep_steps: maximum steps for each episode
        learning_starts: learning starts 
        update_freq: updates the target network every update_freq steps, if update_freq == 1 then uses soft updates
        tau: used when performing soft updates to update the target network
        reset_epsilon: reset epsilon before training
        epsilon_decay: each step it decreases epsilon by epsilon_decay
        metric_update: send metrics to tensorflow every metric_update steps
        tensorboard_log_name: name used to save tensorboard results
        '''
        # reset epsilon
        self.epsilon = self.initial_epsilon if reset_epsilon else self.epsilon
        steps_per_episode = 0
        episode = 0
        longest_run = 0
        metrics_updated = 0
        metrics = dict({'rewards_ep':[], 'losses_ep':[], 'values_ep':[], 'rewards_p':[], 'losses_p':[], 'values_p':[]})
        
        # tensorboard log
        summary_writer = tf.summary.create_file_writer(os.path.join("logs", tensorboard_log_name))

        obs = self.env.reset()
        for step in range(1, timesteps + 1):
            self.epsilon = max(self.epsilon - epsilon_decay, self.final_epsilon)
            obs, reward, done, info, q_value = self._playing_step(obs)
            metrics['rewards_ep'].append(reward)
            metrics['rewards_p'].append(reward)
            metrics['values_ep'].append(q_value)
            metrics['values_p'].append(q_value)
            steps_per_episode += 1
            if done or step == timesteps or steps_per_episode==max_ep_steps:
                episode += 1
                print(f"\rEpisode: {episode}, StepsPerEp: {steps_per_episode}, Steps: {step}, eps: {self.epsilon:.3f}")
                obs = self.env.reset()
                # saves best weights
                if steps_per_episode >= longest_run:
                    best_weights = self.model.get_weights()
                    longest_run = steps_per_episode
                # log the steps per episode, average reward and cumulative reward
                with summary_writer.as_default():
                    tf.summary.scalar('steps_per_episode', steps_per_episode, episode)
                    tf.summary.scalar('average_reward_per_episode', np.average(metrics['rewards_ep']), episode)
                    tf.summary.scalar('cumulative_reward_per_episode', np.sum(metrics['rewards_ep']), episode)
                    tf.summary.scalar('mean_q_values_per_episode', np.average(metrics['values_ep']), episode)
                    tf.summary.scalar('avg_loss_per_episode', np.average(metrics['losses_ep']), episode)
                summary_writer.flush()
                metrics['rewards_ep'].clear()
                metrics['losses_ep'].clear()
                metrics['values_ep'].clear()
                steps_per_episode = 0
            if step > learning_starts:
                loss, val = self._training_step()
                metrics['losses_ep'].append(loss)
                metrics['losses_p'].append(loss)
                # log loss
                with summary_writer.as_default():
                  tf.summary.scalar('step_loss', loss, step)
                summary_writer.flush()
            # target network update
            # soft updates
            if self.use_target and update_freq==1 and step > learning_starts:
                target_weights = self.target.get_weights()
                online_weights = self.model.get_weights()
                for index in range(len(target_weights)):
                    target_weights[index] = tau * target_weights[index] + (1-tau) * online_weights[index]
                self.target.set_weights(target_weights)
            # update target every update_freq steps
            if self.use_target and update_freq!=1 and step % update_freq == 0:
                self.target.set_weights(self.model.get_weights())
            # writing metrics
            if step % metric_update == 0:
                metrics_updated += 1
                with summary_writer.as_default():
                    tf.summary.scalar('average_reward_per_'+str(metric_update)+'steps', np.average(metrics['rewards_p']), metrics_updated)
                    tf.summary.scalar('cumulative_reward_per_'+str(metric_update)+'steps', np.sum(metrics['rewards_p']), metrics_updated)
                    tf.summary.scalar('mean_q_values_per_'+str(metric_update)+'steps', np.average(metrics['values_p']), metrics_updated)
                    tf.summary.scalar('avg_loss_per_'+str(metric_update)+'steps', np.average(metrics['losses_p']), metrics_updated)
                summary_writer.flush()
                metrics['rewards_p'].clear()
                metrics['losses_p'].clear()
                metrics['values_p'].clear()
            # if using GPU stops for 30 seconds every 500 steps to avoid overheating
            if cool_gpu > 0 and step % cool_gpu == 0:  
                time.sleep(30)
        self.model.set_weights(best_weights)

    def predict(self, state, Model5=False):
        # normalize if using cnn
        if self.cnn and not Model5:
            obs_lo = self.env.observation_space.low[0, 0, 0]
            obs_hi = self.env.observation_space.high[0, 0, 0]
            state = (state - obs_lo) / (obs_hi - obs_lo)
        Q_values = self.model.predict(state[np.newaxis])
        action = np.argmax(Q_values[0])
        q_value = np.max(Q_values[0])
        return action, q_value
    
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


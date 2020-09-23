from gym_duckietown.envs import DuckietownEnv
from project_utils import PositionObservation, DtRewardWrapper, DiscreteActionWrapperTrain, NoiseWrapper
import tensorflow as tf

from DQN import DQN

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Create the environment 
env = DuckietownEnv(
    seed=123, # random seed
    map_name="loop_empty",
    max_steps=500001, # we don't want the gym to reset itself
    domain_rand=0,
    accept_start_angle_deg=4, # start close to straight
    full_transparency=True,
    graphics=False,
)
# discrete actions, 4 value observation and modified reward
env = NoiseWrapper(env)
env = DiscreteActionWrapperTrain(env)
env = PositionObservation(env)
env = DtRewardWrapper(env)

env.reset()

N_TRIALS = 1

# best hyperparameters
tmsteps = 200000                 
buffer_size = 50000             # usually buffer len is same as timesteps
eps_dec = 1 / (tmsteps // 10)    # 10% of steps have a decaying epsilon
update_freq = 500               # target network gets updated
lr = 1e-3                       # 1e-3 for MLP 
discount_factor = 0.99
learning_starts = 1000 
max_ep_steps = 2000                # 0 or 2000 (or more) in the best case it takes 500 steps to complete one loop
metric_update = 500                # update metric every 500 steps

for i in range(1, N_TRIALS + 1):
    print("\nTraining number "+str(i))
    train_details = 'DQN_NoTarget_Model_' + str(i)
    weights = 'weights/dqn_duckietown_weights' + train_details + '.h5'

    model = DQN(env, batch_size=32, learning_rate=lr, loss_fn='mse',
                        activation='relu', cnn=False, mlp_width=64, 
                        discount_factor=discount_factor, buffer_len=buffer_size, 
                        initial_epsilon=1, final_epsilon=0.02,
                        use_target=False)
    model.learn(timesteps=tmsteps, max_ep_steps=max_ep_steps, learning_starts=learning_starts, update_freq=update_freq, tau=0.99,
                     reset_epsilon=True, epsilon_decay=eps_dec, metric_update=metric_update, tensorboard_log_name='dqn_duckietown_' + train_details)
    model.save(weights)

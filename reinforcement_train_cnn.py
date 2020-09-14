from gym_duckietown.envs import DuckietownEnv
from my_utils import EasyObservation, DtRewardWrapper, MyDiscreteWrapperTrain, NoiseWrapper, ResizeWrapper
import tensorflow as tf

from DDQN import DDQN

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
env = ResizeWrapper(env)
env = MyDiscreteWrapperTrain(env)
env = DtRewardWrapper(env)

env.reset()

train_details = '50000steps'

weights = 'weights/ddqn_duckietown_cnn_weights' + train_details + '.h5'

model = DDQN(env, batch_size=32, learning_rate=1e-3, loss_fn='hub',
                    activation='relu', cnn=True, mlp_width=32,
                    discount_factor=0.95, buffer_len=50000,
                    initial_epsilon=1, final_epsilon=0.02)
model.learn(timesteps=50000, learning_starts=1000, update_freq=500, reset_epsilon=True,
                epsilon_decay=1e-4, tensorboard_log_name='ddqn_duckietown_cnn_' + train_details)
model.save(weights)

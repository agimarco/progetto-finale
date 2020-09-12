from gym_duckietown.envs import DuckietownEnv
from my_utils import EasyObservation, DtRewardWrapper, MyDiscreteWrapperTrain, NoiseWrapper

from DDQN import DDQN

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

env.reset()

train_details = 'relu75000steps64width500upfreq'

weights = 'weights/ddqn_duckietown_weights' + train_details + '.h5'

model = DDQN(env, activation='relu', mlp_width=64, buffer_len=75000)
model.learn(timesteps=75000, learning_starts=1000, update_freq=500, reset_epsilon=True,
             epsilon_decay=1e-4, tensorboard_log_name='ddqn_duckietown_' + train_details)
model.save(weights)

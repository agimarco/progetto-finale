import numpy as np
import tensorflow as tf
from tensorflow import keras

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

weights = 'weights/ddqn_duckietown_weights.h5'

model = DDQN(env)
model.learn(timesteps=50000, tensorboard_log='logs')
model.save(weights)

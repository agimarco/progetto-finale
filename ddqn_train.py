import argparse

import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import DiscreteWrapper
from my_utils import EasyObservation, DtRewardWrapper, MyDiscreteWrapper, MyDiscreteWrapperTrain

from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN

parser = argparse.ArgumentParser()
parser.add_argument('--details', default='', help='used to set the weights name')
args = parser.parse_args()

weights_name = "deepq_duckietown"
if(args.details!=''):
    weights_name += "_"+args.details

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
env = MyDiscreteWrapperTrain(env)
env = EasyObservation(env)
env = DtRewardWrapper(env)


model = DQN(MlpPolicy, env, verbose=2)
model.learn(total_timesteps=50000)     #25000
model.save(weights_name)





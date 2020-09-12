from gym_duckietown.envs import DuckietownEnv
from my_utils import EasyObservation, DtRewardWrapper, MyDiscreteWrapperTrain, NoiseWrapper, ResizeWrapper

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
env = ResizeWrapper(env)
env = MyDiscreteWrapperTrain(env)
env = DtRewardWrapper(env)

env.reset()

train_details = '50000steps'

weights = 'weights/ddqn_duckietown_cnn_weights' + train_details + '.h5'

model = DDQN(env, cnn=True)
model.learn(timesteps=50000, tensorboard_log_name='ddqn_duckietown_cnn_' + train_details)
model.save(weights)

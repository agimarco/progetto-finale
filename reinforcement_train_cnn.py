from gym_duckietown.envs import DuckietownEnv

from project_utils import DtRewardWrapper, DiscreteActionWrapperTrain, ResizeWrapper, NormalizeWrapper, ImgWrapper, NoiseWrapper
import tensorflow as tf

from DDQN import DDQN

# gpu usage memory growth (incremental memory use)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

'''
# maximum memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=256)])
  except RuntimeError as e:
    print(e)
'''

# add frame_skip to env arguments
frame_skip = 4      # we train on 1/4 of frames

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
    distortion=False,
    graphics=True,
)
# discrete actions, 4 value observation and modified reward
env = NoiseWrapper(env)
env = DiscreteActionWrapperTrain(env)
env = DtRewardWrapper(env)
env = ResizeWrapper(env)
#env = NormalizeWrapper(env)
#env = ImgWrapper(env)

env.reset()

N_TRIALS = 10

# best hyperparameters
tmsteps = 200000                 # buffer len is same as timesteps
buffer_size = 25000   # 25000 instead of 50000 because we have not enough memory
eps_dec = 1 / (tmsteps // 10)    # 20% of steps have a decaying epsilon
update_freq = 500               # target network gets updated
lr = 1e-4                       # 1e-4 for CNN
discount_factor = 0.99
learning_starts = 1000 
max_ep_steps = 2000                # 0 or 2000 (or more) in the best case it takes 500 steps to complete one loop
metric_update = 500                # update metric every 500 steps

for i in range(1, N_TRIALS + 1):
  print("\nTraining number "+str(i))
  train_details = 'Model_' + str(i)
  weights = 'weights/ddqn_duckietown_cnn_weights' + train_details + '.h5'
  # special attention to update_freq
  model = DDQN(env, batch_size=32, learning_rate=lr, loss_fn='mse',
                      activation='relu', cnn=True, mlp_width=64,
                      discount_factor=discount_factor, buffer_len=buffer_size,
                      initial_epsilon=1, final_epsilon=0.02)
  model.learn(timesteps=tmsteps, max_ep_steps=max_ep_steps, learning_starts=learning_starts, update_freq=update_freq, tau=0.99,
                reset_epsilon=True, epsilon_decay=eps_dec, metric_update=metric_update,
                tensorboard_log_name='ddqn_duckietown_cnn_' + train_details, cool_gpu=500)
  model.save(weights)

import argparse
import sys
import pyglet
from pyglet.window import key
import numpy as np
import tensorflow as tf

from gym_duckietown.envs import DuckietownEnv
from my_utils import EasyObservation, DtRewardWrapper, MyDiscreteWrapperTrain, NoiseWrapper

from DDQN import DDQN

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

parser = argparse.ArgumentParser()
parser.add_argument('--details', default='', help='used to set the weights name')
parser.add_argument('--width', default=64, help='used to set the number of neurons for each layer in a mlp network (32 or 64)')
parser.add_argument('--activation', default='relu', help='used to set the model activation function (elu or relu)')
parser.add_argument('--map-name', default='loop_empty', help='to use a different map from that used in training \
    available maps: \n \
        straight_road\n \
        4way\n \
        udem1\n \
        small_loop\n \
        small_loop_cw \n \
        zigzag_dists \n \
        loop_obstacles \n \
        loop_pedestrians \n')

args = parser.parse_args()

weights_name = "weights/ddqn_duckietown_weights" + args.details + ".h5"

# Create the environment 
env = DuckietownEnv(
    seed=123, # random seed
    map_name=args.map_name,
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

model = DDQN(env, activation=args.activation, cnn=False, mlp_width=args.width)
model.load(weights_name)

obs = env.reset()
env.render()

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    global obs 
    action = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(f'action = {action} step_count = {env.unwrapped.step_count}, reward={reward:.3f}')
    '''
    # optional information about position relative to right lane:
    print('position relative to right lane:')
    dist = info['Simulator']['lane_position']['dist']
    print(f'dist: {dist}')
    dot_dir = info['Simulator']['lane_position']['dot_dir']
    print(f'dot_dir: {dot_dir}')
    angle_deg = info['Simulator']['lane_position']['angle_deg']
    print(f'angle_deg: {angle_deg:.4f}')
    angle_rad = info['Simulator']['lane_position']['angle_rad']
    print(f'angle_rad: {angle_rad:.4f}')
    '''
    if done:
        print('done!')
        obs = env.reset()
        env.render()
    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()


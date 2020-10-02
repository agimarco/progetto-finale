import argparse
import sys
import pyglet
from pyglet.window import key
import numpy as np
import pandas as pd
import tensorflow as tf

import gym
from gym import wrappers

from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.simulator import Simulator

from project_utils import PositionObservation, DtRewardWrapper, DiscreteActionWrapper, DiscreteActionWrapperTrain, NoiseWrapper

from DDQN import DDQN

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

parser = argparse.ArgumentParser()
parser.add_argument('--details', default='', help='used to set the weights name')
parser.add_argument('--width', default=64, help='used to set the number of neurons for each layer in a mlp network (32 or 64)')
parser.add_argument('--activation', default='relu', help='used to set the model activation function (elu or relu)')
parser.add_argument('--map-name', default='loop_empty', help='to use a different map from that used in training \
    available maps: straight_road, 4way, udem1, small_loop, small_loop_cw, zigzag_dists, loop_obstacles, loop_pedestrians')
parser.add_argument('--top', default=False, help='True to see from the top')
parser.add_argument('--record', default=False, help='True to record')
parser.add_argument('--bbox', default=False, help='True to draw agent and objects bboxes')

args = parser.parse_args()

weights_name = "weights/ddqn_duckietown_weights" + args.details + ".h5"

# Create the environment 
env = DuckietownEnv(
    seed=123, # random seed
    map_name=args.map_name,
    max_steps=2000, # 500001 if we don't want the gym to reset itself
    domain_rand=0,
    accept_start_angle_deg=4, # start close to straight
    full_transparency=True,
    draw_curve=True, #args.top,
    draw_bbox=args.bbox,
    distortion=False,
    graphics=True,
    draw_traj=args.top,
    free_camera=False,#args.top,
    show_info=False,
)
# discrete actions, 4 value observation and modified reward
if args.record:
    env = wrappers.Monitor(env, 'videos/video_'+args.details, force=True, video_callable=lambda episode_id: True)
env = NoiseWrapper(env)
env = DiscreteActionWrapper(env)   # we use a different action wrapper from training since we can now stay still also
env = PositionObservation(env)
env = DtRewardWrapper(env)

model = DDQN(env, activation=args.activation, cnn=False, mlp_width=args.width)
model.load(weights_name)

obs = env.reset()
env.render()

pause = False

assert isinstance(env.unwrapped, Simulator)

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):

    global pause
    cam_offset, cam_angle = env.unwrapped.cam_offset, env.unwrapped.cam_angle
    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)
    elif symbol == key.SPACE:
        pause = not pause
    # Camera movement
    elif symbol == key.W:
        cam_angle[0] -= 3 #5
    elif symbol == key.S:
        cam_angle[0] += 3 #5
    elif symbol == key.A:
        cam_angle[1] -= 3 #5
    elif symbol == key.D:
        cam_angle[1] += 3 #5
    elif symbol == key.Q:
        cam_angle[2] -= 3 #5
    elif symbol == key.E:
        cam_angle[2] += 3 #5
    elif symbol == key.U:
        if modifiers:  # Mod+Up for height  #ALT + U
            cam_offset[1] += 1 #.1
        else:
            cam_offset[0] += 1 #.1
    elif symbol == key.J:
        if modifiers:  # Mod+Down for height #ALT + J
            cam_offset[1] -= 1 #.1
        else:
            cam_offset[0] -= 1 #.1
    elif symbol == key.H:
        cam_offset[2] -= 1 #.1
    elif symbol == key.K:
        cam_offset[2] += 1 #.1

    # Take a screenshot
    elif symbol == key.RETURN:
        print('saving screenshot')
        img = env.render('rgb_array')
        try:
            from PIL import Image
            im = Image.fromarray(img)
            im.save('screen.png')
        except BaseException as e:
            print(str(e))


def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    global obs 
    global pause
    if(not pause):
        action, q_value = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(f'action = {action} q_value = {q_value:.3f} step_count = {env.unwrapped.step_count}, reward={reward:.3f}')
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
    if not pause and done:
        print('done!')
        obs = env.reset()
        env.render()
    env.render()
    #env.render('free_cam')

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()


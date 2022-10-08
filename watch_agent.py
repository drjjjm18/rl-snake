import torch as t
from torch import tensor as T
import matplotlib.pyplot as plt
from snake.game import do
from rl_env import snake_agent
from rl_env import environment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
import numpy as np
import tensorflow as tf
from tf_agents.trajectories.time_step import TimeStep

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint_dir', default='checkpoint')
args = parser.parse_args()
checkpoint_dir = args.checkpoint_dir


env = environment.SnakeEnvironment()
env = TFPyEnvironment(env)
agent = snake_agent.SnakeAgent(env, checkpoint_dir=checkpoint_dir, restore=True)
agent.initialize()



snake = t.zeros((24, 24), dtype=t.int)
snake[0, :3] = T([1, 2, -1])

fig, ax = plt.subplots(1, 1)
img = ax.imshow(snake)

score = None
while score is None:
    img.set_data(snake)
    fig.canvas.draw_idle()
    plt.pause(0.1)
    ts=TimeStep(step_type=None,
            reward=None,
            discount=None,
            observation=tf.constant(snake.reshape(1,24,24), dtype=np.int32)
        )
    action = agent.policy.action(ts)
    action = int(action.action.numpy())
    print(action)
    score = do(snake, action)

print('Score:', score)

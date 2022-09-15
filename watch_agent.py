import torch as t
from torch import tensor as T
import matplotlib.pyplot as plt
from MiniSnakes import do
import snake_agent
import environment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
import numpy as np
import tensorflow as tf
from tf_agents.trajectories.time_step import TimeStep

env = environment.SnakeEnvironment()
env = TFPyEnvironment(env)
agent = snake_agent.SnakeAgent(env, restore=True)
agent.initialize()



snake = t.zeros((24, 24), dtype=t.int)
snake[0, :3] = T([1, 2, -1])

fig, ax = plt.subplots(1, 1)
img = ax.imshow(snake)
#action = {'val': 1}
#action_dict = {'a': 0, 'd': 2}
#action_dict.setdefault(1)




#fig.canvas.mpl_connect('key_press_event', lambda e:
                       #action.__setitem__('val', action_dict[e.key]))

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
    score = do(snake, int(action.action.numpy()))
    #action['val'] = 1

print('Score:', score)

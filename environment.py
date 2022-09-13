import numpy as np
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import torch as t
from torch import tensor as T
from numpy import unravel_index as unravel


def init_snake():
    snake = t.zeros((24, 24), dtype=t.int)
    snake[0, :3] = T([1, 2, -1])
    
    return snake


def update(snake, action):
    positions = snake.flatten().topk(2)[1]
    [pos_cur, pos_prev] = [T(unravel(x, snake.shape)) for x in positions]
    rotation = T([[0, -1], [1, 0]]).matrix_power(3 + action)
    pos_next = (pos_cur + (pos_cur - pos_prev) @ rotation)
    
    if (pos_next >= snake.shape[0]).any() or (pos_next<0).any():
        return -1, snake, (snake[tuple(pos_cur)] - 2).item()
    if (snake[tuple(pos_next)] > 0).any():
        return -1, snake, (snake[tuple(pos_cur)] - 2).item()
    if snake[tuple(pos_next)] == -1:
        reward = 1
        pos_food = (snake == 0).flatten().to(t.float).multinomial(1)[0]
        snake[unravel(pos_food, snake.shape)] = -1
    else:
        reward = 0
        snake[snake > 0] -= 1
    snake[tuple(pos_next)] = snake[tuple(pos_cur)] + 1
    
    return reward, snake, (snake[tuple(pos_cur)] - 2).item()
    

class SnakeEnvironment(PyEnvironment):

    def __init__(self):

        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec((), np.int32, minimum=0, maximum=2)
        self._observation_spec = array_spec.BoundedArraySpec((24,24), np.int32, minimum=0, maximum=576)
        self._episode_ended = False
        self._state = init_snake()
        self.steps = 0
        self.repeated_move_check = []

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        #print(f'===\n- resetting -\n===')
        self._state = init_snake()
        self._episode_ended = False
        self.steps = 0
        self.repeated_move_check = []
        self.score = 0
        return ts.restart(self._state)

    def _step(self, action):
        self.steps += 1
        
        if action in self.repeated_move_check:
            self.repeated_move_check.append(action)
        else:
            self.repeated_move_check = [action]
            
        
        if (self.steps >= 1000) or (len(self.repeated_move_check) > 25):
            print(f'max steps ({self.steps}) or repeated moves\nscore = {self.score}')
            return ts.termination(self._state, -1)

        result, self._state, self.score = update(self._state, action)

        if result == -1:
            print(f'score = {self.score}')
            return ts.termination(self._state, -1)
        
        else:
            return ts.transition(self._state, result)
        
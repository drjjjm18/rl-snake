from rl_env import environment
from rl_env import snake_agent
from tf_agents.environments.tf_py_environment import TFPyEnvironment
import argparse

from rl_env.train import train_driver

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--episodes', default=100)
parser.add_argument('-r', '--restore_checkpoint', default='False')
parser.add_argument('-c', '--checkpoint_dir', default='checkpoint')

args = parser.parse_args()
episodes = int(args.episodes)
restore = bool(args.restore_checkpoint)
checkpoint_dir = args.checkpoint_dir


env = environment.SnakeEnvironment()
env = TFPyEnvironment(env)
agent = snake_agent.SnakeAgent(env, checkpoint_dir=checkpoint_dir, restore=restore)
agent.initialize()


driver = train_driver(env, agent)


print(f'playing and training for {episodes} episodes')
results, losses = driver.run(episodes)
print(f"""
first result = {results[0]}
last result = {results[-1]}
max result = {max(results)}
""")

import pickle
with open('results.pkl', 'wb') as f:
    pickle.dump((results), f)

with open('losses.pkl', 'wb') as f:
    pickle.dump((losses), f)

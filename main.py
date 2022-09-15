import environment
import snake_agent
from tf_agents.environments.tf_py_environment import TFPyEnvironment
import argparse

from train import train_driver

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--episodes', default=100)
parser.add_argument('-r', '--restore_checkpoint', default='False')
args = parser.parse_args()
episodes = int(args.episodes)
restore = bool(args.restore_checkpoint)


env = environment.SnakeEnvironment()
env = TFPyEnvironment(env)
agent = snake_agent.SnakeAgent(env, restore=restore)
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

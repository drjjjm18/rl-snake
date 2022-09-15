import environment
import snake_agent
from tf_agents.environments.tf_py_environment import TFPyEnvironment

from train import train_driver

env = environment.SnakeEnvironment()
env = TFPyEnvironment(env)
agent = snake_agent.SnakeAgent(env)
agent.initialize()


driver = train_driver(env, agent)

results, losses = driver.run(1)
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

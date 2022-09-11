import environment
import snake_agent
from tf_agents.environments.tf_py_environment import TFPyEnvironment
import pickle
from train import train_driver

env = environment.SnakeEnvironment()
env = TFPyEnvironment(env)
agent = snake_agent.snake_agent(env)

driver = train_driver(env, agent)

results, losses = driver.run(10000)
with open('results.pkl', 'wb') as f:
    pickle.dump((results, losses), f)

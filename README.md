# A reinforcement learning environment for the classic Snake game

This repo contains a pre-built enviornment which can be used to experiment with tf-agents and reinforcement learning (RL). Here you can:
* play snake on your computer
* train an agent to play snake
* watch an agent playing snake

## Set up

First clone the repo. In a terminal, use the following commands:
```
# change directory to the folder you want to clone repo in, e.g. Documents/code shown below
cd Documents/code
git clone https://github.com/drjjjm18/rl-snake.git
cd rl-snake
```
Next make sure you have the dependencies installed. You'll need python 3, and pytorch, tensorflow and tf-agents installed. You can use requirements.txt to make sure they're installed:

**Optional** Consider creating a virtual environment:
```
# create python venv
python -m venv snake_env
# activate venv - note this varies between windows, bash and mac
snake_env\Scripts\activate # linux/bash terminal: source snake_env/Scripts/activate # mac: source snake_env/bin/activate
```
Install dependencies:
```
pip install -r requirements.txt
```
## The game

[MiniSnakes.py](https://github.com/drjjjm18/rl-snake/blob/main/MiniSnakes.py) contains the code for the basic snake game. This is based on the code developed in [How to write a game of snake in 12 lines of code]('https://github.com/eliasffyksen/MiniSnakes), tweaked to not allow the player to travel through the wall boundaries.

This can be played by running [Interface.py](https://github.com/drjjjm18/rl-snake/blob/main/Interface.py): use the 'a' and 'd' keys to control the snake by turning left and right respectively.

simply run the below in a terminal:
```python Interface.py```

## The RL environment
[enironment.py](https://github.com/drjjjm18/rl-snake/blob/main/environment.py) contains the code for a RL environment. This handles some key aspects of an RL training loop:
* initialising the game environment
* resetting the game environment once the game finishes
* handling a 'step' in the environment: taking the action from an agent, and returning the result of that action and the associated award

## The RL agent
[agent.py](https://github.com/drjjjm18/rl-snake/blob/main/snake_agent.py) contains the code for a RL agent to play the game. The default is an example of a DQN Agent, which has some key parameters:
*fc_layer_params: these are the fully connected layers of neurons in the agent's neural network - the number of neurons and number of layers. The default gives a layer of 100 neurons and a layer of 50 neurons
*learning_rate: the learning rate used during training the agent's neural network
*epsilon_greedy: the chance of the agent choosing a random action (set to 0.3 by default, 30% chance)
*checkpoint_dir: the directory the agent will save checkpoints to during and after training

The agent also has an `optimizer`, which is set to Adam, and a `checkpointer`, which is used to save and retore checkpoints saved during training.

## the RL training loop

[train.py](https://github.com/drjjjm18/rl-snake/blob/main/train.py) contains code for a train_driver class: this class handles the playing and training loop for the agent and environment. 

## Running the loop

[main.py](https://github.com/drjjjm18/rl-snake/blob/main/main.py) can be used to train an agent. It takes two optional arguments:
* -e: the number of episodes to play for
* -r: whether to restore a checkpoint from the checkpoint directory

The script can be run as follows:
```
python main.py -e 10000 -r False
```

## Watching your agent
After training an agent, [watch_agent.py](https://github.com/drjjjm18/rl-snake/blob/main/watch_agent.py) can be used to watch an agent play snake live. Simply run:
```
python watch_agent.py
```

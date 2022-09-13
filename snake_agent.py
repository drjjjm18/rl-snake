import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.specs import array_spec, tensor_spec
import numpy as np
from tf_agents.utils import common
from tf_agents.networks import q_network
from tensorflow.train import Checkpoint
import os


class SnakeAgent(dqn_agent.DqnAgent):
    
    def __init__(self,
                env,
                fc_layer_params = (100, 50),
                learning_rate = 0.001,
                **dqn_kwargs):
        
        self.q_net = q_network.QNetwork(
            env.observation_spec(),
            env.action_spec(),
            fc_layer_params = fc_layer_params
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        train_step_counter = tf.Variable(0)
        super().__init__(env.time_step_spec(),
                     env.action_spec(),
                     self.q_net,
                     self.optimizer,
                     td_errors_loss_fn=common.element_wise_squared_loss,
                     train_step_counter=train_step_counter,
                     **dqn_kwargs)#
        
    def save_checkpoint(self, episode):
        checkpoint = Checkpoint(q_net=self.q_net)
        save_dir = f'checkpoints/episode_{episode}'
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        checkpoint.save(os.path.join(save_dir, 'checkpoint'))
        
    def load_checkpoint(self, path):
        checkpoint = Checkpoint(q_net=self.q_net)
        checkpoint.restore(path)

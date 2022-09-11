from environment import SnakeEnvironment
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.specs import array_spec, tensor_spec
import numpy as np
from tf_agents.utils import common
from tf_agents.networks import q_network
#from tf_agents.networks import sequential

fc_layer_params = (100, 50)
action_tensor_spec = tensor_spec.from_spec(array_spec.BoundedArraySpec((1,), np.int32, minimum=0, maximum=2))
num_actions = 3
learning_rate = 0.001
# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
# def dense_layer(num_units):
#     return tf.keras.layers.Dense(
#       num_units,
#       activation=tf.keras.activations.relu,
#       kernel_initializer=tf.keras.initializers.VarianceScaling(
#           scale=2.0, mode='fan_in', distribution='truncated_normal'))

# q_net = q_network.QNetwork(
#             train_env.observation_spec(),
#             train_env.action_spec(),
#             fc_layer_params = fc_layer_params
#         )



train_step_counter = tf.Variable(0)

def snake_agent(env,
                fc_layer_params = (100, 50),
                learning_rate = 0.001):
    
    q_net = q_network.QNetwork(
            env.observation_spec(),
            env.action_spec(),
            fc_layer_params = fc_layer_params
        )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    return dqn_agent.DqnAgent(
    env.time_step_spec(),
    env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

# agent.initialize()
import tensorflow as tf
import numpy as np
from collections import namedtuple
from collections import deque
import time
import random
import chessLibraryHelper as clh


MEMORY_SIZE = 1_000_000   # size of memory buffer
GAMMA = 0.999             # discount factor
ALPHA = 1e-3              # learning rate  
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps
TAU = 1e-3                # Soft update parameter.

board = clh.clhBoard()
print(board.legal_moves)
print(board.fen())

#bellmans equation
#y = r + gamma * maxQ(statePrime, actionPrime; weights)
#error
#r + gamma * maxQtarget(statePrime, actionPrime; weights) - Q(state, action; weights)
#update weights
#𝑤−←𝜏𝑤+(1−𝜏)𝑤−

state_size = len(board.state())
q_network = tf.keras.Sequential([
    tf.keras.layers.Input((state_size,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(4672, activation="linear") #https://ai.stackexchange.com/questions/7979/why-does-the-policy-network-in-alphazero-work
])

target_q_network = tf.keras.Sequential([
    tf.keras.layers.Input((state_size,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(4672, activation="linear") #https://ai.stackexchange.com/questions/7979/why-does-the-policy-network-in-alphazero-work
])

optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA)

experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


def compute_loss(experiences, gamma, q_network, target_q_network):
    
    states, actions, rewards, next_states, done_vals = experiences
    
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
    y_targets = rewards + (1-done_vals)*(gamma*max_qsa)
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]), tf.cast(actions, tf.int32)], axis=1))
        
    loss = tf.keras.losses.MSE(y_targets, q_values) 
    
    return loss


@tf.function
def agent_learn(experiences, gamma):
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)
    gradients = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
    update_target_network(q_network, target_q_network)


def update_target_network(q_network, target_q_network):
    for target_weights, q_net_weights in zip(target_q_network.weights, q_network.weights):
        target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)


def get_action(q_values, epsilon=0.0):
    if random.random() > epsilon:
        return np.argmax(q_values.numpy()[0])
    else:
        return random.choice(np.arange(4))
    

start = time.time()
num_episodes = 2000
max_num_timesteps = 200
total_point_history = []
num_p_av = 100
epsilon = 1.0
memory_buffer = deque(maxlen=MEMORY_SIZE)

target_q_network.set_weights(q_network.get_weights())

state = board.state()


for i in range(num_episodes):

    board.reset()
    state = board.state()
    total_points = 0

    for t in range(max_num_timesteps):

        state_qn = np.expand_dims(state, axis=0)
        q_values = q_network(state_qn)
        action = get_action(q_values, epsilon)


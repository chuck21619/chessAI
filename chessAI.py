import tensorflow as tf
import numpy as np
from collections import namedtuple
from collections import deque
import time
import random
import chessLibraryHelper as clh
import os

MEMORY_SIZE = 100_000     # size of memory buffer
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate  
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps
TAU = 1e-3                # Soft update parameter.
MINIBATCH_SIZE = 64       # Mini-batch size.
E_DECAY = 0.995           # ε-decay rate for the ε-greedy policy.
E_MIN = 0.01              # Minimum ε value for the ε-greedy policy.

SOLVED_TOTAL_POINTS = 200

board = clh.clhBoard()

state_size = (774,)
q_network = tf.keras.Sequential([
    tf.keras.layers.Input(state_size),
    tf.keras.layers.Dense(2000, activation="relu"),
    tf.keras.layers.Dense(64*64, activation="linear") #https://ai.stackexchange.com/questions/7979/why-does-the-policy-network-in-alphazero-work
])

target_q_network = tf.keras.Sequential([
    tf.keras.layers.Input(state_size),
    tf.keras.layers.Dense(2000, activation="relu"),
    tf.keras.layers.Dense(64*64, activation="linear") #https://ai.stackexchange.com/questions/7979/why-does-the-policy-network-in-alphazero-work
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
        return random.choice(np.arange(64*64))

def check_update_conditions(t, num_steps_upd, memory_buffer):
    if (t + 1) % num_steps_upd == 0 and len(memory_buffer) > MINIBATCH_SIZE:
        return True
    else:
        return False

def get_experiences(memory_buffer):
    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
    states = tf.convert_to_tensor(np.array([e.state for e in experiences if e is not None]), dtype=tf.float32)
    actions = tf.convert_to_tensor(np.array([e.action for e in experiences if e is not None]), dtype=tf.float32)
    rewards = tf.convert_to_tensor(np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32)
    next_states = tf.convert_to_tensor(np.array([e.next_state for e in experiences if e is not None]), dtype=tf.float32)
    done_vals = tf.convert_to_tensor(np.array([e.done for e in experiences if e is not None]).astype(np.uint8), dtype=tf.float32,)
    return (states, actions, rewards, next_states, done_vals)

def get_new_eps(epsilon):
    return max(E_MIN, E_DECAY * epsilon)


start = time.time()
num_episodes = 1
max_num_timesteps = 100
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
        next_state, reward, done = board.step(action)
        memory_buffer.append(experience(state, action, reward, next_state, done))
        update = check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)

        if update:
            experiences = get_experiences(memory_buffer)
            agent_learn(experiences, GAMMA)

        state = next_state.copy()
        total_points += reward
        
        if done:
            break

    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-num_p_av:])
    epsilon = get_new_eps(epsilon)

    print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")
    if (i+1) % num_p_av == 0:
        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

    # We will consider that the environment is solved if we get an
    # average of 200 points in the last 100 episodes.
    if av_latest_points >= SOLVED_TOTAL_POINTS:
        print(f"\n\nEnvironment solved in {i+1} episodes!")
        q_network.save('lunar_lander_model.h5')
        break


q_network.save('./q_network.keras')
tot_time = time.time() - start
print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")
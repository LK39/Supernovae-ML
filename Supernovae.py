import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tf_agents.environments import py_environment, tasks
from tf_agents.specs import array_spec
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from tensorflow import keras
#### Versions are incompatible, make sure to find a compatible combination in python 3.11.9
# Define the base path
base_path = r'C:\Users\pazol\Programms\Supernovae'

# Initialize lists to hold data and labels
data = []
labels = []

# Load data from both folders for training
for supernova_type in ['Type II', 'Type IIP']:
    folder_path = os.path.join(base_path, supernova_type, 'CSV Files')

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)
                if not {'sed_freq', 'sed_flux', 'sed_eflux'}.issubset(df.columns):
                    print(f"Missing columns in {filename}, skipping.")
                    continue
                features = df[['sed_freq', 'sed_flux', 'sed_eflux']]
                data.append(features)
                labels.extend([supernova_type] * len(df))
            except Exception as e:
                print(f"Error reading {filename}: {e}")

# Concatenate all dataframes into one
data = pd.concat(data, ignore_index=True)

# Cleaning NaN values
data.dropna(inplace=True)

# Replace supernova types with numeric values
data['Supernova_Type'] = labels[:len(data)]
data['Supernova_Type'] = data['Supernova_Type'].replace({'Type II': 1, 'Type IIP': 0})

# Preprocessing
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['sed_freq', 'sed_flux', 'sed_eflux']])

# Create a custom environment for reinforcement learning
class SupernovaEnv(py_environment.PyEnvironment):
    def __init__(self, data, labels):
        super(SupernovaEnv, self).__init__()
        self._data = data
        self._labels = labels
        self._current_step = 0
        self._num_steps = len(data)
        
        # Define action and observation specs
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(data.shape[1],), dtype=np.float32, name='observation')
        self._state = self._data[self._current_step]

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._current_step = 0
        self._state = self._data[self._current_step]
        return ts.restart(self._state)

    def _step(self, action):
        if self._current_step >= self._num_steps:
            return self.reset()

        reward = 0
        if action == self._labels[self._current_step]:
            reward = 1  # Reward for correct classification
        else:
            reward = -1  # Penalty for incorrect classification

        self._current_step += 1
        if self._current_step >= self._num_steps:
            return ts.termination(self._data[self._current_step - 1], reward)
        
        self._state = self._data[self._current_step]
        return ts.transition(self._state, reward)

# Instantiate the environment
env = SupernovaEnv(data_scaled, data['Supernova_Type'].values)

# Create a Q-Network for the agent
q_net = q_network.QNetwork(
    env.observation_spec(),
    env.action_spec()
)

# Create a DQN agent
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
agent = dqn_agent.DqnAgent(
    env.time_step_spec(),
    env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    train_step_counter=tf.Variable(0)
)
agent.initialize()

# Create a replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=1,
    max_length=100000
)

# Function to collect data
def collect_episode(env, policy, num_episodes):
    for _ in range(num_episodes):
        time_step = env.reset()
        while not time_step.is_last():
            action = policy.action(time_step)
            next_time_step = env.step(action)
            traj = trajectory.from_transition(time_step, action, next_time_step)
            replay_buffer.add_batch(traj)
            time_step = next_time_step

# Training loop
num_iterations = 20000
for _ in range(num_iterations):
    collect_episode(env, agent.collect_policy, 2)

    experience, _ = replay_buffer.sample(64)
    agent.train(experience)

# Add Visualization

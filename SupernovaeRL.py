import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR

# Define the base path for your dataset
base_path = r'C:\Users\pazol\Programms\Supernovae'

# Initialize lists to hold data and labels
data_list = []
labels = []

# Load data from the specified folder
folder_path = os.path.join(base_path, 'supernova_data')

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(file_path)

            # Ensure the required columns exist and handle missing values
            if not {'Gi', 'GVhel', 'mag', 'Type'}.issubset(df.columns):
                print(f"Missing columns in {filename}, skipping.")
                continue

            # Clean data: drop rows with NaN in relevant columns
            df.dropna(subset=['Gi', 'GVhel', 'mag', 'Type'], inplace=True)

            # Select relevant features and labels
            features = df[['Gi', 'GVhel', 'mag']]
            data_list.append(features)

            # Convert Type to numerical labels
            labels.extend(df['Type'].astype('category').cat.codes)

        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Concatenate all dataframes into one
data = pd.concat(data_list, ignore_index=True)

# Ensure labels are correctly assigned
labels = pd.Series(labels)

# Preprocessing
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['Gi', 'GVhel', 'mag']])
print("Data loaded and preprocessed.")

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data_scaled, labels, test_size=0.2, random_state=42)

# Define Classification Environment
class ClassificationEnv:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels.reset_index(drop=True)  # Reset index for consistency
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self.data[self.current_step]  # Return features only

    def step(self, action):
        if self.current_step >= len(self.data):
            raise IndexError("Current step exceeds the length of the data.")
        
        true_label = self.labels[self.current_step]
        reward = 1 if (action == true_label) else 0  # Reward for correct classification
        self.current_step += 1
        done = self.current_step >= len(self.data)
        next_state = self.data[self.current_step] if not done else self.data[self.current_step - 1]
        
        return next_state, reward, done

# Define the Q-Network with Dropout and L2 Regularization
class QNetwork(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.5):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout after first layer
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)  # Apply dropout after second layer
        return self.fc3(x)

# Experience Replay
class ExperienceReplay:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Hyperparameters
input_size = 3
num_classes = len(labels.unique())
learning_rate = 0.001
num_episodes = 25000
gamma = 0.99
batch_size = 32
memory_capacity = 10000
update_frequency = 5
lambda_l2 = 0.01  # L2 regularization strength

# Initialize environment, model, and experience replay
env = ClassificationEnv(X_train, y_train)
q_network = QNetwork(input_size, num_classes)
target_network = QNetwork(input_size, num_classes)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
memory = ExperienceReplay(capacity=memory_capacity)

# Initialize action counts for UCB
action_counts = np.zeros(num_classes)  # Keep track of action counts
rewards_per_episode = []
losses_per_episode = []

# Initialize the learning rate scheduler
scheduler = StepLR(optimizer, step_size=100, gamma=0.1)  # Adjust step_size and gamma as needed

# Initialize epsilon for exploration
initial_epsilon = 1.0
final_epsilon = 0.01
epsilon_decay_episodes = 10000  # Number of episodes over which to decay epsilon
epsilon = initial_epsilon

# Initialize the learning rate scheduler
scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

# Training loop with UCB exploration and dynamic learning rate
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    episode_loss = 0

    # Update epsilon for exploration
    epsilon = max(final_epsilon, initial_epsilon - (episode / epsilon_decay_episodes) * (initial_epsilon - final_epsilon))

    while not done:
        # Calculate UCB for each action
        ucb_values = q_network(torch.FloatTensor(state)).detach().numpy()
        total_actions = np.sum(action_counts) + 1  # Avoid division by zero

        for action in range(num_classes):
            if action_counts[action] > 0:
                ucb_values[action] += np.sqrt(2 * np.log(total_actions) / action_counts[action])
            else:
                # If the action has never been taken, give it a high initial value
                ucb_values[action] += float('inf')

        # Epsilon-greedy exploration
        if np.random.rand() < epsilon:
            action = np.random.choice(num_classes)  # Explore: select a random action
        else:
            action = np.argmax(ucb_values)  # Exploit: select the best action

        next_state, reward, done = env.step(action)
        total_reward += reward

        memory.push((state, action, reward, next_state))

        # Update action count
        action_counts[action] += 1

        if len(memory) > batch_size and (episode * len(env.data) + env.current_step) % update_frequency == 0:
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = zip(*experiences)

            state_tensor = torch.FloatTensor(states)
            next_state_tensor = torch.FloatTensor(next_states)
            action_tensor = torch.LongTensor(actions)
            reward_tensor = torch.FloatTensor(rewards)

            q_values = q_network(state_tensor)
            next_q_values = target_network(next_state_tensor)
            target = reward_tensor + (gamma * torch.max(next_q_values, dim=1)[0])

            target_f = q_values.clone().detach()
            target_f[range(batch_size), action_tensor] = target

            optimizer.zero_grad()
            loss = criterion(q_values, action_tensor)  # Base loss

            # Add L2 regularization
            l2_reg = sum(param.pow(2).sum() for param in q_network.parameters())
            loss += lambda_l2 * l2_reg  # Add L2 penalty to loss

            loss.backward()
            optimizer.step()
            episode_loss += loss.item()

        state = next_state

    # Update target network every few episodes
    if episode % 10 == 0:
        target_network.load_state_dict(q_network.state_dict())

    # Step the scheduler after each episode
    scheduler.step()

    rewards_per_episode.append(total_reward)
    losses_per_episode.append(episode_loss)
    current_lr = scheduler.get_last_lr()[0]  # Get the current learning rate
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Loss: {episode_loss:.4f}, "
          f"LR: {current_lr:.6f}, Epsilon: {epsilon:.4f}")

# (Validation and plotting code goes here)
# Evaluate on validation data
val_rewards = []
for i, state in enumerate(X_val):
    with torch.no_grad():
        action = torch.argmax(q_network(torch.FloatTensor(state))).item()
        true_label = y_val.iloc[i]
        reward = 1 if (action == true_label) else 0
        val_rewards.append(reward)

accuracy = (sum(val_rewards) / len(val_rewards)) * 100

# Plotting results
plt.figure(figsize=(15, 12))

plt.subplot(3, 1, 1)
plt.plot(rewards_per_episode, label='Total Reward per Episode')
plt.title('Total Rewards Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()

average_rewards = [np.mean(rewards_per_episode[max(0, i-100):(i+1)]) for i in range(len(rewards_per_episode))]
plt.subplot(3, 1, 2)
plt.plot(average_rewards, label='Average Reward (Last 100 Episodes)', color='orange')
plt.title('Average Rewards Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(losses_per_episode, label='Loss per Episode', color='red')
plt.title('Loss Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Final Information Display
final_total_reward = sum(rewards_per_episode)
average_reward = final_total_reward / num_episodes
average_loss = np.mean(losses_per_episode)

print("\nFinal Results:")
print(f"Total Rewards over all episodes: {final_total_reward}")
print(f"Average Reward per episode: {average_reward:.2f}")
print(f"Average Loss per episode: {average_loss:.4f}")
print(f"Validation Accuracy: {accuracy:.2f}%")
print(f"Total Episodes: {num_episodes}")
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
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Define the base path for your dataset
base_path = r'C:\Users\pazol\Programms\Supernovae'

# Before the file loading loop, create a consistent mapping
label_mapping = {'Type I': 0, 'Type II': 1}

# Initialize lists to hold data and labels
data_list = []
labels = []

# Load data from the specified folder
folder_path = os.path.join(base_path, 'supernova_data')

# Modify the file loading loop
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(file_path)

            # Ensure the required columns exist and handle missing values
            if not {'Gi', 'GVhel', 'mag', 'Label'}.issubset(df.columns):
                print(f"Missing columns in {filename}, skipping.")
                continue

            # Clean data: drop rows with NaN in relevant columns
            df.dropna(subset=['Gi', 'GVhel', 'mag', 'Label'], inplace=True)

            # Select relevant features and labels
            features = df[['Gi', 'GVhel', 'mag']]
            data_list.append(features)

            # Convert labels using the consistent mapping
            labels.extend([label_mapping[label] for label in df['Label']])

        except Exception as e:
            print(f"Error reading {filename}: {e}")

# After loading the data
unique_counts = pd.Series(labels).value_counts()
print("\nClass distribution:")
print(unique_counts)
print("\nClass percentages:")
print(unique_counts / len(labels) * 100)

# Concatenate all dataframes into one
data = pd.concat(data_list, ignore_index=True)

# Ensure labels are correctly assigned
labels = pd.Series(labels)

# Preprocessing
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['Gi', 'GVhel', 'mag']])
print("Data loaded and preprocessed.")

# Split data into training, validation, and test sets (70/15/15)
X_temp, X_test, y_temp, y_test = train_test_split(data_scaled, labels, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)  # 15% of the original data


print("Feature columns:", data.columns)
print("Sample of scaled data:", data_scaled[:5])
print("Unique labels:", labels.value_counts())



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
        
        # Add some exploration penalty
        reward = 1.0 if action == true_label else -1.0
        
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
    
    
class DuelingQNetwork(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.5):
        super(DuelingQNetwork, self).__init__()
        self.num_classes = num_classes  # Store num_classes
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Value stream
        self.value_stream = nn.Linear(128, 1)
        # Advantage stream
        self.advantage_stream = nn.Linear(128, num_classes)

    def forward(self, x):
        # Ensure input is 2D
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Ensure proper broadcasting
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
    
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
    

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        predictions = []
        for i in range(len(X)):
            state = X[i]
            output = model(torch.FloatTensor(state))
            pred = torch.argmax(output).item()
            predictions.append(pred)
            if pred == y.iloc[i]:
                correct += 1
            total += 1
    return (correct / total) * 100, predictions

# Hyperparameters
input_size = 3
num_classes = len(labels.unique())
learning_rate = 0.0001
num_episodes = 3000
gamma = 0.99
batch_size = 64
memory_capacity = 10000
update_frequency = 5
lambda_l2 = 1e-4  # L2 regularization strength

# Initialize environment, model, and experience replay
env = ClassificationEnv(X_train, y_train)
q_network = DuelingQNetwork(input_size, num_classes)
target_network = DuelingQNetwork(input_size, num_classes)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
criterion = nn.SmoothL1Loss()  # Use SmoothL1loss for Dueling architecture
memory = ExperienceReplay(capacity=memory_capacity)

# Initialize action counts for UCB
action_counts = np.zeros(num_classes)  # Keep track of action counts
rewards_per_episode = []
losses_per_episode = []
val_accuracies = []
train_accuracies = []
learning_rates = []
epsilons = []

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=100, gamma=0.1)  # Adjust step_size and gamma as needed

# Initialize epsilon for exploration
initial_epsilon = 1.0
final_epsilon = 0.01
epsilon_decay_episodes = 1000  # Number of episodes over which to decay epsilon
epsilon = initial_epsilon


# After initializing the environment and before training
print("\nEnvironment setup verification:")
print("Number of actions:", num_classes)
print("Input state shape:", X_train[0].shape)
print("Action space size:", len(action_counts))


# Test a single forward pass
test_state = torch.FloatTensor(X_train[0])
test_output = q_network(test_state)
print("Network output shape:", test_output.shape)
print("Network output:", test_output.detach().numpy())

# Training loop with UCB exploration and dynamic learning rate
for episode in range(num_episodes):
    q_network.train()  # Set model to training mode
    state = env.reset()
    done = False
    total_reward = 0
    episode_loss = 0

    # Update epsilon for exploration
    epsilon = max(final_epsilon, initial_epsilon - (episode / epsilon_decay_episodes) * (initial_epsilon - final_epsilon))

    # Shuffle data at the beginning of each episode for mixing
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train.iloc[indices].reset_index(drop=True)

    # Modified training loop section
    for i in range(len(X_train_shuffled)):
        state = X_train_shuffled[i]
        
        # Get Q-values from network
        state_tensor = torch.FloatTensor(state)
        q_values = q_network(state_tensor).detach().numpy().flatten()  # Ensure 1D array
        
        # Calculate UCB
        total_actions = np.sum(action_counts) + 1
        ucb_values = np.copy(q_values)
        
        for a in range(num_classes):
            if action_counts[a] > 0:
                ucb_values[a] += np.sqrt(2 * np.log(total_actions) / action_counts[a])
            else:
                ucb_values[a] = float('inf')
        
        # Epsilon-greedy exploration
        if np.random.rand() < epsilon:
            action = np.random.choice(num_classes)
        else:
            action = np.argmax(ucb_values)

        next_state, reward, done = env.step(action)
        total_reward += reward

        memory.push((state, action, reward, next_state))
        action_counts[action] += 1

        # Training step modification
        if len(memory) > batch_size and (episode * len(env.data) + i) % update_frequency == 0:
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = zip(*experiences)

            state_tensor = torch.FloatTensor(states)
            next_state_tensor = torch.FloatTensor(next_states)
            action_tensor = torch.LongTensor(actions)
            reward_tensor = torch.FloatTensor(rewards)

            # Get current Q values
            current_q_values = q_network(state_tensor)
            current_q_values = current_q_values.gather(1, action_tensor.unsqueeze(1))
            

            # Get next Q values
            with torch.no_grad():
                next_q_values = target_network(next_state_tensor)
                max_next_q_values = next_q_values.max(1)[0]
                expected_q_values = reward_tensor + (gamma * max_next_q_values)
                expected_q_values = expected_q_values.unsqueeze(1)
            
            # Compute loss
            optimizer.zero_grad()
            loss = criterion(current_q_values, expected_q_values)
            
            # Add L2 regularization
            l2_reg = sum(param.pow(2).sum() for param in q_network.parameters())
            loss += lambda_l2 * l2_reg

            loss.backward()
            optimizer.step()
            episode_loss += loss.item()
        state = next_state

    # Update target network every few episodes
    if episode % 10 == 0:
        target_network.load_state_dict(q_network.state_dict())

    # Step the scheduler after each episode
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    # Evaluate model on training and validation sets
    q_network.eval()  # Set model to evaluation mode
    train_accuracy, train_preds = evaluate_model(q_network, X_train, y_train)
    val_accuracy, val_preds = evaluate_model(q_network, X_val, y_val)
    q_network.train()  # Set model back to training mode

    # Store metrics
    rewards_per_episode.append(total_reward)
    losses_per_episode.append(episode_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    learning_rates.append(current_lr)
    epsilons.append(epsilon)

    # Visualizations (every 1000 episodes or at the end)
    if (episode + 1) % 1000 == 0 or episode == num_episodes - 1:
        plt.figure(figsize=(20, 15))
        
        # Plot 1: Training and Validation Accuracy
        plt.subplot(3, 2, 1)
        plt.plot(train_accuracies, label='Training Accuracy', color='blue')
        plt.plot(val_accuracies, label='Validation Accuracy', color='red')
        plt.title('Model Accuracy over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)

        # Plot 2: Loss Function
        plt.subplot(3, 2, 2)
        plt.plot(losses_per_episode, label='Training Loss', color='orange')
        plt.title('Loss over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Rewards over Time
        plt.subplot(3, 2, 3)
        plt.plot(rewards_per_episode, label='Episode Reward', color='green')
        plt.title('Rewards over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True)

        # Plot 4: Average Reward (Moving Window)
        window_size = 10
        moving_avg = np.convolve(rewards_per_episode, 
                                np.ones(window_size)/window_size, 
                                mode='valid')
        plt.subplot(3, 2, 4)
        plt.plot(moving_avg, label=f'Moving Average (window={window_size})', 
                 color='purple')
        plt.title('Average Reward over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True)

        # Plot 5: Learning Rate and Epsilon
        plt.subplot(3, 2, 5)
        plt.plot(learning_rates, label='Learning Rate', color='brown')
        plt.plot(epsilons, label='Epsilon', color='pink')
        plt.title('Learning Rate and Epsilon over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        # Plot 6: Training vs Validation Accuracy Difference
        plt.subplot(3, 2, 6)
        accuracy_diff = np.array(train_accuracies) - np.array(val_accuracies)
        plt.plot(accuracy_diff, label='Train-Val Accuracy Gap', color='red')
        plt.axhline(y=0, color='k', linestyle='--')
        plt.title('Training-Validation Accuracy Gap')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy Difference (%)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Print current statistics
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print(f"Current Training Accuracy: {train_accuracy:.2f}%")
        print(f"Current Validation Accuracy: {val_accuracy:.2f}%")
        print(f"Current Loss: {episode_loss:.4f}")
        print(f"Current Reward: {total_reward}")
        print(f"Average Reward (last {window_size} episodes): {moving_avg[-1]:.2f}")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"Epsilon: {epsilon:.4f}")
        print("-" * 50)

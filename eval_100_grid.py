import os
import csv
import numpy as np
import gym
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm


# — argparse as before —
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--eps_start", type=float, required=True)
parser.add_argument("--eps_end", type=float, required=True)
parser.add_argument("--seed", type=float, required=True)
args = parser.parse_args()

model_path = args.model_path
eps_start  = args.eps_start
eps_end    = args.eps_end
seed    = args.seed

# — QNetwork definition, env setup, loading model, evaluation —  
class QNetwork(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.fc1 = nn.Linear(state_shape, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_shape)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

device = "mps"
env    = gym.make("CartPole-v1")
state_space  = env.observation_space.shape[0]
action_space = env.action_space.n

model = QNetwork(state_space, action_space).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


total_rewards = []

# Running for 100 episodes with tqdm for progress tracking
for episode in range(100):
    state = env.reset()[0]  # for gymnasium; use `state = env.reset()` for older gym
    done = False
    total_reward = 0
    
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
        action = torch.argmax(q_values, dim=1).item()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    
    total_rewards.append(total_reward)

mean_reward     = np.mean(total_rewards)
std_reward      = np.std(total_rewards)

# ——— write to CSV —————————————————————————————————————————————
CSV = "results.csv"
header = ["eps_start", "eps_end", "mean_reward", "std_reward","seed"]

# if file doesn’t exist, write header
if not os.path.isfile(CSV):
    with open(CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

# append this run’s metrics
with open(CSV, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([eps_start, eps_end, mean_reward, std_reward, seed])

print(f"Appended to {CSV}: start={eps_start}, end={eps_end}, mean={mean_reward:.2f}, std={std_reward:.2f}, seed={seed}")

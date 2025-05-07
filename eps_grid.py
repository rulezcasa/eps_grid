import numpy as np
import gym
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import cv2
import yaml
from datetime import timedelta
import pdb
import xxhash
DEBUG = True
if DEBUG:
    import wandb
    from wandb import AlertLevel
import random
import subprocess
from itertools import product
import multiprocessing
from collections import deque

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# NOTES:
# 1. Hypothesis: lr is affecting divergence of td-error.
#    Test: change with various lr (both up & down)
#    Pred: with lr decreasing, TD error should converge. 
#    Outp: False, It didn't happen. Badly, it exploded.
#           There is something to LR. I see final_lr, what is it? - Not used at all
#           Something smelly in way attention is working(guess)
#
# Normalization with changing tau didn't help
# Having only exploratory policy (0.5>0.4) is not good
# Having combination of exploratory and exploitory (0.5>0.4 and 0.3<0.4 switch at 10k) was giving consistently good results (some instablity)
# Now, how to control the above through attention.
# 1. First see how TD error varies in scale with above exploratory & exploritary comb.
# A: explore: TD grows; exploit: TD stabilizes, even comes down. 
# state and state-action, What a loss. Should have pointed it out earlier.
# exponential moving average.
# Again, state-action step based exploration and exploitation works very well.
# now derive signal from attention.
# proved tgain that switch at 10k, even 1000 is good enough, but we have to dervie this switch signal from attention. how?
# if it works, next target would be tau decay
# tau discrete scaling didn't help. The signal is not helping at all. Find the alpha

# Initilizations
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

if config['AT-DQN']['device']=='mps':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if config['AT-DQN']['device']=='cuda':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
if config['AT-DQN']['device']=='cpu':
    device = torch.device("cpu")

#defining the Q-network
# TODO: 128 Needed?
class QNetwork(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(*state_shape, 64)  # cartpole input : (4,)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_shape)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    
# self.position is starting 0
# add - add till capacity in circular way
# sample - sample without replacement
class ReplayBuffer:
    def __init__(
        self, capacity, device=device
    ):  
        self.capacity = capacity
        self.device = device
        self.device_cpu = "cpu"
        self.position = 0  # used to track the current index in the buffer.
        self.size = 0  # used to track the moving size of the buffer.

        self.states = torch.zeros(
            (capacity, 4), dtype=torch.float32, device=self.device_cpu
        )  # dimension change from 4,84,84 to 4 (cartpole)
        self.actions = torch.zeros(
            (capacity, 1), dtype=torch.long, device=self.device_cpu
        )
        self.rewards = torch.zeros(
            (capacity, 1), dtype=torch.float32, device=self.device_cpu
        )
        self.next_states = torch.zeros(
            (capacity, 4), dtype=torch.float32, device=self.device_cpu
        )  # dimension change
        self.dones = torch.zeros(
            (capacity, 1), dtype=torch.float32, device=self.device_cpu
        )


    #optimization - pinned memory for faster transfers
        # self.states = self.states.pin_memory()
        # self.actions = self.actions.pin_memory()
        # self.rewards = self.rewards.pin_memory()
        # self.next_states = self.next_states.pin_memory()
        # self.dones = self.dones.pin_memory()

    def add(
        self, state, action, reward, next_state, done
    ):  # add experince to the current position of buffer
        self.states[self.position] = torch.tensor(state, dtype=torch.float32)
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = torch.tensor(next_state, dtype=torch.float32)
        self.dones[self.position] = done    

        self.position = (
            self.position + 1
        ) % self.capacity  # increment position index (circular buffer)
        self.size = min(self.size + 1, self.capacity)  # increment size

    def sample(self, batch_size):
        indices = np.random.choice(
            self.size, batch_size, replace=False
        )  # sample experiences randomly

        # single operation
        return (
            self.states[indices].to(self.device),
            self.actions[indices].to(self.device),
            self.rewards[indices].to(self.device),
            self.next_states[indices].to(self.device),
            self.dones[indices].to(self.device),
        )

# Fixed capacity LRU, attention_values intialized to 0.9, curent_index goes from 0
# hash_to_index dict, last access, global access counter

# two orthogonal returns.
class StateActionAttentionTrackerLRU:
    def __init__(self, capacity, device, action_space):
        self.device=device
        self.capacity=capacity
        self.attention_values=(torch.ones(capacity, dtype=torch.float32, device=device))*config["AT-DQN"]["initial_attention"]
        self.current_index=0
        self.hash_to_index={} #dicitonary holding a state-action-attention pairs
        self.last_access = torch.zeros(capacity, dtype=torch.long, device=device) #pytorch tensor that stores the access time of each state (stored as index corresponding to hash_to_index)
        self.access_counter = 0  #global counter tracking when which state is accessed
        self.unique_state_counter=0
        self.old_state_counter=0
        self.old_action=0
        self.new_action=0
        self.action_space = action_space

    # is this going to cause collisions?
    # not tested and some proof
    def get_state_action_hash(self, state, action):
        # 1) Make sure we have a float32 tensor on GPU or CPU
        t = state if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32)
        # 2) Cast to FP8 (E4M3)
        fp8 = t.to(torch.float8_e4m3fn)
        # 3) View the raw 8-bit bytes and hash
        raw_bytes = fp8.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        
        action_bytes = np.array([action], dtype=np.int32).tobytes()
        combined_bytes = raw_bytes + action_bytes
        
        return xxhash.xxh3_64(combined_bytes).hexdigest()

    # What are we doing till now? How did I not catch it
    def get_state_action_index(self, state, action, act=False):
        state_action_hash = self.get_state_action_hash(state, action)
        self.access_counter += 1

        if state_action_hash in self.hash_to_index: #if state-action pair already exists
            idx = self.hash_to_index[state_action_hash]
            self.last_access[idx] = self.access_counter
            if act == False:
                self.old_state_counter += 1
            return idx
    
        if act == False:
            if self.current_index < self.capacity: #if capacity not full and state-action doesn't exist
                idx = self.current_index
                self.hash_to_index[state_action_hash] = idx #map that index to the new state-action hash
                self.last_access[idx] = self.access_counter #update last access of that index
                self.current_index += 1   #return the current_index
                self.unique_state_counter += 1
                return idx
        
            else:  #if LRU full
                used_indices = torch.arange(self.current_index, device=self.device) #indices already used in LRU aranged from 0 to current_index
                idx = used_indices[
                    torch.argmin(self.last_access[: self.current_index]) #Finds the index with minimum last access value
                ].item()
                old_hash = None
                for h, i in list(self.hash_to_index.items()):
                    if i == idx:                #find the hash of state-action corresponding to least used index
                        old_hash = h
                        break
                if old_hash:
                    del self.hash_to_index[old_hash] #delete the old hashed state-action

            self.hash_to_index[state_action_hash] = idx #new state-action's hash is assigned to that index
            self.last_access[idx] = self.access_counter #last access to current access counter
            self.unique_state_counter += 1

            return idx  # returns index after removing and adding new
        
        else:
            return torch.tensor(config["AT-DQN"]["initial_attention"])
    
    def batch_get_indices(self, states, actions):  # vector index retrieval
        return torch.tensor(
            [self.get_state_action_index(state, action.item()) for state, action in zip(states, actions)],
            device=self.device,
            dtype=torch.long,
        )
    
    def batch_get_attention(self, states, actions):  # vector get attention values for the retrieved indices
        indices = self.batch_get_indices(states, actions)
        return self.attention_values[indices]
    
     #debugging code :
    def check_old_new(self, state, action):
        state_action_hash = self.get_state_action_hash(state, action)
        if state_action_hash in self.hash_to_index:
            return True
        else:
            return False
    
    def get_attention(self, state, action):  # get attention value for a single state-action index (used to act)
        exists = self.check_old_new(state, action)
        idx = self.get_state_action_index(state, action, act=True)
        if exists == True:
            self.old_action += 1
            #print("old state-action", self.attention_values[idx].item())
            return self.attention_values[idx]
        else:
            self.new_action += 1
            #print("new state-action", idx.item())
            return idx #this is the initial attention value directly returned by the get_state_action_index function
    
    # def update_attention(self, states, actions, weights):
    #     # Convert to tensor if needed - validate
    #     if not isinstance(weights, torch.Tensor):
    #         weights = torch.tensor(
    #             weights, dtype=torch.float32, device=self.device
    #         )
    #     indices = self.batch_get_indices(states, actions) #retrieve indices of state-action pairs to be updated
    #     #print("old values",self.attention_values[indices])
    #     self.attention_values[indices] = weights
    #     #print("new values",self.attention_values[indices])
    #     #self.normalize_attention()

    # EMA to rescue the agressiveness and smoothness
    def update_attention(self, states, actions, td_errors):
        if not isinstance(td_errors, torch.Tensor):
            td_errors = torch.tensor(td_errors, dtype=torch.float32, device=self.device)
        
        decay_rate = config.get("AT-DQN", {}).get("decay_rate", 0.90)
        indices = self.batch_get_indices(states, actions)
        current_values = self.attention_values[indices]
        
        updated_values = (1 - decay_rate) * current_values + decay_rate * td_errors.abs().squeeze()
        self.attention_values[indices] = updated_values
        
    def normalize_attention(self):
        if self.current_index == 0: #if empty, do nothing
            return
        used_values = self.attention_values[: self.current_index] #select only available values
        min_val = used_values.min() #compute min value amongst available
        max_val = used_values.max() #compute max value amongst available

        if min_val == max_val: #if min value and max value are same, do nothing
            return
        used_values.sub_(min_val).div_(max_val - min_val) #other min mas normalize
    
    # def compute_attention(self, td_errors):
    #    weights = torch.where(td_errors > 0.4, 0.5, 0.3)
    #    return weights
         
    # disaster in code? shooting up
    def compute_attention(self, td_errors):
        raw_values=td_errors.abs() + 1e-6
        # weights = 2.0 / (1.0 + torch.exp(-0.1 * raw_values)) - 1.0  # Maps to [0,1]
        # return weights
        return raw_values
    
    def to(self, device):
        self.device = device
        self.attention_values = self.attention_values.to(device)
        self.last_access = self.last_access.to(device)
        return self

    
class Agent:
    def __init__(self, state_space, action_space, lr, eps_start, eps_end):
        self.state_space = state_space
        self.action_space = action_space
        self.q_network = QNetwork(state_space, action_space).to(device)
        self.target_network = QNetwork(state_space, action_space).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.lr = lr
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(config['AT-DQN']['replay_buffer'], device=device)
        self.gamma = config['AT-DQN']['gamma']
        self.exploration_count = 0
        self.exploitation_count = 0
        self.check_replay_size = config['AT-DQN']['warmup'] #warmup steps
        self.step_count = 0
        self.attention_tracker = StateActionAttentionTrackerLRU(config["AT-DQN"]["LRU"], device, action_space)
        self.tau = config["AT-DQN"]["tau"]
        self.epsilon_start=eps_start
        self.epsilon_end=eps_end

        # god help me beating epsilon greedy
        self.epsilon_decay_steps = config['AT-DQN']['epsilon_decay_steps']
        self.current_epsilon = self.epsilon_start

    def update_epsilon(self):
        reduction = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
        self.current_epsilon = max(self.epsilon_end, self.current_epsilon - reduction)
        
        
    def act(self, state):
        self.update_epsilon()
        if np.random.rand() < self.current_epsilon:
            self.exploration_count += 1
            chosen_action = np.random.randint(self.action_space)
            attention_val = self.attention_tracker.get_attention(state, chosen_action).item() 
            return chosen_action, attention_val

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state_tensor)
        self.q_network.train()
        
        # New logic based state-action paris
        # Get Q-values for all actions
        all_actions = torch.arange(self.action_space, device=device)
        attention_values = torch.zeros(self.action_space, device=device)
        
        # Get attention values for each action
        for a in range(self.action_space):
            attention_values[a] = self.attention_tracker.get_attention(state, a).item()
        
        # Find actions with attention below threshold
        exploitation_mask = attention_values <= self.tau
        
        # If we have actions to exploit, choose the one with highest Q-value
        if exploitation_mask.any():
            self.exploitation_count += 1
            # Among actions with attention <= tau, choose the one with highest Q-value
            exploitable_actions = torch.where(exploitation_mask)[0]
            exploitable_q_values = action_values[0, exploitable_actions]
            best_idx = torch.argmax(exploitable_q_values).item()
            chosen_action = exploitable_actions[best_idx].item()
            return chosen_action, attention_values[chosen_action].item()
        else:
            # All actions have high attention (explore), so choose randomly
            self.exploration_count += 1
            # didn't help at all? almost bringing to halt
            # if np.random.rand() < 0.1:
            #     chosen_action = np.random.randint(self.action_space)
            # else:
            chosen_action = torch.argmax(attention_values).item()
            # chosen_action = np.random.randint(self.action_space)
            # chosen_action = torch.argmax(attention_values).item() 
            return chosen_action, attention_values[chosen_action].item()

            
    def train_step(self):
        if self.replay_buffer.size < self.check_replay_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(128)
        
        # optimization - gpu
        with torch.no_grad():
            target_q_values = self.target_network(next_states)
            max_next_q = target_q_values.max(dim=1, keepdim=True)[0]
            targets = rewards + (1 - dones) * self.gamma * max_next_q

        q_values = self.q_network(states).gather(1, actions.long())
        td_errors = targets - q_values

        # for error in td_errors:
        #     if error.item()>30:
        #         print("LARGE!", error.item())
        
        attention_values = self.attention_tracker.compute_attention(td_errors)
        self.attention_tracker.update_attention(states, actions, attention_values.squeeze())

        loss_fn = torch.nn.HuberLoss()
        loss = loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        
        # add grad clip
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        self.step_count += 1
        if self.step_count % config['AT-DQN']['target_update'] == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        return (
            loss.item(),
            td_errors.abs().squeeze().cpu().tolist(),
            q_values.squeeze().cpu().tolist(),
        )
        
    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def return_state_count(self):
    	return self.attention_tracker.unique_state_counter, self.attention_tracker.old_state_counter

    def return_action_count(self):
        return self.attention_tracker.new_action, self.attention_tracker.old_action

        
def set_seed(env: gym.Env, seed: int):
    # 1) Python RNGs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 2) Gym spaces
    env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)  


def train_agent(env_name, seed, eps_start, eps_end, render=False):
    total_steps = config['AT-DQN']['T']
    lr = config['AT-DQN']['lr']

    
    env = gym.make(env_name)
    set_seed(env, seed)
    state, _ = env.reset(seed=seed)
    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    agent = Agent(state_shape, action_size, lr, eps_start, eps_end)

    
    for step in range(total_steps):
        action, att = agent.act(state)
        next_frame, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.add_experience(state, action, reward, next_frame, done)
        result = agent.train_step()
        state = next_frame

        if done:
            state, _ = env.reset()
        

    print("Training complete!")
    env.close()


    os.makedirs("AT_DQN_Models", exist_ok=True)
    torch.save(
        agent.q_network.state_dict(), f"AT_DQN_Models/{eps_start}_{eps_end}_{seed}.pth"
    )
    print(f"Model saved successfully!")
    
    subprocess.run(["python", "eval_100_grid.py", "--model_path", f"AT_DQN_Models/{eps_start}_{eps_end}_{seed}.pth", "--eps_start", str(eps_start), "--eps_end", str(eps_end), "--seed",str(seed)])

def train_wrapper(args):
    print(f"Training started, PID: {os.getpid()} with args: {args}")
    return train_agent(*args)

if __name__ == "__main__":
    random.seed(42)
    seeds = random.sample(range(1_000_000), config["AT-DQN"]["seed_count"])
    env_name = "CartPole-v1"
    eps_start_list = [1, 0.9, 0.8, 0.7, 0.6]
    eps_end_list = [0.01, 0.05, 0.1, 0.2, 0.3]
    grid = list(product(eps_start_list, eps_end_list))
    all_args=[]
    batch_count=0

    for seed in seeds:
        for eps_start, eps_end in grid:
            all_args.append((env_name, seed, eps_start, eps_end))


    # Train agent with tqdm progress bar
    with multiprocessing.Pool(processes=config["AT-DQN"]["processes"]) as pool:
        print(f'Training Batch {batch_count}')
        batch_count += 1
        for _ in tqdm(pool.imap_unordered(train_wrapper, all_args), total=len(all_args)):
            pass
    

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Policy(nn.Module):
    def __init__(self, hidden_nodes=128, num_layers=2, learning_rate=0.0002):
        super(Policy, self).__init__()
        self.data = []
        layers = [nn.Linear(4, hidden_nodes), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_nodes, hidden_nodes))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_nodes, 2))  # Output layer
        self.network = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.network(x)
        return F.softmax(x, dim=0)

    def put_data(self, item):
        self.data.append(item)

    def train_net(self, gamma):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []

def run_experiment(env, num_episodes, hidden_nodes, num_layers, gamma, learning_rate):
    pi = Policy(hidden_nodes, num_layers, learning_rate).to(device)
    rewards = []
    avg_rewards = []
    print_interval = 20

    for n_epi in range(num_episodes):
        s, _ = env.reset()
        s = torch.tensor(s, dtype=torch.float32).to(device)
        done = False
        episode_reward = 0

        while not done:
            prob = pi(s)
            if not torch.all(prob >= 0).item() or not torch.isclose(prob.sum(), torch.tensor(1.0)).item():
                raise ValueError(f"Invalid probability tensor: {prob}")
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, _, _ = env.step(a.item())
            pi.put_data((r, prob[a]))
            s = torch.tensor(s_prime, dtype=torch.float32).to(device)
            episode_reward += r

        pi.train_net(gamma)
        rewards.append(episode_reward)

        if n_epi % print_interval == 0 and n_epi != 0:
            avg_reward = sum(rewards[-print_interval:]) / print_interval
            avg_rewards.append(avg_reward)

    torch.cuda.empty_cache()  # Free up GPU memory after the trial
    return avg_rewards

def run_multiple_trials(env, num_trials, num_episodes, **params):
    all_trials_avg_rewards = []

    for _ in range(num_trials):
        avg_rewards = run_experiment(env, num_episodes, **params)
        all_trials_avg_rewards.append(avg_rewards)

    avg_across_trials = np.mean(np.array(all_trials_avg_rewards), axis=0)
    return avg_across_trials

def plot_results(hyperparam_name, param_values, all_avg_rewards, print_interval):
    plt.figure()

    for i, avg_rewards in enumerate(all_avg_rewards):
        plt.plot(range(print_interval, len(avg_rewards) * print_interval + 1, print_interval),
                 avg_rewards, label=f'{hyperparam_name}={param_values[i]}')

    plt.xlabel(f'Episodes (x{print_interval})')
    plt.ylabel('Average Reward')
    plt.title(f'Effect of {hyperparam_name} on Performance')
    plt.legend()
    plt.show()

def experiment_varying_hyperparameter(env, num_trials, num_episodes, hyperparam_name, param_values, **fixed_params):
    all_avg_rewards = []

    for param_value in param_values:
        if hyperparam_name == 'learning_rate':
            fixed_params['learning_rate'] = param_value

        avg_rewards = run_multiple_trials(env, num_trials, num_episodes, **fixed_params)
        all_avg_rewards.append(avg_rewards)

    plot_results(hyperparam_name, param_values, all_avg_rewards, print_interval=20)

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    num_episodes = 1000
    num_trials = 10  # Adjust number of trials
    learning_rates = [0.00005, 0.0001, 0.0002, 0.0005, 0.001]

    # Focus only on learning rate, other parameters are fixed
    experiment_varying_hyperparameter(env, num_trials, num_episodes, 'learning_rate', learning_rates,
                                      hidden_nodes=128, num_layers=2, gamma=0.98)

    env.close()

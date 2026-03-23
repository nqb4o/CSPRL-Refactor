import os
import sys
import torch
import numpy as np
import random
import argparse

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Now we can import from project root
from custom_environment.StationPlacementEnv import StationPlacement
from algorithms.ga.ga_utils import GAPolicy, flatten_weights, unflatten_weights, crossover, mutate

def evaluate_agent(policy, env, seed):
    obs, _ = env.reset(seed=seed)
    total_reward = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = policy.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
    return total_reward, env.best_score

def train_ga(args):
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Instantiate the env
    location = args.location
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "custom_environment", "data")
    graph_file = os.path.join(base_dir, "Graph", location, location + ".graphml")
    node_file = os.path.join(base_dir, "Graph", location, "nodes_extended_" + location + ".txt")
    plan_file = os.path.join(base_dir, "Graph", location, "existingplan_" + location + ".pkl")

    env = StationPlacement(graph_file, node_file, plan_file, location=location)
    
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    hidden_dim = 256

    # Population initialization
    population = []
    for _ in range(args.pop_size):
        model = GAPolicy(input_dim, output_dim, hidden_dim)
        population.append(flatten_weights(model))

    best_overall_reward = -np.inf
    best_overall_chromosome = None

    log_dir = f"Results/ga/{location}/"
    os.makedirs(log_dir, exist_ok=True)

    for gen in range(args.generations):
        fitness_rewards = []
        fitness_scores = []
        
        # Evaluation step
        for i, chromosome in enumerate(population):
            print(f"  Evaluating agent {i+1}/{args.pop_size}...", end="\r")
            model = GAPolicy(input_dim, output_dim, hidden_dim)
            unflatten_weights(model, chromosome)
            # Use fixed seed for evaluation to reduce variance within generation
            reward, final_score = evaluate_agent(model, env, args.seed + gen)
            fitness_rewards.append(reward)
            fitness_scores.append(final_score)
        print() # New line after generation evaluation

        fitness_rewards = np.array(fitness_rewards)
        fitness_scores = np.array(fitness_scores)
        
        # Sort by reward (or score)
        idx = np.argsort(fitness_rewards)[::-1]
        population = [population[i] for i in idx]
        
        current_best_reward = fitness_rewards[idx[0]]
        current_best_score = fitness_scores[idx[0]]

        if current_best_reward > best_overall_reward:
            best_overall_reward = current_best_reward
            best_overall_chromosome = population[0]
            # Save the best model
            torch.save(best_overall_chromosome, os.path.join(log_dir, f"best_ga_model_{location}.pt"))
            print(f"Gen {gen}: New best reward: {best_overall_reward:.2f}")

        print(f"Gen {gen}: Best Reward: {current_best_reward:.2f}, Best Score: {current_best_score:.2f}, Avg Reward: {np.mean(fitness_rewards):.2f}")

        # Selection (Elitism)
        new_population = population[:args.elitism]
        
        # Produce offspring
        while len(new_population) < args.pop_size:
            p1, p2 = random.sample(population[:args.pop_size // 2], 2)
            c1, c2 = crossover(p1, p2, rate=0.5)
            new_population.append(mutate(c1, rate=args.mutation_rate, sigma=args.mutation_sigma))
            if len(new_population) < args.pop_size:
                new_population.append(mutate(c2, rate=args.mutation_rate, sigma=args.mutation_sigma))
        
        population = new_population

    print(f"GA Training finished. Best global score: {best_overall_score:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, default="DongDa")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pop_size", type=int, default=50)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--elitism", type=int, default=5)
    parser.add_argument("--mutation_rate", type=float, default=0.05)
    parser.add_argument("--mutation_sigma", type=float, default=0.1)
    
    args = parser.parse_args()
    train_ga(args)

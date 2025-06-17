# graphon_model.py

import numpy as np
import random
from decimal import Decimal, getcontext
import sys
import os

# Set Decimal precision globally for calculations
getcontext().prec = 50

# Ensure SageMath is installed and accessible.
# If running this script directly without 'sage -python', SageMath imports might fail.
try:
    from sage.all import Graph, DiGraph, graphs, show
except ImportError:
    print("SageMath not found. Please ensure SageMath is installed and accessible.")
    print("Some functionality will be limited.")

# Add the project root to the system path to import count_homo
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from count_homo.counthomo import WeightedGraph, count_homomorphisms

def adjacency_to_weighted_graph(adj_matrix: np.ndarray) -> WeightedGraph:
    """
    Converts an adjacency matrix to a WeightedGraph object.
    """
    N = adj_matrix.shape[0]
    edge_weights = {}
    
    for i in range(N):
        edge_weights[i] = {}
        for j in range(N):
            # Include all edges, even with small weights
            if adj_matrix[i,j] > 1e-10:  # Use small threshold instead of 0
                edge_weights[i][j] = Decimal(str(adj_matrix[i,j]))
    
    # Default node weights to 1.0
    node_weights = {v: Decimal('1.0') for v in range(N)}
    G = WeightedGraph(edge_weights=edge_weights, node_weights=node_weights)
    return G

# Define H as K5,5 \ C10
H_adj = [[0,0,0,0,0,0,1,1,1,0],
         [0,0,0,0,0,0,0,1,1,1],
         [0,0,0,0,0,1,0,0,1,1],
         [0,0,0,0,0,1,1,0,0,1],
         [0,0,0,0,0,1,1,1,0,0],
         [0,0,1,1,1,0,0,0,0,0],
         [1,0,0,1,1,0,0,0,0,0],
         [1,1,0,0,1,0,0,0,0,0],
         [1,1,1,0,0,0,0,0,0,0],
         [0,1,1,1,0,0,0,0,0,0]]
H_array = np.array(H_adj, dtype=float)
H = adjacency_to_weighted_graph(H_array)

def initialize_graphon_matrix(N: int) -> np.ndarray:
    """
    Initializes an N x N symmetric graphon matrix with values between 0 and 1.
    """
    W = np.random.rand(N, N)
    # Ensure symmetry for undirected graphs
    W = (W + W.T) / 2
    # np.fill_diagonal(W, 0)  # No self-loops
    W = np.round(W, 2)
    return W

def calculate_edge_density(W: np.ndarray) -> Decimal:
    """
    Calculates 2m/n² where m is the sum of edge weights and n is the number of vertices.
    """
    n = W.shape[0]
    # Sum upper triangle (excluding diagonal) and multiply by 2 for undirected graph
    m = np.sum(np.triu(W, k=1))
    m_decimal = Decimal(str(m))
    n_decimal = Decimal(n)
    density = (Decimal('2') * m_decimal) / (n_decimal * n_decimal)
    return density

def calculate_homomorphism_density(H: WeightedGraph, W: np.ndarray) -> Decimal:
    """
    Calculates t(H,W) using exact homomorphism counting.
    """
    G = adjacency_to_weighted_graph(W)
    num_homomorphisms = count_homomorphisms(H, G)
    
    # Normalize by |V(G)|^|V(H)| to get the density
    normalization = Decimal(W.shape[0]) ** len(H.vertices)
    return num_homomorphisms / normalization

def calculate_reward(W: np.ndarray) -> Decimal:
    """
    Calculates (2m/n²)^E(H) - t(H,W)
    where E(H) is the number of edges in H
    """
    edge_density = calculate_edge_density(W)
    t_hw = calculate_homomorphism_density(H, W)
    
    # Count edges in H
    E_H = sum(1 for v in H.edge_weights for u in H.edge_weights[v] if int(v) < int(u))
    reward = edge_density ** E_H - t_hw
    return reward

def apply_random_graphon_action(W: np.ndarray, step_size: float = 0.05) -> np.ndarray:
    """
    Applies a random modification to the graphon matrix.
    """
    N = W.shape[0]
    W_new = W.copy()

    # Choose a random entry in upper triangle
    i = random.randint(0, N-2)
    j = random.randint(i+1, N-1)

    # Nudge the value
    change = random.choice([-step_size, step_size])
    W_new[i, j] += change
    W_new[j, i] = W_new[i, j]  # Maintain symmetry
    
    # Ensure values stay within [0, 1]
    W_new[i, j] = np.clip(W_new[i, j], 0, 1)
    W_new[j, i] = W_new[i, j]
    return W_new

# --- RL Simulation Loop ---
if __name__ == "__main__":
    # Parameters
    GRAPHON_MATRIX_DIM = 4  # Small dimension for verification
    NUM_SIMULATION_STEPS = 10  # Fewer steps since exact counting is slower
    
    # Initialize
    current_W = initialize_graphon_matrix(GRAPHON_MATRIX_DIM)
    initial_reward = calculate_reward(current_W)
    best_W = current_W.copy()
    best_reward = initial_reward
    current_reward = initial_reward

    print(f"Initial reward: {best_reward:.6e}")

    print("\nStarting graphon evolution...")
    for step in range(NUM_SIMULATION_STEPS):
        # Apply random action
        candidate_W = apply_random_graphon_action(current_W, step_size=0.1)
        candidate_reward = calculate_reward(candidate_W)
        print(f"Step {step + 1} -- Current Reward: {candidate_reward:.6e}")

        # Update if better
        if candidate_reward > current_reward:
            current_W = candidate_W
            current_reward = candidate_reward
            print()
            if current_reward > best_reward:
                best_reward = current_reward
                best_W = current_W.copy()
                print(f"  New Best Graphon!\n")
                print(f"  Best graphon W:\n  {best_W}\n")
        
        if (step + 1) % 10 == 0:  # More frequent updates since fewer steps
            print(f"Step {step + 1}: Current reward: {current_reward:.6e}")

    print("\nSimulation finished.")
    print(f"Final best reward: {best_reward:.6e}")
    print(f"Final edge density: {calculate_edge_density(best_W):.6f}")
    print(f"Final t(H,W): {calculate_homomorphism_density(H, best_W):.6e}")
    print(f"Final best graphon W:\n{best_W}")

    # Check if conjecture is violated
    edge_density = calculate_edge_density(best_W)
    t_hw = calculate_homomorphism_density(H, best_W)
    E_H = sum(1 for v in H.edge_weights for u in H.edge_weights[v] if int(v) < int(u))
    rhs = edge_density ** E_H

    print("\nVerification for best graphon:")
    print(f"  t(H,W) = {t_hw:.6e}")
    print(f"  (2m/n²)^E(H) = {rhs:.6e}")
    if t_hw < rhs:
        print("  Sidorenko's Conjecture: VIOLATED!")
        print(f"  Violation margin: {(rhs - t_hw):.6e}")
        print(f"  Violation ratio: {rhs/t_hw:.2f}x")
    else:
        print("  Sidorenko's Conjecture: Holds")
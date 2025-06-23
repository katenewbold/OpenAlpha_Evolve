import random
from decimal import Decimal, getcontext
import sys
import os

# Set Decimal precision globally for calculations
getcontext().prec = 50

# Import homlib. This requires the CountHom C++ library to be built and installed.
# To install: navigate to the 'CountHom' directory within OpenAlpha_Evolve and run 'pip install ./'
try:
    import homlib
except ImportError:
    print("Error: 'homlib' not found. Please install it from the 'CountHom' directory.")
    print("Navigate to: OpenAlpha_Evolve/CountHom")
    print("Then run: pip install ./")
    sys.exit(1)

def get_num_vertices(adj_matrix: list[list[int]]) -> int:
    """Returns the number of vertices from an adjacency matrix."""
    return len(adj_matrix)

def get_num_edges(adj_matrix: list[list[int]]) -> int:
    """Returns the number of edges from an adjacency matrix (for undirected graphs)."""
    n = len(adj_matrix)
    edges = 0
    for i in range(n):
        for j in range(i + 1, n): # Count each edge once (upper triangle)
            if adj_matrix[i][j] == 1:
                edges += 1
    return edges

# Define the fixed graph H (source graph) - K_{3,3} as an adjacency matrix
# K_{3,3} has 6 vertices. Partitions: {0,1,2} and {3,4,5}. Edges only between partitions.
H_adj_matrix = [[0,0,0,0,0,0,1,1,1,0],
                [0,0,0,0,0,0,0,1,1,1],
                [0,0,0,0,0,1,0,0,1,1],
                [0,0,0,0,0,1,1,0,0,1],
                [0,0,0,0,0,1,1,1,0,0],
                [0,0,1,1,1,0,0,0,0,0],
                [1,0,0,1,1,0,0,0,0,0],
                [1,1,0,0,1,0,0,0,0,0],
                [1,1,1,0,0,0,0,0,0,0],
                [0,1,1,1,0,0,0,0,0,0]]
H_num_vertices = get_num_vertices(H_adj_matrix)
H_num_edges = Decimal(get_num_edges(H_adj_matrix))

# Create homlib.Graph for H once
H_homlib = homlib.Graph(H_adj_matrix)

print(f"Fixed graph H (K3,3) has {H_num_vertices} vertices and {H_num_edges} edges.")

def initialize_random_graph(num_vertices: int, edge_probability: float) -> list[list[int]]:
    """
    Initializes a random graph G as an adjacency matrix using the G(n,p) model.
    """
    adj_matrix = [[0 for _ in range(num_vertices)] for _ in range(num_vertices)]
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices): # Iterate upper triangle to ensure undirected
            if random.random() < edge_probability:
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1 # Make it symmetric for undirected graph
    return adj_matrix

def calculate_homomorphism_density(H_homlib: homlib.Graph, G_adj_matrix: list[list[int]]) -> Decimal:
    """
    Calculates the homomorphism density t(H,G) = hom(H,G) / |V(G)|^|V(H)|
    using the homlib C++ function.
    """
    num_vertices_G = get_num_vertices(G_adj_matrix)
    if num_vertices_G == 0:
        return Decimal('0.0')

    G_homlib = homlib.Graph(G_adj_matrix)

    num_homomorphisms = homlib.countHom(H_homlib, G_homlib)
    
    # Normalization factor: |V(G)|^|V(H)|
    normalization_factor = Decimal(num_vertices_G) ** Decimal(H_num_vertices)
    
    if normalization_factor == Decimal('0.0'):
        return Decimal('inf') # Should not happen if G has vertices

    density = Decimal(str(num_homomorphisms)) / normalization_factor
    return density

def calculate_edge_density_G(G_adj_matrix: list[list[int]]) -> Decimal:
    """
    Calculates the edge density t(K2,G) = (2 * e(G)) / |V(G)|^2 for graph G.
    """
    num_vertices_G = Decimal(get_num_vertices(G_adj_matrix))
    num_edges_G = Decimal(get_num_edges(G_adj_matrix))

    if num_vertices_G == Decimal('0.0'):
        return Decimal('0.0')
    if num_vertices_G == Decimal('1.0'):
        # For a single vertex graph, edge density is 0.
        return Decimal('0.0')

    # Formula (2 * e(G)) / |V(G)|^2
    return (Decimal('2.0') * num_edges_G) / (num_vertices_G ** Decimal('2.0'))

def calculate_reward(G_adj_matrix: list[list[int]], H_homlib: homlib.Graph, H_num_edges: Decimal) -> Decimal:
    """
    Calculates the reward: (t(K2,G)^e(H)) - t(H,G).
    Maximizing this seeks a counterexample to Sidorenko's conjecture.
    """
    t_H_G = calculate_homomorphism_density(H_homlib, G_adj_matrix)
    t_K2_G = calculate_edge_density_G(G_adj_matrix)
    
    # Calculate t(K2,G)^e(H)
    if t_K2_G < 0: # Should not happen for simple graphs
        t_K2_G_pow_e_H = Decimal('0.0')
    else:
        # Exponent for Decimal must be an integer, H_num_edges is Decimal but integer-like
        t_K2_G_pow_e_H = t_K2_G ** int(H_num_edges)
    
    reward = t_K2_G_pow_e_H - t_H_G
    return reward

def apply_random_edge_action(adj_matrix: list[list[int]]) -> list[list[int]]:
    """
    Applies a random edge addition or removal action to graph G, directly modifying the adjacency matrix.
    Ensures the graph remains simple (no loops, no multiple edges).
    """
    n = get_num_vertices(adj_matrix)
    if n < 2:
        return [row[:] for row in adj_matrix] # Return a copy if cannot modify

    # Create a deep copy to modify
    G_new_adj = [row[:] for row in adj_matrix]

    action_type = random.choice(['add', 'remove'])

    if action_type == 'add':
        attempts = 0
        while attempts < 10: # Limit attempts to avoid infinite loops on dense graphs
            u, v = random.sample(range(n), 2)
            if u == v: continue # No loops
            if G_new_adj[u][v] == 0: # If no edge exists
                G_new_adj[u][v] = 1
                G_new_adj[v][u] = 1 # Symmetric for undirected graph
                break
            attempts += 1
    elif action_type == 'remove':
        current_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if G_new_adj[i][j] == 1:
                    current_edges.append((i, j))
        
        if current_edges:
            u, v = random.choice(current_edges)
            G_new_adj[u][v] = 0
            G_new_adj[v][u] = 0
        else:
            # If no edges to remove, try adding an edge as an alternative
            attempts = 0
            while attempts < 10:
                u, v = random.sample(range(n), 2)
                if u == v: continue
                if G_new_adj[u][v] == 0:
                    G_new_adj[u][v] = 1
                    G_new_adj[v][u] = 1
                    break
                attempts += 1
    return G_new_adj

# --- Simulation Loop ---
if __name__ == "__main__":
    # Parameters for the RL simulation
    NUM_VERTICES_G = 10  # Number of vertices in the target graph G
    INITIAL_EDGE_PROBABILITY_G = 0.5 # Initial sparsity/density of G
    NUM_SIMULATION_STEPS = 100 # Number of iterations to evolve G

    # Initialize the target graph G
    current_G_adj = initialize_random_graph(NUM_VERTICES_G, INITIAL_EDGE_PROBABILITY_G)
    
    # Calculate initial reward
    initial_reward = calculate_reward(current_G_adj, H_homlib, H_num_edges)
    
    best_G_adj = [row[:] for row in current_G_adj] # Deep copy
    best_reward = initial_reward
    current_reward = initial_reward

    print(f"Initial G has {get_num_vertices(current_G_adj)} vertices and {get_num_edges(current_G_adj)} edges.")
    print(f"Initial reward (t(K2,G)^e(H) - t(H,G)): {initial_reward:.6e}")

    print("\nStarting graph evolution to find Sidorenko counterexample...")
    for step in range(NUM_SIMULATION_STEPS):
        # Apply a random action to get the next state (next G)
        next_G_adj = apply_random_edge_action(current_G_adj)

        # Calculate reward for the new graph G
        reward = calculate_reward(next_G_adj, H_homlib, H_num_edges)

        print(f"\nStep {step + 1}:")
        print(f"  G has {get_num_vertices(next_G_adj)} vertices and {get_num_edges(next_G_adj)} edges.")
        print(f"  Current reward: {reward:.6e}")

        # Update best graph if current reward is better
        if reward > best_reward:
            best_reward = reward
            best_G_adj = [row[:] for row in next_G_adj] # Deep copy
            print("  New best graph found!")

        # Update current graph for the next step (simple greedy update here)
        current_G_adj = [row[:] for row in next_G_adj]

    print("\nSimulation finished.")
    print(f"Final best reward found: {best_reward:.6e}")
    print(f"Final best graph G has {get_num_vertices(best_G_adj)} vertices and {get_num_edges(best_G_adj)} edges.")
    print("Adjacency matrix of the best graph G:")
    for row in best_G_adj:
        print(row)

    # Verify if a counterexample was found (reward > 0)
    if best_reward > Decimal('0.0'):
        print("\nPossible counterexample to Sidorenko's Conjecture found!")
        t_H_G_best = calculate_homomorphism_density(H_homlib, best_G_adj)
        t_K2_G_best = calculate_edge_density_G(best_G_adj)
        rhs_best = t_K2_G_best ** int(H_num_edges)
        print(f"  For H (K3,3) and G with {get_num_vertices(best_G_adj)} vertices, {get_num_edges(best_G_adj)} edges:")
        print(f"    t(H,G) = {t_H_G_best:.6e}")
        print(f"    t(K2,G)^e(H) = {rhs_best:.6e}")
        print(f"    Difference (t(K2,G)^e(H) - t(H,G)): {best_reward:.6e}")
    else:
        print("\nNo counterexample found in this simulation run. Sidorenko's Conjecture holds for the graphs explored.")
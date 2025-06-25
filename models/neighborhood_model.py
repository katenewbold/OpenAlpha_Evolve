# neighborhood_model.py
import numpy as np
import random

def create_oriented_adjacency_matrix(N: int) -> np.ndarray:
    """
    Creates an N x N adjacency matrix for an oriented graph.
    An oriented graph is a directed graph with no self-loops, no parallel edges, and no 2-cycles.
    Ensures all vertices have out-degree at least min_out_degree.
    """
    G = np.zeros((N, N), dtype=int)
    
    # First, ensure each vertex has at least min_out_degree outgoing edges
    for i in range(N):
        # Find valid targets for vertex i
        valid_targets = [j for j in range(N) if j != i and G[j, i] == 0]
        random.shuffle(valid_targets)
        targets = valid_targets[:MIN_OUT_DEGREE]
        for target in targets:
            G[i, target] = 1
    
    # Add additional random edges (no self-loops, no 2-cycles)
    for i in range(N):
        for j in range(N):
            if i != j and G[i, j] == 0 and G[j, i] == 0:
                if random.random() < 0.8:
                    G[i, j] = 1
    
    return G

def calculate_first_neighborhood(G: np.ndarray, vertex: int) -> set:
    """
    Calculates the first neighborhood (out-neighbors) of a vertex in the oriented graph.
    """
    return set(np.where(G[vertex] == 1)[0])

def calculate_second_neighborhood(G: np.ndarray, vertex: int) -> set:
    """
    Calculates the second neighborhood of a vertex in the oriented graph.
    The second neighborhood is the set of vertices reachable by following exactly 2 edges.
    """
    N = G.shape[0]
    first_neighbors = calculate_first_neighborhood(G, vertex)
    second_neighbors = set()
    
    # For each first neighbor, find its out-neighbors
    for neighbor in first_neighbors:
        # Get out-neighbors of the neighbor
        neighbor_neighbors = set(np.where(G[neighbor] == 1)[0])
        # Add vertices that are not the original vertex or its first neighbors
        second_neighbors.update(neighbor_neighbors - first_neighbors - {vertex})
        
    return second_neighbors

def calculate_reward(G: np.ndarray) -> float:
    """
    Calculates the reward based on the second neighborhood conjecture.
    For a counterexample, ALL vertices must have |N₁(v)| > |N₂(v)|.
    The reward encourages more vertices to violate the conjecture.
    """
    N = G.shape[0]
    violations = 0  # Count vertices that violate the conjecture
    total_difference = 0  # Sum of all differences
    min_difference = float('inf')
    
    for v in range(N):
        # Calculate first neighborhood size
        first_neighbors = calculate_first_neighborhood(G, v)
        first_size = len(first_neighbors)
        
        # Calculate second neighborhood size
        second_neighbors = calculate_second_neighborhood(G, v)
        second_size = len(second_neighbors)
        
        # Calculate difference (first - second)
        difference = first_size - second_size
        total_difference += difference
        min_difference = min(min_difference, difference)
        
        # Count violations (positive difference means violation)
        if difference > 0:
            violations += 1
    
    # Reward components:
    # 1. Number of violating vertices (higher is better)
    # 2. Minimum difference (higher is better)
    # 3. Average difference (higher is better)
    # 4. Bonus for having all vertices violate the conjecture
    
    violation_bonus = 10000 if violations == N else 0  # Large bonus if all vertices violate
    avg_difference = total_difference / N
    
    reward = (violations * 50) + min_difference * 2 + (avg_difference * 0.5) + violation_bonus

    return reward

def is_valid_oriented_graph(G: np.ndarray) -> bool:
    """
    Checks if the graph is a valid oriented graph (no self-loops, no 2-cycles).
    """
    N = G.shape[0]
    
    # Check for self-loops
    if np.any(np.diag(G) != 0):
        return False
    
    # Check for 2-cycles (if G[i,j] = 1, then G[j,i] should be 0)
    for i in range(N):
        for j in range(N):
            if i != j and G[i,j] == 1 and G[j,i] == 1:
                return False
            
    # Check if the graph is connected``
    if not connected_graph(G):
        return False
    
    return True

def connected_graph(G: np.ndarray) -> bool:
    """
    Checks if the graph is connected.
    """
    import networkx as nx
    N = G.shape[0]
    return nx.is_connected(nx.DiGraph(G))

def has_sinks(G: np.ndarray) -> bool:
    """
    Checks if the graph has any sinks (vertices with out-degree 0).
    """
    N = G.shape[0]
    for i in range(N):
        if np.sum(G[i, :]) == 0:  # No outgoing edges
            return True
    return False

def fix_sinks(G: np.ndarray) -> np.ndarray:
    """
    Fixes any sinks in the graph by adding outgoing edges.
    """
    N = G.shape[0]
    G_fixed = G.copy()
    
    for i in range(N):
        if np.sum(G_fixed[i, :]) == 0:  # This is a sink
            # Find a valid target (no 2-cycles)
            valid_targets = [j for j in range(N) if j != i and G_fixed[j, i] == 0]
            if valid_targets:
                target = random.choice(valid_targets)
                G_fixed[i, target] = 1
    
    return G_fixed

def apply_random_action(G: np.ndarray) -> np.ndarray:
    """
    Applies a random modification to the oriented graph by either adding or removing an edge.
    Ensures the result is still a valid oriented graph with out-degree at least min_out_degree for all vertices.
    """
    N = G.shape[0]
    G_new = G.copy()
    
    # Try up to 20 times to find a valid modification
    for _ in range(20):
        # Choose random vertices
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        
        if i != j:  # No self-loops
            # Try toggling the edge
            if G_new[i, j] == 0 and G_new[j, i] == 0:
                G_new[i, j] = 1
                if np.sum(G_new[i, :]) >= MIN_OUT_DEGREE:
                    return G_new
                G_new[i, j] = 0
            elif G_new[i, j] == 1:
                # Only remove if out-degree stays >= min_out_degree
                if np.sum(G_new[i, :]) > MIN_OUT_DEGREE:
                    G_new[i, j] = 0
                    return G_new
    
    return G

def print_graph_stats(G: np.ndarray):
    """
    Prints statistics about the oriented graph.
    """
    N = G.shape[0]
    edge_count = np.sum(G)  # For directed graph, sum all entries
    
    print(f"\nOriented Graph Statistics:")
    print(f"Number of vertices: {N}")
    print(f"Number of edges: {edge_count}")
    print(f"Edge density: {edge_count/(N*(N-1)):.3f}")
    print(f"Valid oriented graph: {is_valid_oriented_graph(G)}")
    
    # Print neighborhood statistics
    print("\nNeighborhood Statistics:")
    for v in range(N):
        first_neighbors = calculate_first_neighborhood(G, v)
        second_neighbors = calculate_second_neighborhood(G, v)
        first_size = len(first_neighbors)
        second_size = len(second_neighbors)
        difference = first_size - second_size
        status = "VIOLATION" if difference > 0 else "OK"
        print(f"Vertex {v}: |N₁| = {first_size}, |N₂| = {second_size}, diff = {difference} ({status})")

def check_conjecture_violation(G: np.ndarray) -> bool:
    """
    Checks if the graph violates the second neighborhood conjecture.
    Returns True if ALL vertices have |N₁(v)| > |N₂(v)|.
    """
    N = G.shape[0]
    
    for v in range(N):
        first_neighbors = calculate_first_neighborhood(G, v)
        second_neighbors = calculate_second_neighborhood(G, v)
        first_size = len(first_neighbors)
        second_size = len(second_neighbors)
        
        if first_size <= second_size:
            return False  # This vertex doesn't violate the conjecture
    
    return True  # All vertices violate the conjecture

def apply_smart_action(G: np.ndarray) -> np.ndarray:
    """
    Applies a smart modification to the oriented graph.
    Prioritizes actions that are likely to improve the reward while avoiding low out-degree.
    """
    N = G.shape[0]
    G_new = G.copy()
    
    # Calculate current neighborhood sizes for all vertices
    current_first_sizes = []
    current_second_sizes = []
    current_differences = []
    
    for v in range(N):
        first_neighbors = calculate_first_neighborhood(G, v)
        second_neighbors = calculate_second_neighborhood(G, v)
        first_size = len(first_neighbors)
        second_size = len(second_neighbors)
        difference = first_size - second_size
        
        current_first_sizes.append(first_size)
        current_second_sizes.append(second_size)
        current_differences.append(difference)
    
    # Find vertices with the worst (most negative) differences
    worst_vertices = []
    for v in range(N):
        if current_differences[v] < 0:
            worst_vertices.append((v, current_differences[v]))
    
    # Sort by worst difference first
    worst_vertices.sort(key=lambda x: x[1])
    
    # Try to improve the worst vertices first
    for vertex, _ in worst_vertices[:3]:  # Focus on 3 worst vertices
        # Try adding edges from this vertex
        for target in range(N):
            if target != vertex and G[vertex, target] == 0 and G[target, vertex] == 0:
                # Try adding edge vertex -> target
                G_new[vertex, target] = 1
                if np.sum(G_new[vertex, :]) >= MIN_OUT_DEGREE and calculate_reward(G_new) > calculate_reward(G):
                    return G_new
                G_new[vertex, target] = 0
        
        # Try removing edges to this vertex (to reduce its second neighborhood)
        # But be careful not to create low out-degree
        for source in range(N):
            if source != vertex and G[source, vertex] == 1:
                # Check if removing this edge would create low out-degree
                if np.sum(G[source, :]) > MIN_OUT_DEGREE:
                    # Try removing edge source -> vertex
                    G_new[source, vertex] = 0
                    if calculate_reward(G_new) > calculate_reward(G):
                        return G_new
                    G_new[source, vertex] = 1
    
    # If no smart action found, fall back to random action
    return apply_random_action(G)

def print_detailed_stats(G: np.ndarray):
    """
    Prints detailed statistics about the graph and reward calculation.
    """
    N = G.shape[0]
    # edge_count = np.sum(G)
    
    # print(f"\nOriented Graph Statistics:")
    # print(f"Number of vertices: {N}")
    # print(f"Number of edges: {edge_count}")
    # print(f"Edge density: {edge_count/(N*(N-1)):.3f}")
    # print(f"Valid oriented graph: {is_valid_oriented_graph(G)}")
    # print(f"Has low out-degree (<{MIN_OUT_DEGREE}): {has_low_outdegree(G)}")
    
    # Print neighborhood statistics
    print("\nNeighborhood Statistics:")
    for v in range(N):
        first_neighbors = calculate_first_neighborhood(G, v)
        second_neighbors = calculate_second_neighborhood(G, v)
        first_size = len(first_neighbors)
        second_size = len(second_neighbors)
        difference = first_size - second_size
        status = "VIOLATION" if difference > 0 else "OK"
        out_degree = np.sum(G[v, :])
        print(f"Vertex {v}: |N₁| = {first_size}, |N₂| = {second_size}, diff = {difference} ({status}), out-degree = {out_degree}")

def has_low_outdegree(G: np.ndarray) -> bool:
    """
    Checks if the graph has any vertices with out-degree less than min_out_degree.
    """
    N = G.shape[0]
    for i in range(N):
        if np.sum(G[i, :]) < MIN_OUT_DEGREE:
            return True
    return False

def fix_low_outdegree(G: np.ndarray) -> np.ndarray:
    """
    Fixes any vertices with out-degree less than min_out_degree by adding outgoing edges.
    """
    N = G.shape[0]
    G_fixed = G.copy()
    for i in range(N):
        out_deg = np.sum(G_fixed[i, :])
        if out_deg < MIN_OUT_DEGREE:
            # Find valid targets
            valid_targets = [j for j in range(N) if j != i and G_fixed[j, i] == 0 and G_fixed[i, j] == 0]
            random.shuffle(valid_targets)
            needed = MIN_OUT_DEGREE - out_deg
            for target in valid_targets[:needed]:
                G_fixed[i, target] = 1
    return G_fixed

# --- RL Simulation Loop ---
if __name__ == "__main__":
    # Parameters
    NUM_VERTICES = 40  # Number of vertices in the graph
    NUM_SIMULATION_STEPS = 500  # Number of iterations
    MIN_OUT_DEGREE = 7  # Minimum out-degree for all vertices
    
    # Initialize
    current_G = create_oriented_adjacency_matrix(NUM_VERTICES)
    initial_reward = calculate_reward(current_G)
    best_G = current_G.copy()
    best_reward = initial_reward
    current_reward = initial_reward
    
    print(f"Initial reward: {best_reward}")
    # print_detailed_stats(current_G, MIN_OUT_DEGREE)
    
    print("\nStarting graph evolution...")
    for step in range(NUM_SIMULATION_STEPS):

        candidate_G = apply_smart_action(current_G)
        
        candidate_reward = calculate_reward(candidate_G)
        
        if candidate_reward > current_reward:
            current_G = candidate_G
            current_reward = candidate_reward
            if current_reward > best_reward:
                best_reward = current_reward
                best_G = current_G.copy()
                print(f"\nStep {step + 1}: New best! Reward: {best_reward}")
                # print_detailed_stats(best_G, MIN_OUT_DEGREE)
        
        if (step + 1) % 20 == 0:
            print(f"Step {step + 1}: Current reward: {current_reward}")
    
    print("\nSimulation finished.")
    print(f"Final best reward: {best_reward}")
    print("\nFinal graph statistics:")
    print_detailed_stats(best_G)
    # print(best_G)
    
    # Check if conjecture is violated
    if check_conjecture_violation(best_G):
        print("\nSecond Neighborhood Conjecture: VIOLATED!")
        print("All vertices have |N₁(v)| > |N₂(v)|")
    else:
        print("\nSecond Neighborhood Conjecture: Holds")
        print("At least one vertex has |N₁(v)| ≤ |N₂(v)|")

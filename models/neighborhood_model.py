# neighborhood_model.py
import numpy as np
import random

# Global constants
MIN_OUT_DEGREE = 7

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
                if random.random() < 0.75:
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
    violations = 0
    total_difference = 0
    min_difference = float('inf')
    for v in range(N):
        first_size = len(calculate_first_neighborhood(G, v))
        second_size = len(calculate_second_neighborhood(G, v))
        difference = first_size - second_size
        total_difference += difference
        min_difference = min(min_difference, difference)
        if difference > 0:
            violations += 1
    violation_fraction = violations / N if N > 0 else 0
    violation_bonus = 10000 if violations == N else 0  # Bonus for complete violation
    
    # Make violation ratio the primary driver with heavy positive reward
    reward = violation_fraction * 100  # Heavy positive weight on violation ratio
    
    # Add smaller contributions from other factors
    reward += min_difference * 0.5  # Small positive contribution from min difference
    reward += violation_bonus  # Bonus for complete violation
    
    return reward

def is_valid_oriented_graph(G: np.ndarray) -> bool:
    """
    Checks if the graph is a valid oriented graph (no self-loops, no 2-cycles).
    Removed connectedness requirement to allow for more flexible evolution.
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
    
    # Check minimum out-degree requirement
    for i in range(N):
        if np.sum(G[i, :]) < MIN_OUT_DEGREE:
            return False
    
    return True

def apply_random_action(G: np.ndarray) -> np.ndarray:
    """
    Applies a random modification to the oriented graph by either adding or removing an edge.
    Tries multiple random changes to find one that improves the reward.
    """
    N = G.shape[0]
    current_reward = calculate_reward(G)
    
    # Try multiple random modifications to find one that improves the reward
    for attempt in range(50):  # Try up to 50 random modifications
        G_test = G.copy()
        
        # Choose random vertices
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        
        if i != j:  # No self-loops
            if G_test[i, j] == 0 and G_test[j, i] == 0:
                # Add edge
                G_test[i, j] = 1
                if np.sum(G_test[i, :]) >= MIN_OUT_DEGREE:
                    test_reward = calculate_reward(G_test)
                    if test_reward > current_reward:
                        return G_test
            elif G_test[i, j] == 1:
                # Remove edge (only if out-degree stays >= min_out_degree)
                if np.sum(G_test[i, :]) > MIN_OUT_DEGREE:
                    G_test[i, j] = 0
                    test_reward = calculate_reward(G_test)
                    if test_reward > current_reward:
                        return G_test
    
    # If no improving modification found, try any valid modification
    for _ in range(20):
        G_test = G.copy()
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        
        if i != j:
            if G_test[i, j] == 0 and G_test[j, i] == 0:
                # Add edge
                G_test[i, j] = 1
                if np.sum(G_test[i, :]) >= MIN_OUT_DEGREE:
                    return G_test
            elif G_test[i, j] == 1:
                # Remove edge
                if np.sum(G_test[i, :]) > MIN_OUT_DEGREE:
                    G_test[i, j] = 0
                    return G_test
    
    # Last resort: return the original graph
    return G.copy()

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
    current_reward = calculate_reward(G)
    
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
    
    # Strategy 1: Try to improve vertices with negative differences (worst performers)
    worst_vertices = []
    for v in range(N):
        if current_differences[v] < 0:
            worst_vertices.append((v, current_differences[v]))
    
    # Sort by worst difference first
    worst_vertices.sort(key=lambda x: x[1])
    
    # Try to improve the worst vertices first
    for vertex, _ in worst_vertices[:5]:  # Focus on 5 worst vertices
        # Try adding edges from this vertex to increase its first neighborhood
        for target in range(N):
            if target != vertex and G[vertex, target] == 0 and G[target, vertex] == 0:
                # Try adding edge vertex -> target
                G_test = G.copy()
                G_test[vertex, target] = 1
                if np.sum(G_test[vertex, :]) >= MIN_OUT_DEGREE:
                    test_reward = calculate_reward(G_test)
                    if test_reward > current_reward:
                        return G_test
        
        # Try removing edges to this vertex to reduce its second neighborhood
        for source in range(N):
            if source != vertex and G[source, vertex] == 1:
                # Check if removing this edge would create low out-degree
                if np.sum(G[source, :]) > MIN_OUT_DEGREE:
                    # Try removing edge source -> vertex
                    G_test = G.copy()
                    G_test[source, vertex] = 0
                    test_reward = calculate_reward(G_test)
                    if test_reward > current_reward:
                        return G_test
    
    # Strategy 2: Try to create violations by increasing first neighborhoods
    # Find vertices that are close to violating (small positive differences)
    close_vertices = []
    for v in range(N):
        if 0 <= current_differences[v] < 3:  # Close to violating
            close_vertices.append((v, current_differences[v]))
    
    close_vertices.sort(key=lambda x: x[1])  # Sort by smallest positive difference
    
    for vertex, _ in close_vertices[:3]:
        # Try adding edges from this vertex
        for target in range(N):
            if target != vertex and G[vertex, target] == 0 and G[target, vertex] == 0:
                G_test = G.copy()
                G_test[vertex, target] = 1
                if np.sum(G_test[vertex, :]) >= MIN_OUT_DEGREE:
                    test_reward = calculate_reward(G_test)
                    if test_reward > current_reward:
                        return G_test
    
    # Strategy 3: Try to reduce second neighborhoods by removing edges
    # Find vertices with large second neighborhoods
    large_second_neighbors = []
    for v in range(N):
        if current_second_sizes[v] > current_first_sizes[v]:
            large_second_neighbors.append((v, current_second_sizes[v] - current_first_sizes[v]))
    
    large_second_neighbors.sort(key=lambda x: x[1], reverse=True)  # Sort by largest excess
    
    for vertex, _ in large_second_neighbors[:3]:
        # Try removing edges that contribute to this vertex's second neighborhood
        for source in range(N):
            if source != vertex and G[source, vertex] == 1:
                if np.sum(G[source, :]) > MIN_OUT_DEGREE:
                    G_test = G.copy()
                    G_test[source, vertex] = 0
                    test_reward = calculate_reward(G_test)
                    if test_reward > current_reward:
                        return G_test
    
    # Strategy 4: Try random edge additions that might help
    for _ in range(20):  # Try 20 random additions
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        if i != j and G[i, j] == 0 and G[j, i] == 0:
            G_test = G.copy()
            G_test[i, j] = 1
            if np.sum(G_test[i, :]) >= MIN_OUT_DEGREE:
                test_reward = calculate_reward(G_test)
                if test_reward > current_reward:
                    return G_test
    
    # If no smart action found, fall back to random action
    return apply_random_action(G)

def print_detailed_stats(G: np.ndarray):
    """
    Prints detailed statistics about the graph and reward calculation.
    """
    N = G.shape[0]
    
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

def add_vertex_with_edges(G: np.ndarray) -> np.ndarray:
    """
    Adds a new vertex to the graph with at least min_out_degree outgoing edges.
    Returns the new adjacency matrix.
    """
    N = G.shape[0]
    new_N = N + 1
    
    # Create new adjacency matrix with one more row and column
    G_new = np.zeros((new_N, new_N), dtype=int)
    G_new[:N, :N] = G  # Copy existing graph
    
    # Add outgoing edges from the new vertex
    valid_targets = list(range(N))  # Can connect to any existing vertex
    random.shuffle(valid_targets)
    
    # Add at least min_out_degree outgoing edges
    for target in valid_targets[:MIN_OUT_DEGREE]:
        G_new[N, target] = 1
    
    # Add some incoming edges to the new vertex (to maintain connectivity)
    # But avoid creating 2-cycles
    num_incoming = random.randint(MIN_OUT_DEGREE//2, MIN_OUT_DEGREE)
    valid_sources = []
    for source in range(N):
        # Only add edge if it doesn't create a 2-cycle
        if G_new[source, N] == 0 and G_new[N, source] == 0:
            valid_sources.append(source)
    
    random.shuffle(valid_sources)
    for source in valid_sources[:num_incoming]:
        G_new[source, N] = 1
    
    return G_new

def apply_multiple_edge_changes(G: np.ndarray, num_changes: int = 3) -> np.ndarray:
    """
    Applies multiple edge changes (additions/removals) to the graph.
    Ensures the result maintains minimum out-degree for all vertices.
    """
    N = G.shape[0]
    G_new = G.copy()
    
    changes_made = 0
    attempts = 0
    max_attempts = 50
    
    while changes_made < num_changes and attempts < max_attempts:
        attempts += 1
        
        # Choose random vertices
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        
        if i != j:  # No self-loops
            # Try toggling the edge
            if G_new[i, j] == 0 and G_new[j, i] == 0:
                # Add edge
                G_new[i, j] = 1
                if np.sum(G_new[i, :]) >= MIN_OUT_DEGREE:
                    changes_made += 1
                else:
                    G_new[i, j] = 0  # Revert if it creates low out-degree
            elif G_new[i, j] == 1:
                # Remove edge (only if out-degree stays >= min_out_degree)
                if np.sum(G_new[i, :]) > MIN_OUT_DEGREE:
                    G_new[i, j] = 0
                    changes_made += 1
                else:
                    G_new[i, j] = 1  # Revert if it would create low out-degree
    
    return G_new

def apply_vertex_operation(G: np.ndarray) -> np.ndarray:
    """
    Only adds a new vertex with at least min_out_degree outgoing edges.
    """
    return add_vertex_with_edges(G)

def calculate_violation_stats(G: np.ndarray) -> tuple:
    """
    Calculates violation statistics for the graph.
    Returns (violation_count, violation_fraction, avg_difference, min_difference)
    """
    N = G.shape[0]
    violations = 0
    total_difference = 0
    min_difference = float('inf')
    
    for v in range(N):
        first_neighbors = calculate_first_neighborhood(G, v)
        second_neighbors = calculate_second_neighborhood(G, v)
        first_size = len(first_neighbors)
        second_size = len(second_neighbors)
        difference = first_size - second_size
        
        total_difference += difference
        min_difference = min(min_difference, difference)
        
        if difference > 0:
            violations += 1
    
    violation_fraction = violations / N if N > 0 else 0
    avg_difference = total_difference / N if N > 0 else 0
    
    return violations, violation_fraction, avg_difference, min_difference

def apply_exploration_action(G: np.ndarray) -> np.ndarray:
    """
    Applies exploration actions when the search is stuck.
    Makes more aggressive changes to escape local optima.
    """
    N = G.shape[0]
    current_reward = calculate_reward(G)
    
    # Strategy 1: Try multiple edge changes at once
    for _ in range(15):  # Increased from 10
        G_test = G.copy()
        num_changes = random.randint(3, 8)  # Increased range
        
        for _ in range(num_changes):
            i = random.randint(0, N-1)
            j = random.randint(0, N-1)
            if i != j:
                if G_test[i, j] == 0 and G_test[j, i] == 0:
                    G_test[i, j] = 1
                elif G_test[i, j] == 1 and np.sum(G_test[i, :]) > MIN_OUT_DEGREE:
                    G_test[i, j] = 0
        
        # Fix any low out-degree issues
        G_test = fix_low_outdegree(G_test)
        if is_valid_oriented_graph(G_test):
            test_reward = calculate_reward(G_test)
            if test_reward > current_reward - 10:  # Allow larger decreases for exploration
                return G_test
    
    # Strategy 2: Try removing edges from vertices with high out-degree
    for _ in range(30):  # Increased from 20
        G_test = G.copy()
        # Find vertices with high out-degree
        high_out_degree = []
        for v in range(N):
            out_deg = np.sum(G_test[v, :])
            if out_deg > MIN_OUT_DEGREE + 1:  # Reduced threshold
                high_out_degree.append((v, out_deg))
        
        if high_out_degree:
            # Sort by highest out-degree
            high_out_degree.sort(key=lambda x: x[1], reverse=True)
            vertex = high_out_degree[0][0]
            
            # Remove a random edge from this vertex
            edges = np.where(G_test[vertex] == 1)[0]
            if len(edges) > 0:
                target = random.choice(edges)
                G_test[vertex, target] = 0
                
                if is_valid_oriented_graph(G_test):
                    test_reward = calculate_reward(G_test)
                    if test_reward > current_reward - 5:  # Allow larger decreases
                        return G_test
    
    # Strategy 3: Try adding edges to vertices with low out-degree
    for _ in range(30):  # Increased from 20
        G_test = G.copy()
        # Find vertices with low out-degree
        low_out_degree = []
        for v in range(N):
            out_deg = np.sum(G_test[v, :])
            if out_deg <= MIN_OUT_DEGREE + 2:  # Increased threshold
                low_out_degree.append((v, out_deg))
        
        if low_out_degree:
            # Sort by lowest out-degree
            low_out_degree.sort(key=lambda x: x[1])
            vertex = low_out_degree[0][0]
            
            # Add a random edge from this vertex
            for target in range(N):
                if target != vertex and G_test[vertex, target] == 0 and G_test[target, vertex] == 0:
                    G_test[vertex, target] = 1
                    if is_valid_oriented_graph(G_test):
                        test_reward = calculate_reward(G_test)
                        if test_reward > current_reward - 3:  # Allow larger decreases
                            return G_test
                    break
    
    # Strategy 4: Try completely random modifications with higher tolerance
    for _ in range(50):  # Try many random modifications
        G_test = G.copy()
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        if i != j:
            if G_test[i, j] == 0 and G_test[j, i] == 0:
                G_test[i, j] = 1
            elif G_test[i, j] == 1 and np.sum(G_test[i, :]) > MIN_OUT_DEGREE:
                G_test[i, j] = 0
        
        G_test = fix_low_outdegree(G_test)
        if is_valid_oriented_graph(G_test):
            test_reward = calculate_reward(G_test)
            if test_reward > current_reward - 8:  # Allow even larger decreases
                return G_test
    
    # If all exploration strategies fail, return a random action
    return apply_random_action(G)

# --- Main loop ---
if __name__ == "__main__":
    INITIAL_VERTICES = 40
    NUM_SIMULATION_STEPS = 1000
    MIN_OUT_DEGREE = 7
    current_G = create_oriented_adjacency_matrix(INITIAL_VERTICES)
    initial_reward = calculate_reward(current_G)
    best_G = current_G.copy()
    best_reward = initial_reward
    current_reward = initial_reward
    print(f"Initial reward: {best_reward:.2f}")
    print(f"Initial vertices: {current_G.shape[0]}")

    steps_without_improvement = 0
    total_steps_at_current_size = 0
    best_reward_at_current_size = initial_reward

    # Simple thresholds for adding a vertex
    ADD_VERTEX_STUCK_STEPS = 100
    ADD_VERTEX_MIN_STEPS = 100

    for step in range(NUM_SIMULATION_STEPS):
        action_choice = random.random()

        # Add a vertex only if thoroughly stuck
        if (
            steps_without_improvement >= ADD_VERTEX_STUCK_STEPS
            and total_steps_at_current_size >= ADD_VERTEX_MIN_STEPS
        ):
            candidate_G = add_vertex_with_edges(current_G)
            candidate_G = fix_low_outdegree(candidate_G)
            if is_valid_oriented_graph(candidate_G):
                candidate_reward = calculate_reward(candidate_G)
                current_G = candidate_G
                current_reward = candidate_reward
                steps_without_improvement = 0
                total_steps_at_current_size = 0
                best_reward_at_current_size = current_reward
                print(f"\nStep {step + 1}: Adding vertex after being stuck. New vertex count: {candidate_G.shape[0]}, New reward: {current_reward:.2f}")
            else:
                # If the new graph is invalid, continue with current graph
                continue
        elif steps_without_improvement >= 100:
            candidate_G = apply_exploration_action(current_G)
        elif action_choice < 0.7:
            candidate_G = apply_smart_action(current_G)
        else:
            candidate_G = apply_random_action(current_G)

        candidate_G = fix_low_outdegree(candidate_G)

        if not is_valid_oriented_graph(candidate_G):
            continue

        candidate_reward = calculate_reward(candidate_G)

        if candidate_reward > current_reward:
            current_G = candidate_G
            current_reward = candidate_reward
            total_steps_at_current_size += 1

            if current_reward > best_reward:
                best_reward = current_reward
                best_G = current_G.copy()
                #print(f"\nStep {step + 1}: New best! Reward: {best_reward:.2f}, Vertices: {best_G.shape[0]}")

            if current_reward > best_reward_at_current_size:
                best_reward_at_current_size = current_reward
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
        else:
            steps_without_improvement += 1
            total_steps_at_current_size += 1

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}: Reward: {current_reward:.2f}, Vertices: {current_G.shape[0]}, Steps without improvement: {steps_without_improvement}, Total steps at current size: {total_steps_at_current_size}")

    print("\nSimulation finished.")
    print(f"Final best reward: {best_reward:.2f}")
    print(f"Final vertices: {best_G.shape[0]}")

    violations, violation_fraction, _, _ = calculate_violation_stats(best_G)
    print(f"Final violation fraction: {violation_fraction:.3f}")

    print("\nFinal graph statistics:")
    print_detailed_stats(best_G)

    if check_conjecture_violation(best_G):
        print("\n🎉 Second Neighborhood Conjecture: VIOLATED!")
        print("All vertices have |N₁(v)| > |N₂(v)|")
    else:
        print("\nSecond Neighborhood Conjecture: Holds")
        print("At least one vertex has |N₁(v)| ≤ |N₂(v)|")

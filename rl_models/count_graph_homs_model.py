import sys
import os
from decimal import Decimal

from sage.all import *
from standard_hom_count import GraphHomomorphismCounter # Assuming standard_hom_count.py is in the same directory or your Python path

# # Define the fixed bipartite graph H as the Mobius strip graph (K5,5 minus a 10-cycle)
H_K55_minus_C10 = graphs.CompleteBipartiteGraph(5,5)
# Edges to remove to form the 10-cycle (ensure both directions for undirected graph)
edges_to_remove = [(0,5),(5,0), (0,6),(6,0), (1,6),(6,1), (1,7),(7,1), (2,7),(7,2), (2,8),(8,2), (3,8),(8,3), (3,9),(9,3), (4,9),(9,4), (4,5),(5,4)]
H_K55_minus_C10.delete_edges(edges_to_remove)
H = H_K55_minus_C10
print(f"Fixed bipartite graph H: {H.graph6_string()}") # Print graph representation

# Initialize a random graph G (e.g., a random graph with 5 vertices and edge probability 0.4)
num_vertices_G = 8
edge_probability_G = 0.5
G = graphs.RandomGNP(num_vertices_G, edge_probability_G)
print(f"Initial random graph G: {G.graph6_string()}") # Print graph representation

def calculate_homomorphism_density(H, G):
    """Calculates the homomorphism density from graph H to graph G."""
    if len(G.vertices()) == 0:
        return 0.0 # Density is 0 if the target graph has no vertices

    counter = GraphHomomorphismCounter(H, G)
    num_homomorphisms = counter.count_homomorphisms()

    # The normalization factor is |V(G)| raised to the power of |V(H)|
    normalization_factor = len(G.vertices())**len(H.vertices())
    
    if normalization_factor == 0:
        return float('inf') # Avoid division by zero if H has no vertices (shouldn't happen with K_{2,3})

    density = Decimal(num_homomorphisms) / Decimal(normalization_factor)
    return density

import random

def apply_random_edge_action(G):
    """Applies a random edge addition or removal action to graph G."""
    vertices = list(G.vertices())
    if len(vertices) < 2:
        # Cannot add/remove edge with less than 2 vertices
        return G.copy() # Return a copy to avoid modifying in place if action is impossible

    action_type = random.choice(['add', 'remove'])
    G_new = G.copy() # Work on a copy

    if action_type == 'add':
        # Try adding a random edge between two distinct vertices
        u, v = random.sample(vertices, 2)
        if not G_new.has_edge(u, v):
            G_new.add_edge(u, v)
            # print(f"Action: Added edge ({u}, {v})")
        # else:
            # print(f"Action: Tried to add edge ({u}, {v}), but it already exists.")
    elif action_type == 'remove':
        # Try removing a random edge
        edges = list(G_new.edges())
        if edges:
            u, v, label = random.choice(edges)
            G_new.delete_edge(u, v)
            # print(f"Action: Removed edge ({u}, {v})")
        # else:
            # print(f"Action: Tried to remove an edge, but the graph has none.")

    return G_new

# Calculate initial density and reward
initial_density = calculate_homomorphism_density(H, G)
initial_reward = ((Decimal(((2 * G.size())) / Decimal((G.order()**2))))**H.size()) - initial_density
print(f"Initial homomorphism density (H->G): {initial_density}")
print(f"Initial reward: {initial_reward}")

# --- Simulation Loop ---
current_G = G.copy()
num_simulation_steps = 1
best_G = current_G.copy()
best_reward = ((Decimal(((2 * G.size())) / Decimal((G.order()**2))))**H.size()) - initial_density

print("\nStarting simulation...")
for step in range(num_simulation_steps):
    # Apply a random action
    next_G = apply_random_edge_action(current_G)

    # Calculate reward for the new graph
    density = calculate_homomorphism_density(H, next_G)
    #reward = -density #why negative density, shouldn't the reward be edge bound - density
    reward = ((Decimal(2 * next_G.size()) / Decimal((next_G.order()**2)))**H.size()) - density

    print(f"\nStep {step + 1}:")
    # print(f"  Graph G after action: {next_G.graph6_string()}") # Optional: print graph string
    print(f"  Homomorphism density (H->G): {density}")
    print(f"  Reward: {reward}")

    # Update best graph if current reward is better
    if reward > best_reward:
        best_reward = reward
        best_G = next_G.copy()
        print("  New best graph found!")

    # Update current graph for the next step
    current_G = next_G.copy()

print("\nSimulation finished.")
print(f"Best graph found (reward: {best_reward}): {best_G.graph6_string()}")
# You can plot the best graph found:
#show(best_G.plot())
# Create a plot object
best_graph_plot = best_G.plot()

# Define a filename for your image
# You can customize the filename, e.g., include the reward or a timestamp
image_filename = "best_graph_reward.png"

# Save the plot to a file
best_graph_plot.save(image_filename)
print(f"Image of the best graph saved to {os.path.abspath(image_filename)}")
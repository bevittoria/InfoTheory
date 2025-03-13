import numpy as np
import networkx as nx
from collections import defaultdict


def compute_transition_matrix(graph):
    """Compute the transition probabilities for a random walk"""
    adjacency = nx.to_numpy_array(graph, dtype=float)  # Convert graph to adjacency matrix
    row_sums = adjacency.sum(axis=1, keepdims=True) # Row sums
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    return adjacency / row_sums  # Normalize rows to get transition probabilities


def simulate_random_walk(graph, transition_matrix, steps=10000):
    """Simulate a long random walk to estimate node visit frequencies"""
    nodes = list(graph.nodes)
    node_to_index = {node: idx for idx, node in enumerate(nodes)}  # Map nodes to their indices
    visit_counts = defaultdict(int) # Define a dictionary to store visit counts
    current_node = np.random.choice(nodes) # Start at a random node
    
    for _ in range(steps):
        visit_counts[current_node] += 1
        neighbors = list(graph.neighbors(current_node)) # Get neighbors of current node
        if neighbors:
            current_index = node_to_index[current_node]  # Find index of current node
            neighbor_indices = [node_to_index[n] for n in neighbors]  # Find indices of neighbors
            probabilities = np.array([transition_matrix[current_index, idx] for idx in neighbor_indices]) # Get transition probabilities
            probabilities /= probabilities.sum()  # Normalize probabilities
            current_node = np.random.choice(neighbors, p=probabilities) # Move to a neighbor based on probabilities
    
    total_visits = sum(visit_counts.values()) # Compute total number of visits for each node
    return {node: visit_counts[node] / total_visits for node in graph.nodes} # Normalize visit counts to get probabilities


def compute_map_equation(graph, transition_matrix, partition):
    """Compute the map equation for the current partitioning"""
    visit_probs = simulate_random_walk(graph, transition_matrix)
    module_probs = defaultdict(float) # Initialize module probabilities
    exit_probs = defaultdict(float) # Initialize exit probabilities

    node_to_index = {node: idx for idx, node in enumerate(graph.nodes)}

    for node, module in partition.items():
        module_probs[module] += visit_probs[node] # Compute module probabilities as sum of visit probabilities
        for neighbor in graph.neighbors(node): 
            if partition[neighbor] != module: # If neighbor is in a different module
                node_idx = node_to_index[node] # Get index of current node
                neighbor_idx = node_to_index[neighbor] # Get index of neighbor
                exit_probs[module] += visit_probs[node] * transition_matrix[node_idx, neighbor_idx] # Compute exit probabilities

    H_modules = -sum(p * np.log2(p) for p in module_probs.values() if p > 0) # Compute entropy of modules
    H_exits = -sum(p * np.log2(p) for p in exit_probs.values() if p > 0) # Compute entropy of exits
    
    return H_exits + H_modules


def optimize_partition(graph, transition_matrix, iterations=10):
    """Optimize the partitioning to minimize the map equation"""
    partition = {node: node for node in graph.nodes}  # Start with each node as its own module
    for _ in range(iterations):
        for node in graph.nodes:
            best_module = partition[node] # Initialize best module as current module
            best_score = compute_map_equation(graph, transition_matrix, partition) 

            for neighbor in graph.neighbors(node): # Try moving node to each neighbor's module
                partition[node] = partition[neighbor] # Move node to neighbor's module
                new_score = compute_map_equation(graph, transition_matrix, partition) # Compute new map equation score
                if new_score < best_score: # If new score is better, update best module and score
                    best_module = partition[node]
                    best_score = new_score

            partition[node] = best_module  # Keep the best move
    return partition


# Generate a synthetic graph
sizes = [50, 50]  # Two communities of 50 nodes each
p_in = 0.1  # Probability of edges within communities
p_out = 0.01  # Probability of edges between communities
G = nx.stochastic_block_model(sizes, [[p_in, p_out], [p_out, p_in]]) # SBM

transition_matrix = compute_transition_matrix(G)
communities = optimize_partition(G, transition_matrix, iterations=10)

print("Number of communities detected:", len(set(communities.values())))
print("Communities detected:")
for node, module in sorted(communities.items()):
    print(f"Node {node} -> Community {module}")

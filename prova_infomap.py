import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt

def compute_transition_matrix(graph):
    """Compute the transition probabilities for a random walk"""
    adjacency = nx.to_numpy_array(graph, dtype=float)  # Convert graph to adjacency matrix
    row_sums = adjacency.sum(axis=1, keepdims=True) # Row sums
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    return adjacency / row_sums  # Normalize rows to get transition probabilities


def simulate_random_walk(graph, transition_matrix, steps=1000):
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
    teleport_term = defaultdict(float) # Initialize teleport probabilities
    non_teleport = defaultdict(float) # Initialize exit probabilities

    tau = 0.15  # Teleport probability
    node_to_index = {node: idx for idx, node in enumerate(graph.nodes)}
    n = len(set(partition.values()))  # Count the number of unique partitions
    q = defaultdict(float)  # Initialize exit probabilities

    for node, module in partition.items():
        module_probs[module] += visit_probs[node] # Compute module probabilities as sum of visit probabilities 
        ni = sum(1 for m in graph.nodes if partition[m] == module)  # Count nodes in the partition
        teleport_term[module] += tau * (n - ni) / (n-1) * visit_probs[node] 
        for neighbor in graph.neighbors(node): 
            if partition[neighbor] != module: # If neighbor is in a different module
                node_idx = node_to_index[node] # Get index of current node
                neighbor_idx = node_to_index[neighbor]  # Get index of neighbor
                non_teleport[module] +=  (1 - tau) * visit_probs[node] * transition_matrix[node_idx, neighbor_idx]  # Compute exit probabilities

    for module in module_probs.keys():
        q[module] = teleport_term[module] + non_teleport[module]  # Compute exit probabilities

    
    def safe_log2(x):
        return np.log2(x) if x > 0 else 0

    '''
    first_term = sum(q[module] for module in module_probs.keys()) * np.log2(sum(q[module] for module in module_probs.keys()))
    second_term = - 2 * sum(q[module]*np.log2(q[module]) for module in module_probs.keys())
    third_term = - sum(visit_probs[node] * np.log2(visit_probs[node]) for node in graph.nodes)
    fourth_term = sum(q[module] *  sum(visit_probs[node] for node in graph.nodes if partition[node] == module) * np.log2(q[module] * sum(visit_probs[node] for node in graph.nodes if partition[node] == module)) for module in module_probs.keys())
    '''
    first_term = sum(q[module] for module in module_probs.keys()) * safe_log2(sum(q[module] for module in module_probs.keys()))
    second_term = - 2 * sum(q[module] * safe_log2(q[module]) for module in module_probs.keys())
    third_term = - sum(visit_probs[node] * safe_log2(visit_probs[node]) for node in graph.nodes)
    fourth_term = sum(q[module] * safe_log2(q[module] + sum(visit_probs[node] for node in graph.nodes if partition[node] == module)) for module in module_probs.keys())

    description_length = first_term + second_term + third_term + fourth_term

    return description_length # Return the map equation


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
                if new_score <= best_score: # If new score is better, update best module and score
                    best_module = partition[node]
                    best_score = new_score

            partition[node] = best_module  # Keep the best move
    return partition


# Generate a synthetic graph
sizes = [20, 10]  # Two communities of 50 nodes each
p_in = 0.5  # Probability of edges within communities
p_out = 0.05  # Probability of edges between communities
G = nx.stochastic_block_model(sizes, [[p_in, p_out], [p_out, p_in]]) # SBM

transition_matrix = compute_transition_matrix(G)
communities = optimize_partition(G, transition_matrix, iterations=10)

print("Number of communities detected:", len(set(communities.values())))
print("Communities detected:")
for node, module in sorted(communities.items()):
    print(f"Node {node} -> Community {module}")


def plot_communities(graph, partition):
    """Plot the network with different colors for different communities"""
    plt.figure(figsize=(10, 7))
    
    # Create a color map for each community
    unique_communities = list(set(partition.values()))
    colors = ["red", "blue", "green", "purple", "orange", "yellow", "black", "gray", "pink", "brown"]
    community_colors = {community: colors[i] for i, community in enumerate(unique_communities)}

    # Assign colors to nodes based on their community
    node_colors = [community_colors[partition[node]] for node in graph.nodes]

    # Draw the graph
    pos = nx.spring_layout(graph, seed=42)  # Layout for visualization
    nx.draw(graph, pos, node_color=node_colors, with_labels=False, node_size=50, edge_color="gray", alpha=0.5)
    
    plt.title("Detected Communities in the Network")
    plt.show()

# Plot the detected communities
plot_communities(G, communities)



def plot_communities_sbm(graph, partition):
    """Plot SBM network with different colors for different communities"""
    plt.figure(figsize=(10, 7))
    
    # Identify unique communities and assign colors using the 'turbo' colormap
    unique_communities = list(set(partition.values()))
    #colors = plt.cm.turbo(np.linspace(0, 1, len(unique_communities)))
    colors = ["red", "blue", "green", "purple", "orange", "yellow", "black", "gray", "pink", "brown"]
    community_colors = {community: colors[i] for i, community in enumerate(unique_communities)}
    
    # Assign colors to nodes based on their community
    node_colors = [community_colors[partition[node]] for node in graph.nodes]

    # Use the multipartite layout for SBM
    pos = nx.multipartite_layout(graph, subset_key="block")

    # Draw the graph
    nx.draw(graph, pos, node_color=node_colors, with_labels=False, node_size=50, edge_color="gray", alpha=0.5)
    
    plt.title("Detected Communities in SBM")
    plt.show()

# Ensure that the nodes have a "block" attribute for multipartite_layout
for i, (block, nodes) in enumerate(enumerate(sizes)):
    for node in range(sum(sizes[:i]), sum(sizes[:i + 1])):
        G.nodes[node]["block"] = i  # Assign block ID to each node

# Plot the detected communities using SBM-specific layout
plot_communities_sbm(G, communities)

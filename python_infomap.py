import networkx as nx
from infomap import Infomap

def run_infomap(G):
    # Load the network
    #G = nx.read_edgelist(edge_list_file, nodetype=int, data=(('weight', float),), create_using=nx.Graph)

    # Initialize Infomap
    im = Infomap()

    # Add edges to Infomap
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1.0)  # Default weight to 1.0 if not provided
        im.add_link(u, v, weight)

    # Run Infomap algorithm
    im.run()

    # Print communities
    communities = {}
    for node in im.iter_tree():
        if node.is_leaf:
            communities[node.physical_id] = node.module_id

    print("Communities detected:")
    for node, community in sorted(communities.items()):
        print(f"Node {node} -> Community {community}")

    return communities

# Example usage
sizes = [50, 50]  # Two communities of 50 nodes each
p_in = 0.1  # Probability of edges within communities
p_out = 0.01  # Probability of edges between communities
G = nx.stochastic_block_model(sizes, [[p_in, p_out], [p_out, p_in]])

run_infomap(G)

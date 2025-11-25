import numpy as np
import networkx as nx

def randomly_distribute_voters(n_voters, n_alternatives, margin):
  # Initialize probability distribution for each voter
  prob_per_voter = np.zeros((n_voters, n_alternatives))

  # Randomly select 50 voters to be "high-repute" (prefer alternative 0 with higher confidence)
  high_repute_voters = np.random.choice(n_voters, 50, replace=False)

  # Assign probabilities based on voter type
  for i in range(n_voters):
      if i in high_repute_voters:
          # High-repute voters: prefer alternative 0 with probability 0.5 + margin
          prob_per_voter[i, 0] = 0.5 + margin
          prob_per_voter[i, 1] = 0.5 - margin
      else:
          # Low-repute voters: prefer alternative 1 with probability 0.5 + margin
          prob_per_voter[i, 0] = 0.5 - margin
          prob_per_voter[i, 1] = 0.5 + margin
  
  return prob_per_voter


def get_centrality_placement(graph, n_high_repute, centrality_type='degree'):
    """
    Select high-repute voters based on centrality measures.
    
    Args:
        graph: adjacency matrix (numpy array)
        n_high_repute: number of high-repute voters to select
        centrality_type: 'degree', 'betweenness', 'pagerank', 'eigenvector'
    
    Returns:
        Array of indices of high-repute voters
    """
    # Remove self-loops for centrality calculation
    graph_no_loops = graph - np.eye(graph.shape[0])
    G = nx.from_numpy_array(graph_no_loops)
    
    if centrality_type == 'degree':
        centrality = nx.degree_centrality(G)
    elif centrality_type == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    elif centrality_type == 'pagerank':
        centrality = nx.pagerank(G)
    elif centrality_type == 'eigenvector':
        try:
            centrality = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            # Fallback to degree centrality if eigenvector doesn't converge
            centrality = nx.degree_centrality(G)
    else:
        raise ValueError(f"Unknown centrality type: {centrality_type}")
    
    # Sort nodes by centrality and select top n_high_repute
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    high_repute_voters = np.array([node for node, _ in sorted_nodes[:n_high_repute]])
    
    return high_repute_voters


def get_peripheral_placement(graph, n_high_repute):
    """
    Select high-repute voters from peripheral (low-degree) nodes.
    
    Args:
        graph: adjacency matrix (numpy array)
        n_high_repute: number of high-repute voters to select
    
    Returns:
        Array of indices of high-repute voters
    """
    # Get nodes with lowest degree centrality
    return get_centrality_placement(graph, n_high_repute, centrality_type='degree')[-n_high_repute:]


def get_clustered_placement(graph, n_high_repute, start_node=None):
    """
    Select high-repute voters in a clustered fashion (connected subgraph).
    
    Args:
        graph: adjacency matrix (numpy array)
        n_high_repute: number of high-repute voters to select
        start_node: optional starting node for BFS (if None, uses highest degree node)
    
    Returns:
        Array of indices of high-repute voters
    """
    graph_no_loops = graph - np.eye(graph.shape[0])
    G = nx.from_numpy_array(graph_no_loops)
    
    if start_node is None:
        # Start from highest degree node
        degrees = dict(G.degree())
        start_node = max(degrees.items(), key=lambda x: x[1])[0]
    
    # BFS to get connected cluster
    visited = set([start_node])
    queue = [start_node]
    
    while len(visited) < n_high_repute and queue:
        current = queue.pop(0)
        neighbors = list(G.neighbors(current))
        for neighbor in neighbors:
            if neighbor not in visited and len(visited) < n_high_repute:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return np.array(list(visited))


def get_random_placement(n_voters, n_high_repute):
    """
    Randomly select high-repute voters.
    
    Args:
        n_voters: total number of voters
        n_high_repute: number of high-repute voters to select
    
    Returns:
        Array of indices of high-repute voters
    """
    return np.random.choice(n_voters, n_high_repute, replace=False)


def create_prob_distribution(n_voters, high_repute_voters, pH, pL, n_alternatives=2):
    """
    Create probability distribution for voters based on high/low repute assignment.
    
    Args:
        n_voters: total number of voters
        high_repute_voters: array of indices of high-repute voters
        pH: probability that high-repute voters choose correct alternative
        pL: probability that low-repute voters choose correct alternative
        n_alternatives: number of alternatives (default 2)
    
    Returns:
        prob_per_voter: n_voters x n_alternatives array of probabilities
    """
    prob_per_voter = np.zeros((n_voters, n_alternatives))
    
    for i in range(n_voters):
        if i in high_repute_voters:
            prob_per_voter[i, 0] = pH
            prob_per_voter[i, 1] = 1 - pH
        else:
            prob_per_voter[i, 0] = pL
            prob_per_voter[i, 1] = 1 - pL
    
    return prob_per_voter


def cluster_voters():
  pass

def centralize_voters():
  pass
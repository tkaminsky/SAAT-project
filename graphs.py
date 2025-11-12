import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

# returns network x adjacency matrix - tells how to make each function

graph_dict = {}

def make_small_world(n, k, p):
    sw = nx.watts_strogatz_graph(n, k, p)
    return nx.adjacency_matrix(sw).todense() + np.eye(n)


def make_line(n):
    line = nx.path_graph(n)
    return nx.adjacency_matrix(line).todense() + np.eye(n)


def make_ring(n):
    ring = nx.cycle_graph(n)
    return nx.adjacency_matrix(ring).todense() + np.eye(n)


def make_barbell(n):
    barbell = nx.barbell_graph(n // 2, n // 2)
    return nx.adjacency_matrix(barbell).todense() + np.eye(n)


def make_ER(n, p):
    er = nx.erdos_renyi_graph(n - 1, p)
    # While not connected, keep trying
    while not nx.is_connected(er):
        er = nx.erdos_renyi_graph(n - 1, p)

    # Attach a node to the graph with edge probability p
    er.add_node(n - 1)
    valid = False
    while not valid:
        edge_set = []
        for i in range(n - 1):
            if random.random() < p:
                edge_set.append(i)
        if len(edge_set) > 0:
            valid = True

    for i in edge_set:
        er.add_edge(i, n - 1)

    return nx.adjacency_matrix(er).todense() + np.eye(n)


# Make a 'barbell' of 3 fully-connected regions of size n, each connected by a line of size k
def make_tribell(n, k):
    G_left = nx.complete_graph(n)
    G_right = nx.complete_graph(n)
    G_mid = nx.complete_graph(n)
    line_left_mid = nx.path_graph(k)
    line_mid_right = nx.path_graph(k)
    G = nx.disjoint_union(G_left, line_left_mid)
    G = nx.disjoint_union(G, G_mid)
    G = nx.disjoint_union(G, line_mid_right)
    G = nx.disjoint_union(G, G_right)
    G.add_edge(n - 1, n)
    G.add_edge(n + k - 1, n + k)
    G.add_edge(2 * n + k - 1, 2 * n + k)
    G.add_edge(2 * n + 2 * k - 1, 2 * n + 2 * k)

    return nx.adjacency_matrix(G).todense() + np.eye(3 * n + 2 * k)


def make_Rbell(R, n, k):
    assert R >= 1
    G = nx.complete_graph(n)
    for i in range(R - 1):
        if k != 0:
            G = nx.disjoint_union(G, nx.path_graph(k))
            G.add_edge(n * (i + 1) + i * k - 1, n * (i + 1) + i * k)
        G = nx.disjoint_union(G, nx.complete_graph(n))
        G.add_edge(n * (i + 1) + (i + 1) * k - 1, n * (i + 1) + (i + 1) * k)
    return nx.adjacency_matrix(G).todense() + np.eye(R * n + (R - 1) * k)

def make_generic_bell(ns, ks):
    assert len(ns) == len(ks) + 1 # CHECKME: is this + or - 1?
    R = len(ns)
    G = nx.complete_graph(ns[0])
    for i in range(R - 1):
        if ks[i] != 0:
            G = nx.disjoint_union(G, nx.path_graph(ks[i]))
            G.add_edge(sum(ns[:i + 1]) + sum(ks[:i]) - 1, sum(ns[:i + 1]) + sum(ks[:i]))
        G = nx.disjoint_union(G, nx.complete_graph(ns[i + 1]))
        G.add_edge(sum(ns[:i + 1]) + sum(ks[:i + 1]) - 1, sum(ns[:i + 1]) + sum(ks[:i + 1]))

    print(nx.adjacency_matrix(G).shape)
    return nx.adjacency_matrix(G).todense() + np.eye(sum(ns) + sum(ks))


def get_graph(key):
    split_key = key.split("_")
    if split_key[0] == "line":
        return make_line(int(split_key[1]))
    elif split_key[0] == "ring":
        return make_ring(int(split_key[1]))
    elif split_key[0] == "barbell":
        return make_barbell(int(split_key[1]))
    elif split_key[0] == "ER":
        n = int(split_key[1])
        p = float(split_key[2])
        return make_ER(n, p)
    elif split_key[0] == "tribell":
        n = int(split_key[1])
        if len(split_key) == 2:
            k = n // 2
        else:
            k = int(split_key[2])
        return make_tribell(n, k)
    elif split_key[0] == "Rbell":
        R = int(split_key[1])
        n = int(split_key[2])
        k = int(split_key[3])
        return make_Rbell(R, n, k)
    else:
        raise ValueError("Graph not found. Please check the key.")


def plot_graph(graph):
    # Remove diagonal weights
    graph_no_loops = graph - np.eye(graph.shape[0])
    G = nx.from_numpy_array(graph_no_loops)
    # Plot the graph, making the last node red
    colors = ["blue" for i in range(graph.shape[0] - 1)] + ["red"]
    nx.draw(G, node_color=colors, with_labels=True)
    plt.show()

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.integrate import quad
from scipy.stats import gamma
import scipy.special as sc

def draw_graph(G):
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

    pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )

    # node labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def find_next_child(G, source_node):
    child_node = None
    max_weight = 0
    for node in source_node:
        for neighbor in G.neighbors(node):
            #print(f"neighbor is {neighbor}")
            if neighbor not in source_node:
                weight = G[node][neighbor]['weight']
                if weight > max_weight:
                    max_weight = weight
                    child_node = neighbor
                    #print(f"child node is {child_node}")
    return child_node

def find_parents(i, G, source=['s']):
    parent_node = []
    # First get all the neighbors of node i
    candidate = G.neighbors(i)
    for n in candidate:
        # Find the nodes in the source set
        if n in source:
            parent_node.append(n)
    return parent_node

# We use theta instead of r in the paper.
def compute_ccdf(x, k, theta):
    return 1 - gamma.cdf(x, k, scale=theta)

I = ['s']
def product_ccdf(t, j, k, theta):
    v = 1
    for par in find_parents(j, G, source=I):
        v *= compute_ccdf(t, k[par], theta[par])
    return v

def integrand_moment(t, j, k, theta, p):
    return p * t**(p - 1) * product_ccdf(t, j, k, theta)

def compute_moment(j, k, theta, p):
    return quad(integrand_moment, 0, np.inf, args=(j, k, theta, p))[0]

# Calculate k and r for node a
def compute_theta(E, V, lambda_p):
    return (E + 1/lambda_p) / (V + 1/lambda_p**2)

def compute_k(E, V, lambda_p):
    return (E + 1/lambda_p)**2 / (V + 1/lambda_p**2)

def compute_expectation(k1, r1, k2, r2):
  beta_top = sc.betainc(k2+1, k1, r2/(r1+r2))
  beta_down = sc.betainc(k2, k1, r2/(r1+r2))
  expectation = (k1+k2) / r1 - k2*(r1+r2) / (r1*r2) * (beta_top / beta_down)
  return 1/expectation
# %%
import numpy as np
import networkx as nx
from scipy.stats import gamma
from scipy.integrate import quad, romberg, quadrature

# %%
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

# %%
def compute_ccdf(x, k, theta):
    return 1 - gamma.cdf(x, k, scale=theta)

I = ['s']
def product_ccdf(x, j, k, theta):
    v = 1
    for par in find_parents(j, G, source=I):
        v *= compute_ccdf(x, k[par], theta[par])
    return v

def integrand_moment(x, j, k, theta, p):
    return p * x**(p - 1) * product_ccdf(x, j, k, theta)

def compute_moment(j, k, theta, p):
    return quad(integrand_moment, 0, np.inf, limit=1000000, args=(j, k, theta, p))[0]

# %%
def compute_moment2(j, k, theta, p):
    return romberg(integrand_moment, 0, 10000000, args=(j, k, theta, p))

# %%
G = nx.Graph()
G.add_nodes_from(["s", "a", "b", "c", "d", "e"])
G.add_edge("s", "a", weight=0.5)
G.add_edge("s", "b", weight=0.4)
G.add_edge("s", "d", weight=0.3)
G.add_edge("s", "e", weight=0.1)
G.add_edge("a", "b", weight=0.1)
G.add_edge("b", "c", weight=0.1)
G.add_edge("b", "e", weight=0.6)

theta_s = 13
k_s = 10
E_s = k_s * theta_s
V_s = k_s * theta_s**2

j = 'a'
k = {'s': k_s}
theta = {'s': theta_s}

moment_1 = compute_moment('s', k, theta, 1)
moment_2 = compute_moment('s', k, theta, 2)

# %%
moment2 = compute_moment2('s', k, theta, 1)
print(moment2)

# %%
def ccdf1(x):
    return (1- gamma.cdf(x, 1, scale=1)) * (1 - gamma.cdf(x, 2, scale=1))

quad(ccdf1, 0, np.inf)[0]



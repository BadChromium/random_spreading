import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import numpy as np
import ipywidgets as widgets
from scipy.integrate import quad
from scipy.stats import gamma
import scipy.special as sc
import copy
import math

def draw_graph(G):
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.1]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.1]

    pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=3)
    nx.draw_networkx_edges(
        G, pos, edgelist=esmall, width=3, alpha=0.5, edge_color="b", style="dashed"
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

def product_ccdf(t, j, k, theta, parents):
    v = 1
    for par in parents:
        v *= compute_ccdf(t, k[par], theta[par])
    return v

def integrand_moment(t, j, k, theta, p, parents):
    return p * t**(p - 1) * product_ccdf(t, j, k, theta, parents)

def compute_moment(j, k, theta, p, parents):
    return quad(integrand_moment, 0, np.inf, args=(j, k, theta, p, parents))[0]

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

def MP_weighted(G, nodes_infected, T_max=5, num_T_positions=10):
    """
    Args:
        G:  an undirected graph constructed by networkx, with edge property 'weight' storing the edge weight.
        nodes_infected: a set of infected nodes.
        T_max: upbound of observation/spreading time considered
        num_T_positions: used for decides T_positions=[i for i in range(1, num_T_positions+1)], and the optimization will search over observation time [T_max*t_pos/num_T_positions for t_pos in T_positions]
    Returns:
        Output the the set of the nodes that has the largest likelihood to be the source, one of which the index of the optimal ovservation time in T_positions is also provided.  
        An ipywidgets is provided to show the change of most likely source against the time indexed by T_positions.
    
    """
       
    
    #### draw the graph G
    pos_SEED=0
    
    edge_labels = nx.get_edge_attributes(G,'weight')
    # Set the node colors
    node_colors = ['gray' if i in nodes_infected else 'white' for i in range(G.number_of_nodes())]

    # Draw the graph
    pos = nx.spring_layout(G, seed=pos_SEED)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, edgecolors= 'black')
    nx.draw_networkx_labels(G, pos, font_size=12)
    nx.draw_networkx_edges(G, pos, width=2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    plt.axis('off')
    plt.show()
    
    
    #### Construct the parent and child relation between nodes after selecting a node as source
    
    source=list(nodes_infected)[0]
    parent={} # 'key:value' pair where 'value' is the parent node for node 'key'
    child={} #'key:value' pair where 'value' is the set of child nodes for node 'key'
    depth={} # 
    nodes_waiting_child=set() # nodes haven't assigned child
    b_uninfected=set() #susceptible nodes at final observation; uninfected nodes which have infected neighbors
    b_infected=set() #infected nodes which have uninfected neighbors
    leaf_infected=set() # the leaf infected nodes

    parent[source]=None
    depth[source]=0
    nodes_waiting_child.add(source)
    while len(nodes_waiting_child)!=0:
        i=nodes_waiting_child.pop()
        child[i]= set(j for j in G.neighbors(i) if j not in parent.keys())
        is_leaf = True
        for j in child[i]:
            parent[j]=i
            if j in nodes_infected:
                depth[j]=depth[i]+1
                nodes_waiting_child.add(j)
                is_leaf = False
            else:
                b_uninfected.add(j)
                b_infected.add(i)
        if is_leaf==True:
            leaf_infected.add(i)    
            
    #### get some variable aiding the judgement whether a node received all messages from its child nodes
    static_num_child_not_ready={}
    static_nodes_ready=set() # nodes whose children nodes all can be ready with messages, i.e. nodes that have no child nodes 
    static_nodes_received=set()

    # only deal with infected nodes, since uninfected nodes in b_uninfected is always ready
    for i in nodes_infected:

        if len(child[i])==0:
            static_num_child_not_ready[i]=0
            static_nodes_ready.add(i)
        else:
            static_num_child_not_ready[i]=len(child[i])
            
    
    ## Discrete FFT for convolution of a and b
    def rfft_convolve(a, b):
        ''' return discrete convolution array of arrays a and b by fft
        '''
        #here we only deal with the case len(a)==len(b)
        if len(a)!=len(b):
            print("Warning. The arrays a and b in fft has different lenghth")
        n = len(a) + len(b) -1
        N = 2 ** (int(np.log2(n))+1)
        A = np.fft.rfft(a, N)
        B = np.fft.rfft(b, N)
        return np.fft.irfft(A*B)[:(n+1)//2]
    
    #### message passing for the source node given by initiallization 
    #### T_max=5

    ####T_positions=[i for i in range(1,11)]
    T_positions=[i for i in range(1, num_T_positions+1)]

    message=[None]*(len(T_positions)+1)
    result=[None]*(len(T_positions)+1)


    for t_pos in T_positions:
        num_child_not_ready=static_num_child_not_ready.copy()
        nodes_ready=static_nodes_ready.copy()
        nodes_received=static_nodes_received.copy()

        message[t_pos]={}
        result[t_pos]={}

        T=T_max*t_pos/len(T_positions) # total time, or observation time
        num_intervals=20
        t_of_T=[T/num_intervals*i for i in range(0, num_intervals+1)]

        # currently nodes_ready are infected nodes with no child nodes
        nodes_no_child=nodes_ready.copy()
        nodes_received=nodes_ready.copy()
        nodes_ready=set()
        while len(nodes_no_child)!=0: 
            i=nodes_no_child.pop()
            message[t_pos][i]=[1-math.exp(-G[i][parent[i]]['weight'] * t_of_T[j]) for j in range(0, num_intervals+1)]
            i_p=parent[i]
            num_child_not_ready[i_p]-=1
            if num_child_not_ready[i_p] == 0:
                nodes_ready.add(i_p)

        for i in b_uninfected:
            message[t_pos][i]=[math.exp(-G[i][parent[i]]['weight'] * t_of_T[j]) for j in range(0, num_intervals+1)]
            i_p=parent[i]
            num_child_not_ready[i_p]-=1
            if num_child_not_ready[i_p] == 0:
                nodes_ready.add(i_p)

        while len(nodes_ready)!=0:
            i=nodes_ready.pop()
            nodes_received.add(i)
            message[t_pos][i]=[]

            if parent[i] is not None:
                child_product=np.array([1]*len(t_of_T))
                for k in child[i]:
                    child_product=np.multiply(child_product, np.array(message[t_pos][k]))
                i_pdf=[G[i][parent[i]]['weight'] * math.exp(-G[i][parent[i]]['weight'] * t_of_T[j]) for j in range(0, num_intervals+1)]
                for j in range(0, num_intervals+1):
                    ## not use fft
                    message[t_pos][i].append(np.sum(np.multiply(child_product[::-1][:j+1], np.array(i_pdf)[:j+1]))*T/num_intervals)
                    ##use fft  
                    # message[t_pos][i].append(rfft_convolve(child_product[num_intervals-j:], i_pdf[:j+1])[::-1][0]*T/num_intervals)


                i_p=parent[i]
                num_child_not_ready[i_p]-=1
                if num_child_not_ready[i_p] == 0:
                    nodes_ready.add(i_p)    
            else:
                root_product=1
                for k in child[i]:
                    if k in nodes_infected:
                        root_product=root_product*np.sum(message[t_pos][k])*T/num_intervals
                    else:
                        root_product=root_product*message[t_pos][k][num_intervals] # because for susceptible nodes in b_uninfected, we calculated its CDF in its message[t_pos]

                result[t_pos][source]=root_product.copy()
                
                
    
    ## To compute likelihood probability for other nodes 
    message_from_parent=[None]*(len(T_positions)+1)


    for t_pos in T_positions:
        ## do not forgot the T and t_of_T when "for t_pos in T_positions" is separated to two part
        T=T_max*t_pos/len(T_positions) # total time
        t_of_T=[T/num_intervals*i for i in range(0, num_intervals+1)]


        message_from_parent[t_pos]={}
        nodes_to_compute=[i for i in child[source] if i in nodes_infected]
        while len(nodes_to_compute)!=0:
            i=nodes_to_compute.pop()
            i_p=parent[i]

            message_from_parent[t_pos][i]=[]
            # calculate the product of messages from neighbors of i_p except i 
            child_product=np.array([1]*len(t_of_T))
            child_i_p_exclude_i = child[i_p].copy()
            child_i_p_exclude_i.remove(i)
            for k in child_i_p_exclude_i:
                child_product=np.multiply(child_product, np.array(message[t_pos][k]))
            # parent[i_p] is not None means i is not connected to the source
            if parent[i_p] is not None:
                child_product=np.multiply(child_product, np.array(message_from_parent[t_pos][i_p]))

            # message_from_parent[t_pos][i] is the message i_p should send to i
            i_pdf=[G[i][i_p]['weight'] * math.exp(-G[i][i_p]['weight'] * t_of_T[j]) for j in range(0, num_intervals+1)]
            for j in range(0, num_intervals+1):
                ## not use fft
                message_from_parent[t_pos][i].append(np.sum(np.multiply(child_product[::-1][:j+1], np.array(i_pdf)[:j+1]))*T/num_intervals)
                ##use fft  
                # message_from_parent[t_pos][i].append(rfft_convolve(child_product[num_intervals-j:], i_pdf[:j+1])[::-1][0]*T/num_intervals)            

            # calculate the probability as if i is the root (source), for message from i_p, should use message_from_parent[t_pos][i]
            root_product=1
            root_product=root_product*np.sum(message_from_parent[t_pos][i])*T/num_intervals
            for k in child[i]:
                if k in nodes_infected:
                    root_product=root_product*np.sum(message[t_pos][k])*T/num_intervals
                else:
                    root_product=root_product*message[t_pos][k][num_intervals] # because for susceptible nodes in b_uninfected, we calculated its CDF in its message
            # print(t_pos, len(result),result[t_pos])
            result[t_pos][i]=root_product.copy()

             # nodes_no_result.remove(i)
            nodes_to_compute=nodes_to_compute + [j for j in child[i] if j in nodes_infected]                
                
                
    
    #### processing the results
    ## store the likelyhood with nodes as keys of a dict. 'result_by_node[i][j]' is the likehood for node i to be the source when observation time is the T_positions[j]-th observation time
    result_by_node={}
    for i in nodes_infected:
        result_by_node[i]=[]
        for t_pos in T_positions:
            result_by_node[i].append(result[t_pos][i]) # pay attention that the index 0 correspond to t_pos=1

    ## 'max_nodes_t_pos' is a dict; max_nodes_t_pos[i][j] is an integer such that at max_nodes_t_pos[i][j]-th observation time is when node i achives largest likelyhood probability to be the source over all the ovservation times
    ## 'max_node' are the node has the largest likelyhood probability to be the source, and 'max_value' is the value.
    max_nodes_t_pos={}
    max_value=0
    max_nodes=set()

    for k,v in result_by_node.items():
        max_index=np.argmax(v)
        max_nodes_t_pos[k]=max_index+1
        if np.isclose(v[max_index],max_value):
            max_nodes.add(k)
        elif v[max_index]>max_value:               
            max_value=v[max_index]
            max_nodes={k}

    # print(max_nodes_t_pos)
    print('The most likely source nodes are:', max_nodes, '\n', 'the corresponding likelyhood is', max_value, '\n', 
              'the index of the ovservation time correspond to node', list(max_nodes)[0], ' is ', max_nodes_t_pos[list(max_nodes)[0]])

    ## result_by_t_pos[i][j] is the likelyhood probability for node j to be the source at i-th observation time.
    result_by_t_pos={}
    # pay attention that the index 0 correspond to t_pos=1
    for t_pos in T_positions:
        result_by_t_pos[t_pos]=[]
        for k in G.nodes():
            result_by_t_pos[t_pos].append(result_by_node[k][t_pos-1] if k in nodes_infected else 0)

    colors_dict=copy.deepcopy(result_by_t_pos)
    norm = mpl.colors.Normalize(vmin=0, vmax=max_value, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Greens)

    def plot_result_by_t_pos(t_pos_for_plot):
        nx.draw(G, pos=pos, node_color=[mapper.to_rgba(i) for i in colors_dict[t_pos_for_plot]], with_labels=True)
        plt.show()


    widgets.interact(plot_result_by_t_pos,t_pos_for_plot=widgets.IntSlider(min=1, max=max(T_positions), step=1, value=1,description="index of the observation time",style={'description_width': 'initial'},layout=widgets.Layout(width='500px')))

    print("the darker the node color, the larger probability for the node to be the source")
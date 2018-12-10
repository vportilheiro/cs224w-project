import networkx as nx
import random

def read_graph(graph_name, sep=",", nodetype=str, temporal=False):
    '''
    read_graph(graph_name)
    
    params:
        graph_name: string of csv file of edge list, in format: u, v, weight(u,v) [, time(u,v)]
        sep: the separator of columns in the file (default=",")
        temporal: whether the extra timestamp column exists in the file
        
    return:
        G: networkx DiGraph with edge weights (and possibly edge times)
    '''
    G = nx.DiGraph()
    with open(graph_name,"r") as f:
        for line in f:
            line = line.strip()
            if line == "" or line[0] == "#": continue
            l = line.split(sep)
            if temporal:
                G.add_edge(nodetype(l[0]), nodetype(l[1]), weight = float(l[2]), time = float(l[3]))
            else:
                G.add_edge(nodetype(l[0]), nodetype(l[1]), weight = float(l[2]))
    return G

def remove_random_edges(G, num_edges):
    '''
    remove_random_edges(G, num_edges)
    
    Randomly removes num_edges edges from G, and returns the list
    of removed edges and a list of their corresponding edge weights.
    
    params: 
        G: networkx graph with edge weights stored under 'weight' attribute
        num_edges: number of edges to remove
    
    return:
        remove_list: list of tuples of removed edges
        removed_weights: list of weights of removed edges
    '''
    remove_list = random.sample(list(G.edges()), num_edges)
    removed_weights = [G.get_edge_data(*edge)['weight'] for edge in remove_list]
    G.remove_edges_from(remove_list)
    return remove_list, removed_weights
'''
Code for the paper:
Edge Weight Prediction in Weighted Signed Networks. 
Conference: ICDM 2016
Authors: Srijan Kumar, Francesca Spezzano, VS Subrahmanian and Christos Faloutsos

Author of code: Srijan Kumar
Email of code author: srijan@cs.stanford.edu
'''

import networkx as nx
import math

def initialize_scores(G):
    fairness = {}
    goodness = {}
    
    nodes = G.nodes()
    for node in nodes:
        fairness[node] = 1
        try:
            goodness[node] = G.in_degree(node, weight='weight')*1.0/G.in_degree(node)
        except:
            goodness[node] = 0
    return fairness, goodness

def compute_fairness_goodness(G, max_iters = 100, tol=1e-6, quiet=False):
    fairness, goodness = initialize_scores(G)
    
    nodes = G.nodes()
    iter = 0
    while iter < max_iters:
        df = 0
        dg = 0

        if not quiet:
            print '-----------------'
            print "Iteration number", iter
            
            print 'Updating goodness'
            
        for node in nodes:
            inedges = G.in_edges(node, data='weight')
            g = 0
            for edge in inedges:
                g += fairness[edge[0]]*edge[2]

            try:
                dg += abs(g/len(inedges) - goodness[node])
                goodness[node] = g/len(inedges)
            except:
                pass

        if not quiet:
            print 'Updating fairness'
        for node in nodes:
            outedges = G.out_edges(node, data='weight')
            f = 0
            for edge in outedges:
                f += 1.0 - abs(edge[2] - goodness[edge[1]])/2.0
            try:
                df += abs(f/len(outedges) - fairness[node])
                fairness[node] = f/len(outedges)
            except:
                pass
        
        if not quiet:
            print 'Differences in fairness score and goodness score = %.2f, %.2f' % (df, dg)
        if df < tol and dg < tol:
            break
        iter+=1
    
    return fairness, goodness

if __name__ == '__main__':
    skip = int(sys.argv[1])

    G = nx.DiGraph()

    f = open("network.csv","r")
    for l in f:
        ls = l.strip().split(",")
        G.add_edge(ls[0], ls[1], weight = float(ls[2])) ## the weight should already be in the range of -1 to 1
    f.close()


    # these two dictionaries have the required scores
    fairness, goodness = compute_fairness_goodness(G)


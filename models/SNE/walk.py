# -*- coding: utf-8 -*-

"""Graph utilities."""

import logging
import sys
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
import random


logger = logging.getLogger("deepwalk")

__author__ = "Bryan Perozzi"
__email__ = "bperozzi@cs.stonybrook.edu"

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def random_walk(G, path_length, alpha=0, rand=random.Random(), start=None, temporal=False):
    """ Returns a truncated random walk.
        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
        temporal: whether to use "time" edge attribute to only generate a temporal walk
    """
    if start:
        path = [start]
    else:
        # Sampling is uniform w.r.t V, and not w.r.t E
        path = [rand.choice(G.nodes())]

    if temporal:
        u = path[-1]
        times = [G[u][v]['time'] for v in G.neighbors(u)]
        if times:
            t = rand.choice(times)
        else:
            return path

    while len(path) < path_length:
        cur = path[-1]
        if len(G[cur]) > 0:
            if rand.random() >= alpha:
                if temporal:
                    node_and_time = [(v,G[cur][v]['time']) for v in G.neighbors(cur) if G[cur][v]['time'] > t]
                    if node_and_time:
                        next_node, t = rand.choice(node_and_time)
                        path.append(next_node)
                    else:
                        break
                else:
                    path.append(rand.choice(list(G[cur].keys())))
            else:
                path.append(path[0])
        else:
            break
    return path


# TODO add build_walks in here

def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                          rand=random.Random(0), temporal=False):
    walks = []

    nodes = list(G.nodes())

    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(random_walk(G, path_length, rand=rand, alpha=alpha, start=node, temporal=temporal))

    return walks


def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
                               rand=random.Random(0), temporal=False):
    nodes = list(G.nodes())

    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            yield random_walk(G, path_length, rand=rand, alpha=alpha, start=node, temporal=temporal)

def write_walks_to_disk(G, f, num_paths, path_length, alpha=0, rand=random.Random(1024), temporal=False):
    # G, f, num_paths, path_length, alpha, rand = args
    t_0 = time()
    with open(f, 'w') as fout:
        for walk in build_deepwalk_corpus_iter(G=G, num_paths=num_paths, path_length=path_length,
                                                     alpha=alpha, rand=rand, temporal=temporal):
          fout.write(u"{}\n".format(u" ".join(str(v) for v in walk)))
    logger.debug("Generated new file {}, it took {} seconds".format(f, time() - t_0))
    return f

#number_walks = 20
#walk_length = 40
#G = load_edgelist('data/wiki_edit.txt')
#f = write_walks_to_disk(G, f='data/wiki_edit_num_40.walk', num_paths=number_walks, path_length=walk_length, alpha=0, rand=random.Random(1024))

from collections import defaultdict
import numpy as np
import gensim
from joblib import Parallel, delayed
from tqdm import tqdm
from .parallel import parallel_generate_walks


class Node2Vec:
    FIRST_TRAVEL_KEY = 'first_travel_key'
    PROBABILITIES_KEY = 'probabilities'
    NEIGHBORS_KEY = 'neighbors'
    WEIGHT_KEY = 'weight'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'
    TIME_KEY = 'time'
    REL_WEIGHT_KEY = 'rel_weight'
    TOLERANCE = 1e-5

    def __init__(self, graph, dimensions=128, walk_length=80, num_walks=10, p=1, q=1, weight_key='weight',
                 workers=1, sampling_strategy=None, quiet=False, temporal=False, time_key='time',
                 relational_weighting=False, trans_factor=1):
        """
        Initiates the Node2Vec object, precomputes walking probabilities and generates the walks.
        :param graph: Input graph
        :type graph: Networkx Graph
        :param dimensions: Embedding dimensions (default: 128)
        :type dimensions: int
        :param walk_length: Number of nodes in each walk (default: 80)
        :type walk_length: int
        :param num_walks: Number of walks per node (default: 10)
        :type num_walks: int
        :param p: Return hyper parameter (default: 1)
        :type p: float
        :param q: Inout parameter (default: 1)
        :type q: float
        :param weight_key: On weighted graphs, this is the key for the weight attribute (default: 'weight')
        :type weight_key: str
        :param workers: Number of workers for parallel execution (default: 1)
        :type workers: int
        :param sampling_strategy: Node specific sampling strategies, supports setting node specific 'q', 'p', 'num_walks' and 'walk_length'.
        Use these keys exactly. If not set, will use the global ones which were passed on the object initialization
        :param temporal: Whether the graph has a 'time' attribute for edges, and this attribute is to be used to generate only temporal walks.
        :type temporal: bool
        :param time_key: key in given graph for edge times
        :type time_key: str
        """
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.weight_key = weight_key
        self.workers = workers
        self.quiet = quiet
        self.temporal = temporal
        self.time_key = time_key
        self.relational_weighting = relational_weighting
        self.trans_factor = trans_factor

        if sampling_strategy is None:
            self.sampling_strategy = {}
        else:
            self.sampling_strategy = sampling_strategy

        self.d_graph = self._precompute_probabilities()

        #if self.relational_weighting:
        #    self.walks, self.rel_weights_by_walk = self._generate_walks()
        #else:
        self.walks = self._generate_walks()

    def _precompute_probabilities(self):
        """
        Precomputes transition probabilities for each node.
        """
        d_graph = defaultdict(dict)
        first_travel_done = set()

        nodes_generator = self.graph.nodes() if self.quiet \
            else tqdm(self.graph.nodes(), desc='Computing transition probabilities')

        for source in nodes_generator:

            # Init probabilities dict for first travel
            if self.PROBABILITIES_KEY not in d_graph[source]:
                d_graph[source][self.PROBABILITIES_KEY] = dict()

            for current_node in self.graph.neighbors(source):

                # Init probabilities dict
                if self.PROBABILITIES_KEY not in d_graph[current_node]:
                    d_graph[current_node][self.PROBABILITIES_KEY] = dict()

                if self.REL_WEIGHT_KEY not in d_graph[current_node]:
                    d_graph[current_node][self.REL_WEIGHT_KEY] = dict()

                unnormalized_weights = list()
                first_travel_weights = list()
                d_neighbors = list()
                if self.temporal:
                    d_times = list()

                # Calculate unnormalized weights
                for destination in self.graph.neighbors(current_node):

                    p = self.sampling_strategy[current_node].get(self.P_KEY,
                                                                 self.p) if current_node in self.sampling_strategy else self.p
                    q = self.sampling_strategy[current_node].get(self.Q_KEY,
                                                                 self.q) if current_node in self.sampling_strategy else self.q

                    if self.relational_weighting:
                        temp = self.weight_key
                        self.weight_key = "x"
        
                    if destination == source:  # Backwards probability
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / p
                    elif destination in self.graph[source]:  # If the neighbor is connected to the source
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1)
                    else:
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / q

                                        # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)
                    if current_node not in first_travel_done:
                        first_travel_weights.append(self.graph[current_node][destination].get(self.weight_key, 1))
                    d_neighbors.append(destination)
                    if self.temporal:
                        d_times.append(self.graph[current_node][destination][self.time_key])

                    if self.relational_weighting:
                        self.weight_key = temp
                        d_graph[current_node][self.REL_WEIGHT_KEY][destination] = self.graph[current_node][destination][self.weight_key]

                # Normalize
                unnormalized_weights = np.array(unnormalized_weights)
                d_graph[current_node][self.PROBABILITIES_KEY][
                    source] = unnormalized_weights / unnormalized_weights.sum()

                if current_node not in first_travel_done:
                    unnormalized_weights = np.array(first_travel_weights)
                    d_graph[current_node][self.FIRST_TRAVEL_KEY] = unnormalized_weights / unnormalized_weights.sum()
                    first_travel_done.add(current_node)

                # Save neighbors
                d_graph[current_node][self.NEIGHBORS_KEY] = d_neighbors
                if self.temporal:
                    d_graph[current_node][self.TIME_KEY] = d_times

        return d_graph

    def _generate_walks(self):
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        flatten = lambda l: [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)

        walk_results = Parallel(n_jobs=self.workers)(delayed(parallel_generate_walks)(self.d_graph,
                                                                                      self.walk_length,
                                                                                      len(num_walks),
                                                                                      idx,
                                                                                      self.sampling_strategy,
                                                                                      self.NUM_WALKS_KEY,
                                                                                      self.WALK_LENGTH_KEY,
                                                                                      self.NEIGHBORS_KEY,
                                                                                      self.PROBABILITIES_KEY,
                                                                                      self.FIRST_TRAVEL_KEY,
                                                                                      self.temporal,
                                                                                      self.TIME_KEY,
                                                                                      self.relational_weighting,
                                                                                      self.REL_WEIGHT_KEY,
                                                                                      self.quiet) for
                                                     idx, num_walks
                                                     in enumerate(num_walks_lists, 1))
        
        #if self.relational_weighting:
            #walk_results, rel_weights_by_walk = list(zip(*walk_results))

        walks = flatten(walk_results)

        #if self.relational_weighting:
            #rel_weights_by_walk = flatten(rel_weights_by_walk)

        #if self.relational_weighting:
            #for i, walk in enumerate(walks):
            #    if (len(walk)-1 != len(rel_weights_by_walk[i])):
            #        print(walk)
            #        print(rel_weights_by_walk[i])
            #return walks, rel_weights_by_walk
        print(len(walks))
        print(len(walks[0]))
        return walks

    def fit(self, **skip_gram_params):
        """
        Creates the embeddings using gensim's Word2Vec.
        :param skip_gram_params: Parameteres for gensim.models.Word2Vec - do not supply 'size' it is taken from the Node2Vec 'dimensions' parameter
        :type skip_gram_params: dict
        :return: A gensim word2vec model
        """

        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers

        if 'size' not in skip_gram_params:
            skip_gram_params['size'] = self.dimensions

        if self.relational_weighting:
            ave_weight = np.mean([self.graph.get_edge_data(*edge)[self.weight_key] for edge in self.graph.edges()])
            assert(abs(ave_weight) > self.TOLERANCE)
            skip_gram_params['ave_weight'] = ave_weight
            skip_gram_params['trans_factor'] = self.trans_factor

        #if self.relational_weighting:
        #    skip_gram_params['rel_weights_by_walk'] = self.rel_weights_by_walk

        return gensim.models.Word2Vec(self.walks, **skip_gram_params)

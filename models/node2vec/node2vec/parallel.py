import random
import numpy as np
from tqdm import tqdm

def parallel_generate_walks(d_graph, global_walk_length, num_walks, cpu_num, sampling_strategy=None,
                            num_walks_key=None, walk_length_key=None, neighbors_key=None, probabilities_key=None,
                            first_travel_key=None, temporal=False, time_key=None, relational_weighting=False, weight_key=None,
                            quiet=False):
    """
    Generates the random walks which will be used as the skip-gram input.
    :return: List of walks. Each walk is a list of nodes.
    """

    walks = list()

    if not quiet:
        pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

    for n_walk in range(num_walks):

        # Update progress bar
        if not quiet:
            pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        random.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:
            if relational_weighting:
                walk_weights = []
            # Skip nodes with specific num_walks
            if source in sampling_strategy and \
                    num_walks_key in sampling_strategy[source] and \
                    sampling_strategy[source][num_walks_key] <= n_walk:
                continue

            # Start walk
            walk = [source]
            if temporal:
                start_time_options = d_graph[source].get(time_key, None)
                if start_time_options:
                    time = random.choice(start_time_options)

            # Calculate walk length
            if source in sampling_strategy:
                walk_length = sampling_strategy[source].get(walk_length_key, global_walk_length)
            else:
                walk_length = global_walk_length



            # Perform walk
            while len(walk) < walk_length:

                walk_options = d_graph[walk[-1]].get(neighbors_key, None)

                # Skip dead end nodes
                if not walk_options:
                    break

                if len(walk) == 1:  # For the first step
                    probabilities = d_graph[walk[-1]][first_travel_key]
                else:
                    probabilities = d_graph[walk[-1]][probabilities_key][walk[-2]]

                if temporal: 
                    times = d_graph[walk[-1]][time_key]
                    probabilities = np.array([prob*(times[neighbor_idx] >= time) for neighbor_idx, prob in enumerate(probabilities)])
                    sum_probs = np.sum(probabilities)
                    if sum_probs == 0: break
                    probabilities = probabilities / np.sum(probabilities)

                walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]

                if relational_weighting:
                    walk_weights.append(d_graph[walk[-1]][weight_key][walk_to])

                walk.append(walk_to)
                if temporal:
                    time = times[walk_options.index(walk_to)]

            walk = list(map(str, walk))  # Convert all to strings

            if relational_weighting:
                #rel_weights = dict()
                #for i,u in enumerate(walk):
                #    rel_weight = 1
                #    step = 0
                #    for v in walk[i+1:]:
                #        rel_weight *= walk_weights[i+step] * trans_factor
                #        step += 1
                #        rel_weights[(u,v)] = rel_weight
                #        rel_weights[(v,u)] = rel_weight
                walks.append((walk, walk_weights))
            else:
                walks.append(walk)

    if not quiet:
        pbar.close()

    return walks

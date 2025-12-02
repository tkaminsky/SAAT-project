import numpy as np
from math import comb  # faster than scipy.special.comb in tight loops

import numpy as np
from math import comb

from voter_env import VoterEnv
from helpers import create_prob_distribution

def score_one_step(graph, placement, params, neighbors=None, majority_prob=None):
    """
    Your existing analytic surrogate: expected fraction correct after
    one majority update.
    (Renamed here for clarity; you can keep it as `score` if you prefer.)
    """
    pH, pL = params
    n = graph.shape[0]

    if neighbors is None:
        neighbors = [np.flatnonzero(graph[i]) for i in range(n)]

    if majority_prob is None:
        majority_prob = make_majority_prob_cache(pH, pL)

    is_high = np.zeros(n, dtype=bool)
    is_high[placement] = True

    total = 0.0
    for node in range(n):
        neigh = neighbors[node]
        n_neighbors = len(neigh)
        if n_neighbors == 0:
            continue

        nH = is_high[neigh].sum()
        nL = n_neighbors - nH
        total += majority_prob(nH, nL)

    return total / n


def score_mc_majority(
    graph,
    placement,
    params,
    T=50,
    n_iters=50,
):
    """
    Monte Carlo score:
    Run T full majority-dynamics trajectories and return the fraction of trials
    where the final majority of votes is correct (winner == 0).
    """
    pH, pL = params
    n_voters = graph.shape[0]
    placement = np.array(placement, dtype=int)

    # Probabilities are deterministic given placement + (pH, pL),
    # so compute once outside the loop:
    prob_per_voter = create_prob_distribution(
        n_voters,
        placement,
        pH,
        pL,
        n_alternatives=2,
    )

    # Build a single env and reuse it
    env = VoterEnv(
        graph=graph,
        num_voters=n_voters,
        num_alternatives=2,
        probs_per_voter=prob_per_voter,
        headless=True,
        record_data=False,   # be explicit; avoids any accidental disk I/O
    )

    successes = 0

    for t in range(T):
        # Resample initial preferences
        env.reset(probs_per_voter=prob_per_voter)
        env.run(iters=n_iters)
        w = env.winner()

        if w == 0:
            successes += 1

    return successes / T




def make_neighbors(graph):
    """Convert adjacency matrix to a list of neighbor index arrays."""
    n = graph.shape[0]
    return [np.flatnonzero(graph[i]) for i in range(n)]

def make_majority_prob_cache(pH, pL):
    """
    Returns a function majority_prob(nH, nL) with an internal cache
    for probabilities that majority of neighbors are correct
    given nH high-repute, nL low-repute neighbors.
    """
    cache = {}

    def majority_prob(nH, nL):
        key = (nH, nL)
        if key in cache:
            return cache[key]

        total_neighbors = nH + nL
        if total_neighbors == 0:
            # no neighbors: depending on your model, maybe 0.5 or something else.
            # sticking with 0.0 to match your current strict majority logic.
            cache[key] = 0.0
            return 0.0

        threshold = total_neighbors // 2 + 1  # strictly > half

        prob = 0.0
        # sum over kH, kL such that kH + kL >= threshold
        for kH in range(nH + 1):
            prob_H = comb(nH, kH) * (pH ** kH) * ((1 - pH) ** (nH - kH))
            min_kL = max(0, threshold - kH)
            if min_kL > nL:
                continue
            for kL in range(min_kL, nL + 1):
                prob_L = comb(nL, kL) * (pL ** kL) * ((1 - pL) ** (nL - kL))
                prob += prob_H * prob_L

        cache[key] = prob
        return prob

    return majority_prob

def score_analytic(graph, placement, params, neighbors=None, majority_prob=None):
    """
    Expected fraction correct after one majority update, as in your original score,
    but with caching and precomputed neighbor lists.
    """
    pH, pL = params
    n = graph.shape[0]

    # Precompute neighbors if not passed in
    if neighbors is None:
        neighbors = make_neighbors(graph)

    # Majority probability cache (per (nH,nL))
    if majority_prob is None:
        majority_prob = make_majority_prob_cache(pH, pL)

    # Boolean mask for O(1) membership checks
    is_high = np.zeros(n, dtype=bool)
    is_high[placement] = True

    total = 0.0
    for node in range(n):
        neigh = neighbors[node]
        n_neighbors = len(neigh)
        if n_neighbors == 0:
            continue  # or handle specially

        nH = is_high[neigh].sum()
        nL = n_neighbors - nH
        total += majority_prob(nH, nL)

    return total / n

def get_simulated_annealing_placement(
    graph,
    n_high_repute,
    params,
    score_type='analytic',
    initial_temp=1000.0,
    cooling_rate=0.95,
    max_iter=1000,
):
    n = graph.shape[0]
    pH, pL = params

    # if score_type == 'mc':
    #     max_iter = 100
    #     initial_temp = 100.0
    #     cooling_rate = 0.99

    # Precompute neighbor lists and majority cache once
    neighbors = make_neighbors(graph)
    majority_prob = make_majority_prob_cache(pH, pL)

    # Initial placement: random set of nodes
    current_placement = np.random.choice(n, n_high_repute, replace=False)
    current_score = score_analytic(graph, current_placement, params,
                          neighbors=neighbors,
                          majority_prob=majority_prob) if score_type == 'analytic' else score_mc_majority(
                              graph,
                              current_placement,
                              params,
                              T=20,
                              n_iters=5,
                          )

    # We'll sample from high and low nodes using a boolean mask
    is_high = np.zeros(n, dtype=bool)
    is_high[current_placement] = True

    temp = initial_temp

    for iteration in range(max_iter):
        # Choose one high node to drop, one low node to add
        high_nodes = np.flatnonzero(is_high)
        low_nodes = np.flatnonzero(~is_high)

        node_to_remove = np.random.choice(high_nodes)
        node_to_add = np.random.choice(low_nodes)

        # Build new placement
        new_placement = current_placement.copy()
        # Replace node_to_remove with node_to_add
        idx = np.where(new_placement == node_to_remove)[0][0]
        new_placement[idx] = node_to_add

        # Evaluate new placement
        new_score = score_analytic(graph, new_placement, params,
                          neighbors=neighbors,
                          majority_prob=majority_prob) if score_type == 'analytic' else score_mc_majority(
                              graph,
                              new_placement,
                              params,
                              T=20,
                              n_iters=5,
                          )

        # Simulated annealing acceptance
        if new_score > current_score:
            accept = True
        else:
            delta = new_score - current_score
            accept_prob = np.exp(delta / temp) if temp > 0 else 0.0
            accept = (np.random.rand() < accept_prob)

        if accept:
            current_placement = new_placement
            current_score = new_score
            # update mask
            is_high[node_to_remove] = False
            is_high[node_to_add] = True

        # Cool down
        temp *= cooling_rate

    return current_placement

import numpy as np
from graphs import make_Rbell, make_ER, make_ring, make_generic_bell, make_small_world
from voter_env import VoterEnv
import tqdm

# randomizing the placement
# graph = make_Rbell(R=2, n=10, k=0)
graph = make_generic_bell([25,25],[0])
 # Remove self-loops
n_trials=5 # how many times to run experiment
margins=np.linspace(0.0, 0.5, 50) # space to search over
n_voters = 50
n_alternatives = 2

correct_prop = np.zeros(len(margins))

for idx, margin in tqdm.tqdm(enumerate(margins), desc="Processing margins"):
    for _ in range(n_trials):

        graph = make_small_world(50, 4, 0.1) # randomizing the graph?
        # graph = make_ER(50, 0.1)
        # graph = make_ring(50)
        graph -= np.eye(graph.shape[0]) # subtract identity

        # Make first 21 voters likely to prefer alternative 0, and last 21 likely to prefer alternative 1
        # matrix where i,j is prob voter i chooses alternative j
        prob_per_voter = np.zeros((n_voters, n_alternatives))
        # half_voters = np.random.choice(n_voters, n_voters // 2, replace=False)
        # half_voters = list(range(n_voters // 2))
        # for i in range(n_voters):
        #     if i in half_voters:
        #         prob_per_voter[i, 0] = 0.5 + margin
        #         prob_per_voter[i, 1] = 0.5 - margin
        #     else:
        #         prob_per_voter[i, 0] = 0.5 - margin
        #         prob_per_voter[i, 1] = 0.5 + margin

        # half_voters = np.random.choice(n_voters, n_voters // 2, replace=False)
        # half_voters = np.random.choice(n_voters, (3* n_voters) // 4, replace=False)
        half_voters = np.arange((2* n_voters) // 4)
        for i in range(n_voters):
            if i in half_voters:
                prob_per_voter[i, 0] = 0.4
                prob_per_voter[i, 1] = 0.6
            else:
                prob_per_voter[i, 0] = .5 + margin
                prob_per_voter[i, 1] = .5 - margin


        env = VoterEnv(
            graph=graph,
            num_voters=n_voters,
            num_alternatives=n_alternatives,
            probs_per_voter=prob_per_voter,
            headless=False
        )
        env.run(iters=25) #fixed number of iterations - expectation is convergence after 3

        if env.winner() == 0:
            correct_prop[idx] += 1
        elif env.winner() == -1:
            correct_prop[idx] += 1 / n_alternatives
correct_prop /= n_trials


import matplotlib.pyplot as plt
plt.plot(margins + .5, correct_prop, marker='o')
plt.xlabel("Confidence Level of Correct Half (e.g. .7 means a voter chooses their preferred alternative 70% of the time)")
plt.ylabel("Proportion Correctly Choosing Alternative 0")
plt.title("1/2 Bern(.4), 1/2 Bern([Confidence]). Barbell Graph, Right Quarter Confident.")
plt.grid()
plt.savefig("test.png")
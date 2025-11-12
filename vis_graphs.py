import numpy as np
from graphs import make_Rbell, make_ER, make_ring, make_generic_bell, make_small_world
from voter_env import VoterEnv
import tqdm

graph = make_small_world(50, 4, 0.1)
graph -= np.eye(graph.shape[0])  # Remove self-loops

n_trials=1000
margins=np.linspace(0.0, 0.5, 50)
n_voters = 50
n_alternatives = 2


correct_prop = np.zeros(len(margins))

for margin in tqdm.tqdm(margins, desc="Processing margins"):
    for _ in range(n_trials):

        # seed = np.random.randint(0, 10000)
        margin = 0.25

        # Make first 21 voters likely to prefer alternative 0, and last 21 likely to prefer alternative 1
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

        half_voters = np.random.choice(n_voters, (3* n_voters) // 4, replace=False)
        # half_voters = np.arange((3* n_voters) // 4)
        # half_voters = np.arange((3* n_voters) // 4)
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
            headless=True
        )
        env.run(iters=25)

        if env.winner() == 0:
            correct_prop[np.where(margins == margin)[0][0]] += 1
    exit()
correct_prop /= n_trials

import matplotlib.pyplot as plt
plt.plot(margins + .5, correct_prop, marker='o')
plt.xlabel("Confidence Level (e.g. .7 means a voter chooses their preferred alternative 70% of the time)")
plt.ylabel("Proportion Correctly Choosing Alternative 0")
plt.title("3/4 Wrong on Barbell Graph, Varying Confidence Levels")
plt.grid()
plt.show()

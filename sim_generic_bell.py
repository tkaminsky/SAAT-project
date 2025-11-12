import numpy as np
from graphs import make_generic_bell
from voter_env import VoterEnv
import tqdm

graph = make_generic_bell([25, 25, 25, 25], [1, 1, 1])
graph -= np.eye(graph.shape[0])  # Remove self-loops

n_trials=100
margins=np.linspace(0.0, 0.5, 50)
n_voters = 100 # 25 + 25 + 25 + 25
n_alternatives = 2

correct_prop = np.zeros(len(margins))

for idx, margin in tqdm.tqdm(enumerate(margins), desc="Processing margins"):
    for _ in range(n_trials):

        # Initialize probability distribution for each voter
        prob_per_voter = np.zeros((n_voters, n_alternatives))

        # Randomly select 50 voters to be "high-repute" (prefer alternative 0 with higher confidence)
        high_repute_voters = np.random.choice(n_voters, 50, replace=False)

        # Assign probabilities based on voter type
        for i in range(n_voters):
            if i in high_repute_voters:
                # High-repute voters: prefer alternative 0 with probability 0.5 + margin
                prob_per_voter[i, 0] = 0.5 + margin
                prob_per_voter[i, 1] = 0.5 - margin
            else:
                # Low-repute voters: prefer alternative 1 with probability 0.5 + margin
                prob_per_voter[i, 0] = 0.5 - margin
                prob_per_voter[i, 1] = 0.5 + margin

        env = VoterEnv(
            graph=graph,
            num_voters=n_voters,
            num_alternatives=n_alternatives,
            probs_per_voter=prob_per_voter,
            headless=False
        )
        env.run(iters=25)

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
plt.show()
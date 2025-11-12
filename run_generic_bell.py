import numpy as np
from graphs import make_generic_bell
from voter_env import VoterEnv
from helpers import randomly_distribute_voters
import tqdm
import matplotlib.pyplot as plt

graph = make_generic_bell([25,25],[0])
 # Remove self-loops
n_trials=5 # how many times to run experiment
margins=np.linspace(0.0, 0.5, 50) # space to search over
n_voters = 50
n_alternatives = 2

correct_prop = np.zeros(len(margins))

for idx, margin in tqdm.tqdm(enumerate(margins), desc="Processing margins"):
    for _ in range(n_trials):
      
        prob_per_voter = randomly_distribute_voters(n_voters, n_alternatives, margin)
  
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

plt.plot(margins + .5, correct_prop, marker='o')
plt.xlabel("Confidence Level of Correct Half (e.g. .7 means a voter chooses their preferred alternative 70% of the time)")
plt.ylabel("Proportion Correctly Choosing Alternative 0")
plt.title("1/2 Bern(.4), 1/2 Bern([Confidence]). Barbell Graph, Right Quarter Confident.")
plt.grid()
plt.show()
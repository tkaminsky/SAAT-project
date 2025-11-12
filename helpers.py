import numpy as np

def randomly_distribute_voters(n_voters, n_alternatives, margin):
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
  
  return prob_per_voter

def cluser_voters():
  pass

def centralize_voters():
  pass
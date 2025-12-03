
"""

import random
import collections
import csv

#joint distribution
p = {
    (0,0): 0.40,
    (0,1): 0.20,
    (1,0): 0.15,
    (1,1): 0.25
}

#marginals
Pr_Y0=p[(0,0)]+ p[(1,0)]
Pr_Y1=p[(0,1)]+ p[(1,1)]
Pr_X0=p[(0,0)]+ p[(0,1)]
Pr_X1=p[(1,0)]+ p[(1,1)]

#sampling functions for conditionals
def sample_X_given_Y(y):
  if y == 0:
    prob = p[(1,0)]/Pr_Y0
  else:
    prob = p[(1,1)]/Pr_Y1
  U= random.random()
  return 1 if U < prob else 0
def sample_Y_given_X(x):
  if x == 0:
    prob = p[(0,1)]/Pr_X0
  else:
    prob = p[(1,1)]/Pr_X1
  U = random.random()
  return 1 if U < prob else 0


#Gibbs sampling -> print at checkpoints
checkpoints = [10, 100, 1000, 10000, 100000]
max_iter = max(checkpoints)

X, Y = 0, 0 #initial state
samples = []
#print table
def print_empirical(iteration, samples):
  counts = collections.Counter(samples)
  total= len(samples)
  print(f"\n=== After {iteration} iterations ===")
  print("State  |Empirical |  True")
  for state in [(0,0), (0,1), (1,0), (1,1)]:
    emp= counts[state]/total
    print(f"{state} | {emp:.3f}    | {p[state]:.3f}")

#convert to csv
with open("gibbs_output.csv", "w", newline="") as f:
  writer = csv.writer(f)
  writer.writerow(["Iterations", "State", "Emperical", "True"])

  for t in range(1, max_iter+1):
    X= sample_X_given_Y(Y)
    Y= sample_Y_given_X(X)
    samples.append((X,Y))

    if t in checkpoints:
      print_empirical(t, samples)
      counts= collections.Counter(samples)
      total= len(samples)
      for state in [(0,0), (0,1), (1,0), (1,1)]:
        empirical= counts[state]/total
        true_val = p[state]
        writer.writerow([t, state, empirical, true_val])

print("CSV file 'gibbs_output.csv' has been created")

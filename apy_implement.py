import sys
import heapq
import random
import math

def kk(A):
    A = [-a for a in A]
    heapq.heapify(A)
    while len(A) > 1:
        largest = -heapq.heappop(A)   # pop the largest (negate back)
        second_largest = -heapq.heappop(A)
        diff = largest - second_largest
        heapq.heappush(A, -diff)      # push the negated diff back
    return -heapq.heappop(A) if A else 0

def rand_S(A):
    return [random.choice([-1, 1]) for _ in range(len(A))]

def rand_neighbor_S(S):
    S_prime = S[:]
    i, j = random.sample(range(len(S)), 2)
    S_prime[i] = -S_prime[i]

    if random.choice([True, False]):
        S_prime[j] = -S_prime[j]

    return S_prime

def repeated_rand(A, max_iter=25000):
    S = rand_S(A)
    for iter in range(max_iter):
        S_prime = rand_S(A)
        if residue(S_prime, A) < residue(S, A):
            S = S_prime
    return residue(S, A)

def hill_climb(A, max_iter=25000):
    S = rand_S(A)
    for iter in range(max_iter):
        S_prime = rand_neighbor_S(S)
        if residue(S_prime, A) < residue(S, A):
            S = S_prime
    return residue(S, A)

def simulated_ann(A, max_iter=25000):
    S = rand_S(A)
    S_double_prime = S
    for iter in range(max_iter):
        T = (10**10) * (0.8**math.floor(iter / 300))
        S_prime = rand_neighbor_S(S)
        if residue(S_prime, A) < residue(S, A):
            S = S_prime
        else:
            prob = random.random()
            threshold = math.exp(-(residue(S_prime, A) - residue(S, A)) / T)
            if prob < threshold:
                S = S_prime
        if residue(S, A) < residue(S_double_prime, A):
            S_double_prime = S
    return residue(S_double_prime, A)

def rand_P(A):
    n = len(A)
    return [random.randint(1, n) for _ in range(n)]

def prepartition(A, P):
    n = len(A)
    A_prime = [0] * n
    for j in range(n):
        A_prime[P[j] - 1] += A[j]
    return A_prime

def randint_exclude(lower, upper, exclude):
    r = random.randint(lower, upper)
    while r in exclude:
        r = random.randint(lower, upper)
    return r

def rand_neighbor_P(P):
    P_prime = P
    i = random.randint(0, len(P) - 1)
    P_prime[i] = randint_exclude(1, len(P), [P[i]])
    return P_prime

def prepart_repeated_rand(A, max_iter=25000):
    P = rand_P(A)
    A1 = prepartition(A, P)
    for iter in range(max_iter):
        P_prime = rand_P(P)
        A2 = prepartition(A, P_prime)
        if kk(A2) < kk(A1):
            A1 = A2
            P = P_prime
    return kk(A1) # P

def prepart_hill_climb(A, max_iter=25000):
    P = rand_P(A)
    A1 = prepartition(A, P)
    for iter in range(max_iter):
        P_prime = rand_neighbor_P(P)
        A2 = prepartition(A, P_prime)
        if kk(A2) < kk(A1):
            A1 = A2
            P = P_prime
    return kk(A1)  # P

def prepart_simulated_ann(A, max_iter=25000):
    P = rand_P(A)
    A1 = prepartition(A, P)
    A3, P_double_prime = A1, P
    for iter in range(max_iter):
        T = (10**10) * (0.8**math.floor(iter / 300))
        P_prime = rand_neighbor_P(P)
        A2 = prepartition(A, P_prime)

        if kk(A2) < kk(A1):
            A1 = A2
            P = P_prime
        else:
            prob = random.random()
            threshold = math.exp(-(kk(A2) - kk(A1)) / T)
            if prob < threshold:
                A1 = A2
                P = P_prime
        if kk(A2) < kk(A3):
            A3 = A1
            P_double_prime = P
    return kk(A3) # P_double_prime
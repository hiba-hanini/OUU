def compute_optimal_policy(N):
    V = [[0.0, 0.0] for _ in range(N + 2)]
    policy = [None] * (N + 1)

    for n in range(N, 0, -1):

        # State 0
        V[n][0] = (
            1 / n * V[n + 1][1]
            + (n - 1) / n * V[n + 1][0]
        )

        # State 1
        hire = n / N

        continue_ = (
            1 / (n + 1) * V[n + 1][1]
            + n / (n + 1) * V[n + 1][0]
        )

        if hire >= continue_:
            V[n][1] = hire
            policy[n] = "H"
        else:
            V[n][1] = continue_
            policy[n] = "C"

    return V, policy



N = 10000000

V, policy = compute_optimal_policy(N)

print("V1* =", V[1][1])

for n in range(1, N + 1):
    if policy[n] == "H":
        print("k* =", n)
        break
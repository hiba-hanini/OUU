

from skater import SkaterMDP


mdp = SkaterMDP(
    n=8,
    p=0.7,
    obstacle_prob=0.2
)

T = 50

values, policies = mdp.solve(T)

initial_state = (mdp.start, None)

print(
    "Probability of reaching the goal:",
    values[0][initial_state]
)

trajectory, actions, actual_directions = mdp.simulate_optimal(
    T,
    policies
)

print("Trajectory:")
print(trajectory)

print("\nOptimal actions:")
print(actions)

mdp.plot_trajectory(
    trajectory,
    actions,
    actual_directions
)
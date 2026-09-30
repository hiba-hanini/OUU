import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from enum import Enum
from collections import deque
import math

class Direction(Enum):
    N = (-1, 0)
    E = (0, 1)
    S = (1, 0)
    W = (0, -1)

class SkaterMDP:

    def __init__(self, n, p=0.8, obstacle_prob=0.2, seed=None):

        self.n = n
        self.p = p

        if seed is not None:
            random.seed(seed)

        # Start, end
        self.start = (0, 0)
        self.end = (n - 1, n - 1)

        # Terminal states
        self.DEAD = "DEAD"
        self.GOAL = "GOAL"

        self.GOAL_STATE = (self.GOAL, None) # no previous momentum for terminal states
        self.DEAD_STATE = (self.DEAD, None)

        # Random obstacles
        self.obstacles = self._generate_obstacles(obstacle_prob)


    def _generate_obstacles(self, obstacle_prob):

        obstacles = set()

        for i in range(self.n):
            for j in range(self.n):

                # Start and End cannot be obstacles
                if (i, j) == self.start or (i, j) == self.end:
                    continue

                if random.random() < obstacle_prob:
                    obstacles.add((i, j))

        return obstacles


    def is_valid(self, position):

        i, j = position

        return (
            0 <= i < self.n
            and 0 <= j < self.n
            and position not in self.obstacles
        )


    def move(self, position, direction):

        i, j = position
        di, dj = direction.value

        new_position = (i + di, j + dj)

        # obstacle/outside grid : DEAD
        if not self.is_valid(new_position):
            return self.DEAD

        # Reached the goal
        if new_position == self.end:
            return self.GOAL

        return new_position

    # Transition P(s'|s,a)
    def transition(self, state, action):
        # return a list of (probability, next_state) pairs
        position, previous_direction = state

        if position == self.GOAL:
            return [(1.0, self.GOAL_STATE)]

        if position == self.DEAD:
            return [(1.0, self.DEAD_STATE)]


        # Start: no momentum yet
        if previous_direction is None:

            next_position = self.move(position, action)

            if next_position == self.GOAL:
                next_state = self.GOAL_STATE

            elif next_position == self.DEAD:
                next_state = self.DEAD_STATE

            else:
                next_state = (next_position, action)

            return [(1.0, next_state)]

        # chosen action ; proba p

        next_position_action = self.move(
            position,
            action
        )

        if next_position_action == self.GOAL:
            state_action = self.GOAL_STATE

        elif next_position_action == self.DEAD:
            state_action = self.DEAD_STATE

        else:
            state_action = (
                next_position_action,
                action
            )

        # keep momentum ; proba 1-p

        next_position_momentum = self.move(
            position,
            previous_direction
        )

        if next_position_momentum == self.GOAL:
            state_momentum = self.GOAL_STATE

        elif next_position_momentum == self.DEAD:
            state_momentum = self.DEAD_STATE

        else:
            state_momentum = (
                next_position_momentum,
                previous_direction
            )

        # same direction as previous: deterministic transition

        if action == previous_direction:

            return [
                (1.0, state_action)
            ]

        # therefore, two possible next states: chosen action or keep momentum

        return [
            (self.p, state_action),
            (1 - self.p, state_momentum)
        ]


    def get_states(self):
        # set of states S
        states = set()

        # Normal states
        for i in range(self.n):
            for j in range(self.n):

                position = (i, j)

                if not self.is_valid(position):
                    continue

                for direction in Direction:
                    states.add((position, direction))

        # Initial state
        states.add((self.start, None))

        # Terminal states
        states.add(self.GOAL_STATE)
        states.add(self.DEAD_STATE)

        return states

    # Bellman

    def solve(self, T):

        states = self.get_states()

        # Terminal states
        # u_T(GOAL) = 1
        # u_T(other) = 0

        u_next = {}

        for state in states:

            position, _ = state

            if position == self.GOAL:
                u_next[state] = 1.0

            else:
                u_next[state] = 0.0

        # Store u_t for every t
        values = {
            T: u_next.copy()
        }

        # Optimal policy
        policies = {}

        # backward induction

        for t in range(T - 1, -1, -1):

            u_current = {}
            policy_current = {}

            for state in states:

                position, _ = state

                if position == self.GOAL:

                    u_current[state] = 1.0
                    policy_current[state] = None
                    continue

                if position == self.DEAD:

                    u_current[state] = 0.0
                    policy_current[state] = None
                    continue

                # maximization over actions

                best_value = float("-inf")
                best_action = None

                for action in Direction:

                    transitions = self.transition(
                        state,
                        action
                    )

                    expected_value = 0.0

                    for probability, next_state in transitions:

                        expected_value += (
                            probability
                            * u_next[next_state]
                        )

                    # Keep the best action
                    if expected_value > best_value:

                        best_value = expected_value
                        best_action = action

                u_current[state] = best_value
                policy_current[state] = best_action

            # Save results for this t
            values[t] = u_current
            policies[t] = policy_current

            # Go one step backward
            u_next = u_current

        return values, policies


    def simulate_optimal(self, T, policies):

        position = self.start
        previous_direction = None

        trajectory = [position]
        actions = []
        actual_directions = []

        for t in range(T):

            state = (
                position,
                previous_direction
            )

            # Optimal action
            action = policies[t][state]

            # No action if terminal
            if action is None:
                break

            actions.append(action)

            if previous_direction is None:
                # First step: no momentum yet
                actual_direction = action

            else:
                if random.random() < self.p:
                    # Follow requested action
                    actual_direction = action
                else:
                    # Keep momentum
                    actual_direction = previous_direction

            actual_directions.append(actual_direction)

        

            new_position = self.move(
                position,
                actual_direction
            )

            trajectory.append(new_position)

            if new_position == self.GOAL:
                break

            if new_position == self.DEAD:
                break

            # Update state
            position = new_position
            previous_direction = actual_direction

        return trajectory, actions, actual_directions

    # plotting
    def plot_trajectory(
        self,
        trajectory,
        actions=None,
        actual_directions=None
    ):
        _, ax = plt.subplots(figsize=(8, 8))

        # grid and limits
        ax.set_xlim(-0.5, self.n - 0.5)
        ax.set_ylim(-0.5, self.n - 0.5)
        ax.set_xticks(range(self.n))
        ax.set_yticks(range(self.n))
        ax.grid(True)

        # obstacles
        for i, j in self.obstacles:
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    color="black"
                )
            )

        # start
        si, sj = self.start
        ax.scatter(
            sj,
            si,
            s=200,
            marker="o",
            color="green",
            label="Start"
        )

        # goal
        ei, ej = self.end
        ax.scatter(
            ej,
            ei,
            s=200,
            marker="*",
            color="red",
            label="Goal"
        )

        # draw trajectory
        for t in range(len(trajectory) - 1):

            current = trajectory[t]
            next_position = trajectory[t + 1]

            i, j = current

            # action chosen by the policy
            if actions is not None and t < len(actions):

                action = actions[t]
                di_action, dj_action = action.value

                # Draw chosen action in BLUE, dashed
                ax.arrow(
                    j,
                    i,
                    dj_action,
                    di_action,
                    head_width=0.10,
                    length_includes_head=True,
                    color="blue",
                    linestyle="--",
                    alpha=0.6
                )

            # actual movement (may differ from chosen action)
            if actual_directions is not None and t < len(actual_directions):

                actual_direction = actual_directions[t]
                di_actual, dj_actual = actual_direction.value

                # Normal movement
                if isinstance(next_position, tuple):

                    ni, nj = next_position

                    ax.arrow(
                        j,
                        i,
                        nj - j,
                        ni - i,
                        head_width=0.14,
                        length_includes_head=True,
                        color="red",
                        linewidth=2
                    )

                # Reached the goal
                elif next_position == self.GOAL:

                    ax.arrow(
                        j,
                        i,
                        dj_actual,
                        di_actual,
                        head_width=0.14,
                        length_includes_head=True,
                        color="red",
                        linewidth=2
                    )

                # Left the grid / hit obstacle
                elif next_position == self.DEAD:

                    ax.arrow(
                        j,
                        i,
                        dj_actual,
                        di_actual,
                        head_width=0.14,
                        length_includes_head=True,
                        color="red",
                        linewidth=2
                    )

        # time step labels
        for t, position in enumerate(trajectory):

            if isinstance(position, tuple):

                i, j = position

                ax.text(
                    j,
                    i,
                    str(t),
                    ha="center",
                    va="center"
                )

        legend_elements = [
            Line2D(
                [0],
                [0],
                color="blue",
                linestyle="--",
                linewidth=2,
                label="Chosen action"
            ),
            Line2D(
                [0],
                [0],
                color="red",
                linewidth=2,
                label="Real movement"
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="green",
                linestyle="None",
                markersize=10,
                label="Start"
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="red",
                linestyle="None",
                markersize=14,
                label="Goal"
            )
        ]

        ax.legend(handles=legend_elements)


        ax.invert_yaxis()
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title("Skater MDP Trajectory")

        plt.show()

    def shortest_path_length(self):
        """
        BFS
        """
        queue = deque([(self.start, 0)])
        visited = {self.start}

        while queue:
            position, distance = queue.popleft()

            if position == self.end:
                return distance

            for direction in Direction:
                next_position = self.move(position, direction)

                # Only consider normal grid positions
                if isinstance(next_position, tuple):
                    if next_position not in visited:
                        visited.add(next_position)
                        queue.append((next_position, distance + 1))

                elif next_position == self.GOAL:
                    return distance + 1

        # No path exists
        return None
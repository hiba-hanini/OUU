# Intern project

This folder contains a small internal optimization script related to the OUU course.

## File

- `compute_optimal_value.py`: computes an optimal policy/value threshold for the intern problem (https://team.inria.fr/polaris/files/2023/09/exo-intern.pdf) using Bellman equations and prints the first threshold where the optimal action switches to `H` (hire).

## Purpose

The script studies a simple dynamic decision model where the agent compares:
- hiring / stopping now, versus
- continuing to the next state.

It computes the value function and prints the first threshold where the optimal action switches to `H` (hire).

## Run

```bash
cd OUU/intern
python3 compute_optimal_value.py
```

This is a lightweight numerical experiment used to explore optimal decision rules.

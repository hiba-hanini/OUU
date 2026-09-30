# OUU - Skater MDP Exercise

This project is an exercise in Optimization under Uncertainty (OUU) based on a stochastic skater on a grid.

## Goal

The model studies a grid world where:
- the skater moves with momentum,
- actions may fail with probability $1-p$,
- obstacles can block the path,
- the objective is to maximize the probability of reaching the goal before falling into a dead state.

The implementation uses dynamic programming (Bellman equations) to compute the value function and an optimal policy.

## Project structure

- `skater.py`: MDP definitions, transitions, Bellman solver, simulation, and plotting utilities.
- `main.py`: small demonstration that computes the value function and simulates one optimal trajectory.
- `fix_T.py`: experiment that estimates the horizon $T^*$ needed to reach a target level of the plateau value.
- `requirements.txt`: Python dependencies.

## Setup

```bash
cd OUU/skater_ex
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the examples

```bash
python3 main.py
python3 fix_T.py
```


This is a compact exercise in stochastic optimal control and dynamic programming for a finite-horizon MDP.

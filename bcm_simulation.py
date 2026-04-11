"""
Bounded Confidence Model with Algorithmic Bias
================================================
Implements the Sirbu et al. (2019) BCM on Watts-Strogatz, Barabasi-Albert,
and Erdos-Renyi networks, following the research proposal of Isaac Boentoro.

Usage:
    python bcm_simulation.py              # quick validation run
    python bcm_simulation.py --full       # full parameter sweep (36,000 sims)
    python bcm_simulation.py --sweep      # medium sweep for testing
"""

import argparse
import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    N: int = 1000                    # number of agents
    mu: float = 0.5                  # convergence parameter
    max_steps: int = 500_000         # max timesteps before giving up
    convergence_tol: float = 1e-4    # opinion change threshold for early stop
    convergence_window: int = 1000   # check every N steps


@dataclass
class SweepParams:
    """All parameter combinations to sweep over."""
    gammas: list = field(default_factory=lambda: [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5])
    epsilons: list = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    ws_rewirings: list = field(default_factory=lambda: [0, 0.01, 0.05, 0.1, 0.5, 1.0])
    runs_per_condition: int = 50
    init_distributions: list = field(default_factory=lambda: ["uniform", "bimodal"])


# ---------------------------------------------------------------------------
# Network generation
# ---------------------------------------------------------------------------

def make_network(topology: str, N: int, rng: np.random.Generator, **kwargs) -> nx.Graph:
    """
    Generate a network of given topology.

    topology: "ws" | "ba" | "er"
    kwargs for ws: p (rewiring probability), k (initial degree, default 6)
    """
    if topology == "ws":
        k = kwargs.get("k", 6)
        p = kwargs["p"]
        G = nx.watts_strogatz_graph(N, k, p, seed=int(rng.integers(1 << 31)))
    elif topology == "ba":
        m = kwargs.get("m", 3)
        G = nx.barabasi_albert_graph(N, m, seed=int(rng.integers(1 << 31)))
    elif topology == "er":
        # target mean degree ~6, so p = 6/(N-1)
        p_er = kwargs.get("p_er", 6 / (N - 1))
        G = nx.erdos_renyi_graph(N, p_er, seed=int(rng.integers(1 << 31)))
    else:
        raise ValueError(f"Unknown topology: {topology!r}")

    # Ensure connectivity (take largest component if needed)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        G = nx.convert_node_labels_to_integers(G)

    return G


# ---------------------------------------------------------------------------
# Initial opinion distributions
# ---------------------------------------------------------------------------

def init_opinions(N: int, distribution: str, rng: np.random.Generator) -> np.ndarray:
    if distribution == "uniform":
        return rng.uniform(0, 1, N)
    elif distribution == "bimodal":
        # Two peaks at 0.2 and 0.8, std=0.05 each
        half = N // 2
        left  = rng.normal(0.2, 0.05, half)
        right = rng.normal(0.8, 0.05, N - half)
        opinions = np.concatenate([left, right])
        opinions = np.clip(opinions, 0, 1)
        rng.shuffle(opinions)
        return opinions
    else:
        raise ValueError(f"Unknown distribution: {distribution!r}")


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def run_simulation(
    G: nx.Graph,
    opinions: np.ndarray,
    gamma: float,
    epsilon: float,
    mu: float = 0.5,
    max_steps: int = 500_000,
    convergence_tol: float = 1e-4,
    convergence_window: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Run one BCM simulation.

    Agent i selects neighbour j with probability proportional to exp(-gamma * |xi - xj|).
    If |xi - xj| < epsilon, opinions update via the Deffuant rule.

    Returns a dict of outcome metrics.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = G.number_of_nodes()
    x = opinions.copy()

    # Pre-compute adjacency lists for speed
    adj = [list(G.neighbors(v)) for v in range(N)]

    prev_mean_change = np.inf
    converged_at = max_steps

    for step in range(max_steps):
        # Activate all agents in random order each timestep
        order = rng.permutation(N)
        total_change = 0.0

        for i in order:
            nbrs = adj[i]
            if not nbrs:
                continue

            # Compute selection probabilities
            diffs = np.abs(x[i] - x[nbrs])
            weights = np.exp(-gamma * diffs)
            weights /= weights.sum()

            # Select neighbour
            j = rng.choice(nbrs, p=weights)

            # Update if within confidence bound
            if abs(x[i] - x[j]) < epsilon:
                delta = mu * (x[j] - x[i])
                x[i] += delta
                total_change += abs(delta)

        # Check convergence every window steps
        if step % convergence_window == 0 and step > 0:
            mean_change = total_change / N
            if mean_change < convergence_tol:
                converged_at = step
                break
            prev_mean_change = mean_change

    # --- Measure outcomes ---
    clusters, cluster_sizes = _count_clusters(x, epsilon)
    opinion_dist = np.std(x)

    return {
        "mean_opinion":      float(np.mean(x)),
        "std_opinion":       float(opinion_dist),
        "n_clusters":        clusters,
        "mean_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0,
        "max_cluster_frac":  float(max(cluster_sizes) / N) if cluster_sizes else 0,
        "convergence_step":  converged_at,
        "polarised":         clusters > 1,
        "final_opinions":    x,   # kept for phase diagram use
    }


def _count_clusters(opinions: np.ndarray, epsilon: float) -> tuple[int, list[int]]:
    """
    Count opinion clusters: groups of agents within epsilon of each other.
    Uses a simple sort-and-scan approach.
    """
    sorted_ops = np.sort(opinions)
    clusters = []
    current_size = 1
    current_start = sorted_ops[0]

    for op in sorted_ops[1:]:
        if op - current_start < epsilon:
            current_size += 1
        else:
            clusters.append(current_size)
            current_size = 1
            current_start = op

    clusters.append(current_size)
    return len(clusters), clusters


# ---------------------------------------------------------------------------
# Validation run (γ=0 must reproduce ε*≈0.3)
# ---------------------------------------------------------------------------

def run_validation(cfg: SimConfig, n_runs: int = 20, seed: int = 42) -> pd.DataFrame:
    """
    Validate at γ=0 on ER and BA networks.
    Should show consensus for ε≥0.3 and fragmentation for ε<0.3.
    """
    print("Running validation (γ=0, ER and BA networks)...")
    rng = np.random.default_rng(seed)
    records = []

    for topology in ["er", "ba"]:
        for epsilon in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
            for run in range(n_runs):
                G_kwargs = {"p_er": 6 / (cfg.N - 1)} if topology == "er" else {"m": 3}
                G = make_network(topology, cfg.N, rng, **G_kwargs)
                x0 = init_opinions(cfg.N, "uniform", rng)

                result = run_simulation(
                    G, x0,
                    gamma=0.0,
                    epsilon=epsilon,
                    mu=cfg.mu,
                    max_steps=cfg.max_steps,
                    convergence_tol=cfg.convergence_tol,
                    convergence_window=cfg.convergence_window,
                    rng=rng,
                )
                records.append({
                    "topology": topology,
                    "epsilon":  epsilon,
                    "run":      run,
                    "n_clusters": result["n_clusters"],
                    "polarised":  result["polarised"],
                    "std_opinion": result["std_opinion"],
                })

    df = pd.DataFrame(records)
    summary = df.groupby(["topology", "epsilon"]).agg(
        mean_clusters=("n_clusters", "mean"),
        polarised_frac=("polarised", "mean"),
    ).reset_index()

    print("\nValidation results (should show transition near ε=0.3):")
    print(summary.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Full parameter sweep
# ---------------------------------------------------------------------------

def run_sweep(
    cfg: SimConfig,
    params: SweepParams,
    output_path: Path,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the full parameter sweep and save results to CSV."""
    rng = np.random.default_rng(seed)
    records = []

    # Build all conditions
    ws_conditions = [("ws", {"p": p}) for p in params.ws_rewirings]
    ba_conditions = [("ba", {"m": 3})]
    er_conditions = [("er", {"p_er": 6 / (cfg.N - 1)})]
    all_conditions = ws_conditions + ba_conditions + er_conditions

    total = (
        len(all_conditions)
        * len(params.gammas)
        * len(params.epsilons)
        * len(params.init_distributions)
        * params.runs_per_condition
    )
    print(f"Total simulations: {total:,}")

    done = 0
    t0 = time.time()

    for (topology, net_kwargs), gamma, epsilon, dist in itertools.product(
        all_conditions, params.gammas, params.epsilons, params.init_distributions
    ):
        for run in range(params.runs_per_condition):
            G = make_network(topology, cfg.N, rng, **net_kwargs)
            x0 = init_opinions(cfg.N, dist, rng)

            result = run_simulation(
                G, x0,
                gamma=gamma,
                epsilon=epsilon,
                mu=cfg.mu,
                max_steps=cfg.max_steps,
                convergence_tol=cfg.convergence_tol,
                convergence_window=cfg.convergence_window,
                rng=rng,
            )

            record = {
                "topology":   topology,
                "p_ws":       net_kwargs.get("p", None),
                "gamma":      gamma,
                "epsilon":    epsilon,
                "init_dist":  dist,
                "run":        run,
                "n_clusters": result["n_clusters"],
                "polarised":  result["polarised"],
                "std_opinion":      result["std_opinion"],
                "mean_cluster_size": result["mean_cluster_size"],
                "max_cluster_frac": result["max_cluster_frac"],
                "convergence_step": result["convergence_step"],
            }
            records.append(record)
            done += 1

            if verbose and done % 500 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                remaining = (total - done) / rate
                print(f"  {done:>6}/{total} | {rate:.0f} sims/s | ~{remaining/60:.1f} min remaining")

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BCM algorithmic bias simulation")
    parser.add_argument("--full",   action="store_true", help="Run full 36k sweep")
    parser.add_argument("--sweep",  action="store_true", help="Run reduced test sweep")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--output", type=str, default="results.csv")
    args = parser.parse_args()

    cfg = SimConfig()

    if args.full:
        params = SweepParams()
        df = run_sweep(cfg, params, Path(args.output), seed=args.seed)
        print("Full sweep complete.")

    elif args.sweep:
        # Reduced sweep for testing: fewer runs, fewer gamma values
        params = SweepParams(
            gammas=[0, 1, 2, 3, 5],
            epsilons=[0.1, 0.2, 0.3, 0.4, 0.5],
            ws_rewirings=[0, 0.05, 0.5, 1.0],
            runs_per_condition=5,
            init_distributions=["uniform"],
        )
        df = run_sweep(cfg, params, Path(args.output), seed=args.seed)
        print("Test sweep complete.")

    else:
        # Default: validation only
        run_validation(cfg, n_runs=20, seed=args.seed)
        print("\nValidation complete. Run with --sweep or --full for the full study.")


if __name__ == "__main__":
    main()
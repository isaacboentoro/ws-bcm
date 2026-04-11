"""
BCM Parallel Sweep
==================
Drop-in replacement for the sweep in bcm_simulation.py, using all available
CPU cores via multiprocessing.Pool.

Usage:
    python bcm_parallel.py                    # full sweep, auto-detect cores
    python bcm_parallel.py --workers 8        # fix core count
    python bcm_parallel.py --sweep            # reduced test sweep
    python bcm_parallel.py --output out.csv   # custom output path

Each worker gets an independent RNG seeded from the job index, so results
are fully reproducible regardless of how many cores you use.
"""

import argparse
import itertools
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from bcm_simulation import (
    SimConfig,
    SweepParams,
    make_network,
    init_opinions,
    run_simulation,
)


# ---------------------------------------------------------------------------
# Job definition — one unit of work per core
# ---------------------------------------------------------------------------

@dataclass
class Job:
    job_id:     int
    topology:   str
    net_kwargs: dict
    gamma:      float
    epsilon:    float
    dist:       str
    run:        int
    cfg:        SimConfig


def _run_job(job: Job) -> dict:
    """
    Execute a single simulation. Called in a worker process.
    Seeded from job_id so results are reproducible across runs.
    """
    rng = np.random.default_rng(job.job_id)
    G   = make_network(job.topology, job.cfg.N, rng, **job.net_kwargs)
    x0  = init_opinions(job.cfg.N, job.dist, rng)

    result = run_simulation(
        G, x0,
        gamma=job.gamma,
        epsilon=job.epsilon,
        mu=job.cfg.mu,
        max_steps=job.cfg.max_steps,
        convergence_tol=job.cfg.convergence_tol,
        convergence_window=job.cfg.convergence_window,
        rng=rng,
    )

    return {
        "job_id":            job.job_id,
        "topology":          job.topology,
        "p_ws":              job.net_kwargs.get("p", None),
        "gamma":             job.gamma,
        "epsilon":           job.epsilon,
        "init_dist":         job.dist,
        "run":               job.run,
        "n_clusters":        result["n_clusters"],
        "polarised":         result["polarised"],
        "std_opinion":       result["std_opinion"],
        "mean_cluster_size": result["mean_cluster_size"],
        "max_cluster_frac":  result["max_cluster_frac"],
        "convergence_step":  result["convergence_step"],
    }


# ---------------------------------------------------------------------------
# Job list builder
# ---------------------------------------------------------------------------

def build_jobs(cfg: SimConfig, params: SweepParams) -> list[Job]:
    ws_conditions = [("ws", {"p": p}) for p in params.ws_rewirings]
    ba_conditions = [("ba", {"m": 3})]
    er_conditions = [("er", {"p_er": 6 / (cfg.N - 1)})]
    all_conditions = ws_conditions + ba_conditions + er_conditions

    jobs = []
    job_id = 0
    for (topology, net_kwargs), gamma, epsilon, dist in itertools.product(
        all_conditions, params.gammas, params.epsilons, params.init_distributions
    ):
        for run in range(params.runs_per_condition):
            jobs.append(Job(
                job_id=job_id,
                topology=topology,
                net_kwargs=net_kwargs,
                gamma=gamma,
                epsilon=epsilon,
                dist=dist,
                run=run,
                cfg=cfg,
            ))
            job_id += 1

    return jobs


# ---------------------------------------------------------------------------
# Parallel sweep with live progress
# ---------------------------------------------------------------------------

def run_parallel_sweep(
    cfg: SimConfig,
    params: SweepParams,
    output_path: Path,
    n_workers: int | None = None,
    checkpoint_every: int = 2000,
) -> pd.DataFrame:
    """
    Run the full sweep across all cores.

    Checkpoints results to CSV every `checkpoint_every` completed jobs so
    you don't lose progress if the run is interrupted.
    """
    if n_workers is None:
        n_workers = os.cpu_count()

    jobs = build_jobs(cfg, params)
    total = len(jobs)

    print(f"Jobs: {total:,}  |  Workers: {n_workers}  |  Output: {output_path}\n")

    records = []
    checkpoint_path = output_path.with_suffix(".checkpoint.csv")

    with mp.Pool(processes=n_workers) as pool:
        with tqdm(total=total, unit="sim", dynamic_ncols=True) as pbar:
            for record in pool.imap_unordered(_run_job, jobs, chunksize=1):
                records.append(record)
                pbar.update(1)

                if len(records) % checkpoint_every == 0:
                    pd.DataFrame(records).to_csv(checkpoint_path, index=False)

    df = pd.DataFrame(records).sort_values("job_id").reset_index(drop=True)
    df.to_csv(output_path, index=False)

    # Clean up checkpoint file if sweep completed
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    elapsed = time.time() - t0
    print(f"\nDone. {total:,} simulations in {elapsed/60:.1f} min → {output_path}")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parallel BCM sweep")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: all cores)")
    parser.add_argument("--sweep",  action="store_true",
                        help="Run a reduced test sweep instead of the full 36k")
    parser.add_argument("--output", type=str, default="results.csv")
    args = parser.parse_args()

    cfg = SimConfig()

    if args.sweep:
        params = SweepParams(
            gammas=[0, 1, 2, 3, 5],
            epsilons=[0.1, 0.2, 0.3, 0.4, 0.5],
            ws_rewirings=[0, 0.05, 0.5, 1.0],
            runs_per_condition=5,
            init_distributions=["uniform"],
        )
    else:
        params = SweepParams()  # full 36,000-sim sweep

    run_parallel_sweep(
        cfg, params,
        output_path=Path(args.output),
        n_workers=args.workers,
    )


if __name__ == "__main__":
    # Required on macOS/Windows to avoid fork-bomb on spawn-based multiprocessing
    mp.set_start_method("spawn", force=True)
    main()
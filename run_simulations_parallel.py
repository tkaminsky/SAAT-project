"""
Systematic Experiments: Parallelized Version
Optimized for high-core-count CPUs (Threadripper/Epyc/Xeon).

This script performs the same logic as run_simulations.py but distributes
the simulation workload across all available CPU cores.

Architecture:
    1. Outer Loop: Iterate through Graph Configurations (Tribell k=..., BA m=...)
    2. Batching: For each graph, generate ALL required simulation tasks (pL * fracH * pH * methods).
    3. Parallel Execution: Run all tasks in parallel using ProcessPoolExecutor.
    4. Aggregation & Plotting: Collect results and generate plots sequentially.

Usage:
    python run_simulations_parallel.py
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
import random  # <-- NEW
from datetime import datetime
import argparse
import concurrent.futures
import itertools
from collections import defaultdict

from graphs import make_tribell, make_ER, make_BA
from voter_env import VoterEnv
from helpers import (
    get_centrality_placement, 
    get_peripheral_placement, 
    get_random_placement,
    get_maximin_placement,
    create_prob_distribution
)
from simulated_annealing import get_simulated_annealing_placement

# =============================================================================
# CONFIGURATION LOAD
# =============================================================================

try:
    from experiment_config import *
    print("Loaded configuration from experiment_config.py")
except ImportError:
    print("ERROR: experiment_config.py not found!")
    sys.exit(1)

# -----------------------------------------------------------------------------
# GLOBAL BASE SEED FOR REPRODUCIBILITY
# -----------------------------------------------------------------------------
# If experiment_config.py defines something like GLOBAL_SEED or RANDOM_SEED,
# we can use it. Otherwise fall back to a default constant.
#
# You can also just set BASE_SEED explicitly here if you prefer.
# -----------------------------------------------------------------------------
if "GLOBAL_SEED" in globals():
    BASE_SEED = GLOBAL_SEED
else:
    BASE_SEED = 2380  # <-- change this if you want

print(f"Using BASE_SEED = {BASE_SEED}")

# -----------------------------------------------------------------------------
# GLOBALS FOR WORKERS
# -----------------------------------------------------------------------------
GLOBAL_GRAPH = None
GLOBAL_N_VOTERS = None

def init_worker(graph, n_voters):
    """
    Initializer for each worker process.
    Sets the global graph and n_voters; RNG is handled per-task.
    """
    global GLOBAL_GRAPH, GLOBAL_N_VOTERS
    GLOBAL_GRAPH = graph
    GLOBAL_N_VOTERS = n_voters

# =============================================================================
# WORKER FUNCTIONS (Must be picklable/top-level)
# =============================================================================

def run_single_trial_logic(graph, actual_n_voters, high_repute_voters, pH, pL):
    """
    Core simulation logic for a single trial.
    Assumes RNG has already been seeded in the caller for reproducibility.
    """
    prob_per_voter = create_prob_distribution(actual_n_voters, high_repute_voters, pH, pL)
    
    # headless=True is CRITICAL for parallel processing to avoid GUI/Render locks
    env = VoterEnv(
        graph=graph,
        num_voters=actual_n_voters,
        num_alternatives=2,
        probs_per_voter=prob_per_voter,
        headless=True 
    )
    env.run(iters=N_ITERATIONS)
    
    winner = env.winner()
    if winner == 0:
        return 1.0
    elif winner == -1:
        return 0.5
    else:
        return 0.0

def execute_simulation_task(task_data):
    """
    Worker entry point.
    Unpacks arguments and runs the experiment for a specific configuration.
    
    Args:
        task_data: tuple containing (task_id, fracH, pH, pL, method, n_trials)
        
    Returns:
        tuple: (pL, fracH, pH, method, average_accuracy)
    """
    task_id, fracH, pH, pL, method, n_trials = task_data

    # -------------------------------------------------------------------------
    # REPRODUCIBLE RNG PER TASK
    # -------------------------------------------------------------------------
    # Each task gets its own deterministic seed derived from BASE_SEED
    # and task_id. This makes results independent of scheduling / CPU count.
    # -------------------------------------------------------------------------
    seed = BASE_SEED + task_id
    np.random.seed(seed)
    random.seed(seed)

    graph = GLOBAL_GRAPH
    n_voters = GLOBAL_N_VOTERS
    
    n_high_repute = int(n_voters * fracH)
    total_score = 0.0
    
    # 1. Determine Placement (Deterministic methods done once per task, Random per trial)
    if method == 'central':
        placements = [get_centrality_placement(graph, n_high_repute, 'degree')] * n_trials
    elif method == 'peripheral':
        placements = [get_peripheral_placement(graph, n_high_repute)] * n_trials
    elif method == 'maximin':
        placements = [get_maximin_placement(graph, n_high_repute)] * n_trials
    elif method == 'random':
        placements = [get_random_placement(n_voters, n_high_repute) for _ in range(n_trials)]
    elif method == 'simulated_annealing':
        placements = [get_simulated_annealing_placement(graph, n_high_repute, (pH, pL), 'analytic') for _ in range(n_trials)]
        # placements = [placement] * n_trials
    elif method == 'mc_annealing':
        placement = get_simulated_annealing_placement(graph, n_high_repute, (pH, pL), 'mc')
        placements = [placement] * n_trials
    else:
        raise ValueError(f"Unknown method: {method}")
        
    # 2. Run Trials
    for i in range(n_trials):
        total_score += run_single_trial_logic(graph, n_voters, placements[i], pH, pL)
        
    return (pL, fracH, pH, method, total_score / n_trials)

# =============================================================================
# EXPERIMENT SUITE
# =============================================================================

def run_parallel_experiment_suite(graph_name, graph_generator, graph_params_list):
    """
    Run experiments for a graph type using parallel execution.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(OUTPUT_DIR, f"{graph_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    total_configs = len(graph_params_list)
    
    print(f"\n{'='*80}")
    print(f"Starting PARALLEL {graph_name.upper()} experiments")
    print(f"Configurations: {total_configs}")
    print(f"Output: {exp_dir}")
    print(f"CPUs available: {os.cpu_count()}")
    print(f"{'='*80}\n")
    
    # 1. Iterate over Graph Configurations (Sequential)
    for config_idx, graph_params in enumerate(graph_params_list):
        
        # OPTIONAL: Make graph generation itself reproducible relative to BASE_SEED
        # If you want different graphs per config but deterministic across runs,
        # you can reseed here with an offset based on config_idx:
        # np.random.seed(BASE_SEED + 10_000 * config_idx)
        # random.seed(BASE_SEED + 10_000 * config_idx)

        # A. Generate Graph
        graph_gen_params = {k: v for k, v in graph_params.items() if k != 'actual_n_voters'}
        graph = graph_generator(**graph_gen_params)
        graph = graph - np.eye(graph.shape[0])  # Remove self-loops
        actual_n_voters = graph_params.get('actual_n_voters', N_VOTERS)
        
        param_str = '_'.join([f"{k}{v}" for k, v in graph_gen_params.items()])
        print(f"Processing Config {config_idx+1}/{total_configs}: {param_str} (n={actual_n_voters})")

        # B. Prepare Task Batch
        tasks = []
        
        # Cartesian product of all variables
        # We attach a deterministic task_id to each configuration
        task_id = 0
        for pL, fracH, pH, method in itertools.product(PL_VALUES, FRACH_VALUES, PH_VALUES, PLACEMENT_METHODS):
            
            # Skip edge cases logic if configured
            if SKIP_EDGE_CASES and (fracH == 0.0 or fracH == 1.0):
                continue

            tasks.append((task_id, fracH, pH, pL, method, N_TRIALS))
            task_id += 1
            
        print(f"  > Generated {len(tasks)} simulation tasks. executing in parallel...")
        
        # C. Execute in Parallel
        # Results storage: results[pL][fracH][method][pH] = accuracy
        results_store = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=os.cpu_count(),
            initializer=init_worker,
            initargs=(graph, actual_n_voters),
        ) as executor:
            futures = []
            try:
                futures = [executor.submit(execute_simulation_task, task) for task in tasks]
                
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="  Simulating",
                    leave=False
                ):
                    res_pL, res_fracH, res_pH, res_method, accuracy = future.result()
                    results_store[res_pL][res_fracH][res_method][res_pH] = accuracy
                    
            except KeyboardInterrupt:
                print("\n\n" + "!"*80)
                print("CAUGHT KEYBOARD INTERRUPT - STOPPING")
                print("!"*80)
                print("Cancelling pending tasks...")
                
                for f in futures:
                    f.cancel()
                    
                executor.shutdown(wait=False)
                print("Pool shutdown. Exiting.")
                sys.exit(1)
        
        # D. Generate Plots (Sequential)
        print("  > Generating plots...")
        plot_count = 0
        
        for pL in PL_VALUES:
            for fracH in FRACH_VALUES:
                if SKIP_EDGE_CASES and (fracH == 0.0 or fracH == 1.0):
                    continue
                
                fig, ax = plt.subplots(figsize=FIGURE_SIZE)
                
                colors = {
                    'random': 'blue', 
                    'central': 'red', 
                    'peripheral': 'green',
                    'maximin': 'purple',
                    'simulated_annealing': 'orange',
                    'mc_annealing': 'cyan'
                }
                markers = {
                    'random': 'o', 
                    'central': 's', 
                    'peripheral': '^',
                    'maximin': '*',
                    'simulated_annealing': 'D',
                    'mc_annealing': 'v'
                }
                
                for method in PLACEMENT_METHODS:
                    data_points = results_store[pL][fracH][method]
                    if not data_points:
                        continue
                    
                    sorted_ph = sorted(data_points.keys())
                    sorted_acc = [data_points[ph] for ph in sorted_ph]
                    
                    ax.plot(
                        sorted_ph, sorted_acc,
                        color=colors.get(method, 'black'),
                        marker=markers.get(method, '.'),
                        label=method.capitalize(),
                        linewidth=2, markersize=6, alpha=0.7
                    )
                
                ax.set_xlabel('pH (High-Repute Voter Accuracy)', fontsize=12)
                ax.set_ylabel('Proportion Correct', fontsize=12)
                title = f"{graph_name.upper()}: {param_str}\n"
                title += f"pL={pL:.2f}, fracH={fracH:.2f} ({fracH*100:.0f}% High-Repute), n={actual_n_voters}"
                ax.set_title(title, fontsize=10)
                ax.legend(fontsize=10, loc='best')
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1.05])
                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
                
                plt.tight_layout()
                
                filename = f"{graph_name}_{param_str}_pL{pL:.2f}_fracH{fracH:.2f}.png"
                filepath = os.path.join(exp_dir, filename)
                plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
                plt.close(fig)
                plot_count += 1
                
        print(f"  > Saved {plot_count} plots to {exp_dir}")

    return exp_dir

# =============================================================================
# WRAPPERS
# =============================================================================

def run_tribell_parallel():
    graph_params = []
    for k in TRIBELL_K_VALUES:
        target_n = (N_VOTERS - 2*k) / 3
        n_per_region = int(round(target_n))
        if n_per_region < 1:
            continue
        actual_n = 3 * n_per_region + 2 * k
        graph_params.append({'n': n_per_region, 'k': k, 'actual_n_voters': actual_n})
    return run_parallel_experiment_suite('tribell', make_tribell, graph_params)

def run_BA_parallel():
    graph_params = []
    for m in BA_M_VALUES:
        if m >= N_VOTERS or m < 1:
            continue
        graph_params.append({'n': N_VOTERS, 'm': m})
    return run_parallel_experiment_suite('BA', make_BA, graph_params)

def run_ER_parallel():
    graph_params = []
    for p in ER_P_VALUES:
        if p <= 0:
            continue
        graph_params.append({'n': N_VOTERS, 'p': float(p)})
    return run_parallel_experiment_suite('ER', make_ER, graph_params)

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*80)
    print("PARALLEL EXPERIMENT RUNNER")
    print("="*80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Seed the main process RNGs once for reproducible graph generation, etc.
    np.random.seed(BASE_SEED)
    random.seed(BASE_SEED)
    
    parser = argparse.ArgumentParser(description='Run parallel systematic experiments')
    parser.add_argument('--test', action='store_true', help='Test mode')
    parser.add_argument('--tribell', action='store_true')
    parser.add_argument('--BA', action='store_true')
    parser.add_argument('--ER', action='store_true')
    args = parser.parse_args()
    
    if args.test:
        print("Test mode not fully implemented in parallel runner, running full tribell...")
        run_tribell_parallel()
    elif args.tribell:
        run_tribell_parallel()
    elif args.BA:
        run_BA_parallel()
    elif args.ER:
        run_ER_parallel()
    else:
        # run_tribell_parallel()
        # run_BA_parallel()
        run_ER_parallel()
        
    print("\nAll parallel experiments completed.")

if __name__ == "__main__":
    main()

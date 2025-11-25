"""
Systematic Experiments: Comprehensive Parameter Sweep

This script runs systematic experiments across different graph topologies,
testing the effect of placement strategies on collective accuracy.

Parameters:
    - n_voters: Fixed at 1000
    - pH: Varies from 0.5 to 1.0 (x-axis of plots)
    - pL: {0.0, 0.25, 0.5} (separate plot sets)
    - fracH: {0.25, 0.5, 0.75} (separate plots, skipping 0.0 and 1.0)
    - Placement: {random, central, peripheral} (lines on same plot)

Graph types tested:
    - Tribell: k in {1, 50, 250}
    - Barabási-Albert: m in {1, 11, 22, ..., 99}
    - Erdős-Rényi: p in {0.1, 0.2, ..., 0.9}

Usage:
    python systematic_experiments.py              # Run all
    python systematic_experiments.py --test       # Quick test
    python systematic_experiments.py --tribell    # Only tribell
    python systematic_experiments.py --BA         # Only BA
    python systematic_experiments.py --ER         # Only ER
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
from datetime import datetime
import argparse

# sys.path.append('/mnt/project')

from graphs import make_generic_bell, make_ER, make_BA
from voter_env import VoterEnv
from helpers import (
    get_centrality_placement, 
    get_peripheral_placement, 
    get_random_placement,
    create_prob_distribution
)

# =============================================================================
# CONFIGURATION (Can be overridden by experiment_config.py)
# =============================================================================

try:
    from experiment_config import *
    print("Loaded configuration from experiment_config.py")
except ImportError:
    print("Using default configuration (experiment_config.py not found)")
    N_VOTERS = 1000
    N_TRIALS = 1
    N_ITERATIONS = 25
    PH_VALUES = np.arange(0.5, 1.05, 0.05)
    PL_VALUES = [0.0, 0.25, 0.5]
    FRACH_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
    PLACEMENT_METHODS = ['random', 'central', 'peripheral']
    TRIBELL_K_VALUES = [1, 50, 250]
    BA_M_VALUES = [1, 11, 22, 33, 44, 55, 66, 77, 88, 99]
    ER_P_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    OUTPUT_DIR = 'systematic_results_test'
    DPI = 200
    FIGURE_SIZE = (10, 6)
    SKIP_EDGE_CASES = True

# =============================================================================
# CORE EXPERIMENT FUNCTIONS
# =============================================================================

def run_single_experiment(graph, n_voters, fracH, pH, pL, placement_methods):
    """
    Run experiment for single configuration across all placement methods.
    
    Returns dict: {placement_method: accuracy}
    """
    n_high_repute = int(n_voters * fracH)
    if n_high_repute == 0 or n_high_repute == n_voters:
        # Edge case: all same type
        return {method: 0.5 for method in placement_methods}
    
    results = {method: 0.0 for method in placement_methods}
    
    for _ in range(N_TRIALS):
        for method in placement_methods:
            # Select high-repute voters
            if method == 'random':
                high_repute_voters = get_random_placement(n_voters, n_high_repute)
            elif method == 'central':
                high_repute_voters = get_centrality_placement(graph, n_high_repute, 'degree')
            elif method == 'peripheral':
                high_repute_voters = get_peripheral_placement(graph, n_high_repute)
            else:
                raise ValueError(f"Unknown placement method: {method}")
            
            # Create probability distribution
            prob_per_voter = create_prob_distribution(n_voters, high_repute_voters, pH, pL)
            
            # Run simulation
            env = VoterEnv(
                graph=graph,
                num_voters=n_voters,
                num_alternatives=2,
                probs_per_voter=prob_per_voter,
                headless=True
            )
            env.run(iters=N_ITERATIONS)
            
            # Record result
            if env.winner() == 0:
                results[method] += 1
            elif env.winner() == -1:
                results[method] += 0.5
    
    # Normalize
    for method in placement_methods:
        results[method] /= N_TRIALS
    
    return results


def run_experiment_suite(graph_name, graph_generator, graph_params_list):
    """
    Run complete suite of experiments for a graph type.
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(OUTPUT_DIR, f"{graph_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Calculate expected plots
    plot_count = 0
    n_fracH = len(FRACH_VALUES)
    if SKIP_EDGE_CASES:
        n_fracH -= 2  # Skip 0.0 and 1.0
    total_plots = len(graph_params_list) * len(PL_VALUES) * n_fracH
    
    print(f"\n{'='*80}")
    print(f"Starting {graph_name.upper()} experiments")
    print(f"Graph configurations: {len(graph_params_list)}")
    print(f"Expected plots: {total_plots}")
    print(f"Output directory: {exp_dir}")
    print(f"{'='*80}\n")
    
    # Loop over graph parameters
    for graph_params in graph_params_list:
        # Create parameter string for filename
        param_str = '_'.join([f"{k}{v}" for k, v in graph_params.items()])
        print(f"\n{'─'*60}")
        print(f"Graph config: {graph_params}")
        print(f"{'─'*60}")
        
        # Loop over pL values
        for pL in PL_VALUES:
            
            # Loop over fracH values
            for fracH in FRACH_VALUES:
                
                # Skip edge cases if configured
                if SKIP_EDGE_CASES and (fracH == 0.0 or fracH == 1.0):
                    continue
                
                plot_count += 1
                print(f"[{plot_count}/{total_plots}] pL={pL:.2f}, fracH={fracH:.2f}")
                
                # Store results for each placement method
                results = {method: np.zeros(len(PH_VALUES)) for method in PLACEMENT_METHODS}
                
                # Loop over pH values (x-axis)
                for pH_idx, pH in enumerate(tqdm(PH_VALUES, 
                                                 desc="  pH sweep", 
                                                 leave=False)):
                    
                    # Generate graph
                    graph = graph_generator(**graph_params)
                    graph = graph - np.eye(graph.shape[0])
                    
                    # Run experiment
                    trial_results = run_single_experiment(
                        graph, N_VOTERS, fracH, pH, pL, PLACEMENT_METHODS
                    )
                    
                    # Store results
                    for method in PLACEMENT_METHODS:
                        results[method][pH_idx] = trial_results[method]
                
                # Create plot
                fig, ax = plt.subplots(figsize=FIGURE_SIZE)
                
                colors = {'random': 'blue', 'central': 'red', 'peripheral': 'green'}
                markers = {'random': 'o', 'central': 's', 'peripheral': '^'}
                
                for method in PLACEMENT_METHODS:
                    ax.plot(PH_VALUES, results[method], 
                           color=colors[method], marker=markers[method],
                           label=method.capitalize(), linewidth=2, 
                           markersize=4, alpha=0.7)
                
                ax.set_xlabel('pH (High-Repute Voter Accuracy)', fontsize=12)
                ax.set_ylabel('Proportion Correct', fontsize=12)
                
                # Title with all parameters
                title = f"{graph_name.upper()}: {param_str}\n"
                title += f"pL={pL:.2f}, fracH={fracH:.2f} ({fracH*100:.0f}% High-Repute), n={N_VOTERS}"
                ax.set_title(title, fontsize=10)
                
                ax.legend(fontsize=10, loc='best')
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1.05])
                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
                
                plt.tight_layout()
                
                # Save with descriptive filename
                filename = f"{graph_name}_{param_str}_pL{pL:.2f}_fracH{fracH:.2f}.png"
                filepath = os.path.join(exp_dir, filename)
                plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
                plt.close(fig)
    
    print(f"\n{'='*80}")
    print(f"Completed {graph_name.upper()}")
    print(f"Generated: {plot_count} plots")
    print(f"Saved to: {exp_dir}")
    print(f"{'='*80}\n")
    
    return exp_dir


# =============================================================================
# GRAPH-SPECIFIC EXPERIMENT RUNNERS
# =============================================================================

def run_tribell_experiments():
    """Tribell experiments with varying connector lengths."""
    print("\n" + "#"*80)
    print("# TRIBELL EXPERIMENTS")
    print("#"*80)
    
    graph_params_list = []
    for k in TRIBELL_K_VALUES:
        # Split voters into 3 regions
        n_per_region = N_VOTERS // 3
        remainder = N_VOTERS % 3
        
        ns = [n_per_region, n_per_region, n_per_region]
        if remainder > 0:
            ns[0] += remainder
        
        ks = [k, k]
        
        graph_params_list.append({'ns': ns, 'ks': ks, 'k': k})
    
    return run_experiment_suite('tribell', make_generic_bell, graph_params_list)


def run_BA_experiments():
    """Barabási-Albert experiments with varying m."""
    print("\n" + "#"*80)
    print("# BARABÁSI-ALBERT EXPERIMENTS")
    print("#"*80)
    
    graph_params_list = []
    for m in BA_M_VALUES:
        if m >= N_VOTERS:
            print(f"Warning: m={m} >= n={N_VOTERS}, skipping")
            continue
        graph_params_list.append({'n': N_VOTERS, 'm': m})
    
    return run_experiment_suite('BA', make_BA, graph_params_list)


def run_ER_experiments():
    """Erdős-Rényi experiments with varying edge probability."""
    print("\n" + "#"*80)
    print("# ERDŐS-RÉNYI EXPERIMENTS")
    print("#"*80)
    
    graph_params_list = []
    for p in ER_P_VALUES:
        graph_params_list.append({'n': N_VOTERS, 'p': float(p)})
    
    return run_experiment_suite('ER', make_ER, graph_params_list)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def print_config():
    """Print current configuration."""
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"N_VOTERS: {N_VOTERS}")
    print(f"N_TRIALS: {N_TRIALS}")
    print(f"N_ITERATIONS: {N_ITERATIONS}")
    print(f"pH range: {min(PH_VALUES):.2f} to {max(PH_VALUES):.2f}")
    print(f"pL values: {PL_VALUES}")
    print(f"fracH values: {FRACH_VALUES}")
    print(f"Placement methods: {PLACEMENT_METHODS}")
    print(f"Skip edge cases: {SKIP_EDGE_CASES}")
    print(f"\nGraph parameters:")
    print(f"  Tribell k: {TRIBELL_K_VALUES}")
    print(f"  BA m: {BA_M_VALUES}")
    print(f"  ER p: {ER_P_VALUES}")
    print(f"\nOutput: {OUTPUT_DIR}")
    print("="*80 + "\n")


def main():
    """Run all experiments."""
    print_config()
    
    # Create base output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results = {}
    
    # Run all suites
    results['tribell'] = run_tribell_experiments()
    results['BA'] = run_BA_experiments()
    results['ER'] = run_ER_experiments()
    
    # Final summary
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)
    print("\nResults saved to:")
    for graph_type, result_dir in results.items():
        print(f"  {graph_type}: {result_dir}")
    print("\n" + "="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run systematic voter experiments')
    parser.add_argument('--test', action='store_true', help='Run quick test with reduced parameters')
    parser.add_argument('--tribell', action='store_true', help='Run only tribell experiments')
    parser.add_argument('--BA', action='store_true', help='Run only BA experiments')
    parser.add_argument('--ER', action='store_true', help='Run only ER experiments')
    
    args = parser.parse_args()
    
    if args.test:
        print("\n*** RUNNING IN TEST MODE ***")
        print("Using reduced parameters for quick testing\n")
        N_VOTERS = 100
        N_TRIALS = 10
        PH_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        TRIBELL_K_VALUES = [1, 50]
        BA_M_VALUES = [1, 10]
        ER_P_VALUES = [0.1, 0.5]
        OUTPUT_DIR = 'test_systematic_results'
        print_config()
        run_tribell_experiments()
    
    elif args.tribell:
        print_config()
        run_tribell_experiments()
    
    elif args.BA:
        print_config()
        run_BA_experiments()
    
    elif args.ER:
        print_config()
        run_ER_experiments()
    
    else:
        # Run all
        main()
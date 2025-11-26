"""
Systematic Experiments: Comprehensive Parameter Sweep

This script runs systematic experiments across different graph topologies,
testing the effect of placement strategies on collective accuracy.

Loop structure:
    For each graph type (Tribell, BA, ER):
        For each graph parameter (k, m, or p):
            For each pL value (0.0, 0.25, 0.5):
                For each fracH value (0.0, 0.25, 0.5, 0.75, 1.0):
                    For each pH value (0.5 to 1.0) - X-AXIS:
                        For each placement method (random, central, peripheral, optimized) - LINES:
                            Run N_TRIALS simulations
                    Create one plot with 4 lines (one per placement method)

Expected outputs:
    - Tribell: 3 k-values × 3 pL × 5 fracH = 45 plots
    - BA: 10 m-values × 3 pL × 5 fracH = 150 plots  
    - ER: 9 p-values × 3 pL × 5 fracH = 135 plots
    - TOTAL: 330 plots

Usage:
    python run_simulations.py              # Run all
    python run_simulations.py --test       # Quick test
    python run_simulations.py --tribell    # Only tribell
    python run_simulations.py --BA         # Only BA
    python run_simulations.py --ER         # Only ER
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
from datetime import datetime
import argparse

# sys.path.append('/mnt/project')

from graphs import make_tribell, make_ER, make_BA
from voter_env import VoterEnv

# Updated import to point to the file containing our new function
from voter_placement import (
    get_centrality_placement, 
    get_peripheral_placement, 
    get_random_placement,
    get_optimized_placement,
    create_prob_distribution
)

# =============================================================================
# CONFIGURATION
# =============================================================================

try:
    from experiment_config import *
    print("Loaded configuration from experiment_config.py")
except ImportError:
    print("ERROR: experiment_config.py not found!")
    print("Please ensure experiment_config.py is in the same directory.")
    sys.exit(1)

# Ensure 'optimized' is in the placement methods list
if 'optimized' not in PLACEMENT_METHODS:
    PLACEMENT_METHODS.append('optimized')
    print("Added 'optimized' to PLACEMENT_METHODS")

# =============================================================================
# CORE EXPERIMENT FUNCTIONS
# =============================================================================

def run_single_trial(graph, actual_n_voters, high_repute_voters, pH, pL):
    """
    Run a single trial with given configuration.
    
    Returns: 1 if correct, 0.5 if tie, 0 if incorrect
    """
    # Create probability distribution
    prob_per_voter = create_prob_distribution(actual_n_voters, high_repute_voters, pH, pL)
    
    # Run simulation (headless=True but it will still create experiment directory)
    # We suppress this by not using headless mode features
    env = VoterEnv(
        graph=graph,
        num_voters=actual_n_voters,
        num_alternatives=2,
        probs_per_voter=prob_per_voter,
        headless=False  # Changed to False to avoid creating videos
    )
    env.run(iters=N_ITERATIONS)
    
    # Record result
    winner = env.winner()
    if winner == 0:
        return 1.0
    elif winner == -1:
        return 0.5
    else:
        return 0.0


def run_experiment_for_configuration(graph, actual_n_voters, fracH, pH, pL, placement_methods):
    """
    Run experiment for single (pH, pL, fracH) configuration across all placement methods.
    
    Returns dict: {placement_method: accuracy}
    """
    n_high_repute = int(actual_n_voters * fracH)
    
    results = {method: 0.0 for method in placement_methods}
    
    # For each placement method
    for method in placement_methods:
        # Select high-repute voters based on method
        if method == 'random':
            # Run multiple trials with different random placements
            for _ in range(N_TRIALS):
                high_repute_voters = get_random_placement(actual_n_voters, n_high_repute)
                results[method] += run_single_trial(graph, actual_n_voters, high_repute_voters, pH, pL)
        
        elif method == 'central':
            # Centrality is deterministic, but we still run multiple trials for sampling variance
            high_repute_voters = get_centrality_placement(graph, n_high_repute, 'degree')
            for _ in range(N_TRIALS):
                results[method] += run_single_trial(graph, actual_n_voters, high_repute_voters, pH, pL)
        
        elif method == 'peripheral':
            # Peripheral is also deterministic
            high_repute_voters = get_peripheral_placement(graph, n_high_repute)
            for _ in range(N_TRIALS):
                results[method] += run_single_trial(graph, actual_n_voters, high_repute_voters, pH, pL)

        elif method == 'optimized':
            # Optimized is deterministic (greedy constructive)
            high_repute_voters = get_optimized_placement(graph, n_high_repute)
            for _ in range(N_TRIALS):
                results[method] += run_single_trial(graph, actual_n_voters, high_repute_voters, pH, pL)
        
        else:
            raise ValueError(f"Unknown placement method: {method}")
        
        # Normalize
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
    total_plots = len(graph_params_list) * len(PL_VALUES) * len(FRACH_VALUES)
    plot_count = 0
    
    print(f"\n{'='*80}")
    print(f"Starting {graph_name.upper()} experiments")
    print(f"Graph configurations: {len(graph_params_list)}")
    print(f"Expected plots: {total_plots}")
    print(f"Output directory: {exp_dir}")
    print(f"{'='*80}\n")
    
    # Loop over graph parameters
    for graph_params in graph_params_list:
        # Generate graph once for this configuration
        graph_gen_params = {k: v for k, v in graph_params.items() 
                           if k not in ['actual_n_voters']}
        graph = graph_generator(**graph_gen_params)
        graph = graph - np.eye(graph.shape[0])
        
        # Get actual number of voters
        actual_n_voters = graph_params.get('actual_n_voters', N_VOTERS)
        
        # Create parameter string for filename
        param_str = '_'.join([f"{k}{v}" for k, v in graph_gen_params.items()])
        
        print(f"\n{'─'*60}")
        print(f"Graph config: {graph_params}")
        print(f"Actual nodes: {actual_n_voters}")
        print(f"{'─'*60}")
        
        # Loop over pL values
        for pL in PL_VALUES:
            
            # Loop over fracH values
            for fracH in FRACH_VALUES:
                plot_count += 1
                print(f"[{plot_count}/{total_plots}] pL={pL:.2f}, fracH={fracH:.2f}")
                
                # Store results for each placement method
                results = {method: np.zeros(len(PH_VALUES)) for method in PLACEMENT_METHODS}
                
                # Loop over pH values (x-axis)
                for pH_idx, pH in enumerate(tqdm(PH_VALUES, 
                                                 desc="  pH sweep", 
                                                 leave=False)):
                    
                    # Run experiment for this configuration
                    trial_results = run_experiment_for_configuration(
                        graph, actual_n_voters, fracH, pH, pL, PLACEMENT_METHODS
                    )
                    
                    # Store results
                    for method in PLACEMENT_METHODS:
                        results[method][pH_idx] = trial_results[method]
                
                # Create plot
                fig, ax = plt.subplots(figsize=FIGURE_SIZE)
                
                colors = {
                    'random': 'blue', 
                    'central': 'red', 
                    'peripheral': 'green',
                    'optimized': 'purple'
                }
                markers = {
                    'random': 'o', 
                    'central': 's', 
                    'peripheral': '^',
                    'optimized': '*'
                }
                
                for method in PLACEMENT_METHODS:
                    ax.plot(PH_VALUES, results[method], 
                           color=colors.get(method, 'black'), 
                           marker=markers.get(method, '.'),
                           label=method.capitalize(), linewidth=2, 
                           markersize=6, alpha=0.7)
                
                ax.set_xlabel('pH (High-Repute Voter Accuracy)', fontsize=12)
                ax.set_ylabel('Proportion Correct', fontsize=12)
                
                # Title with all parameters
                title = f"{graph_name.upper()}: {param_str}\n"
                title += f"pL={pL:.2f}, fracH={fracH:.2f} ({fracH*100:.0f}% High-Repute), n={actual_n_voters}"
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
        # Calculate n such that 3n + 2k ≈ N_VOTERS
        # Total nodes in tribell = 3n + 2k
        target_n = (N_VOTERS - 2*k) / 3
        n_per_region = int(round(target_n))
        
        # Make sure we have a valid n
        if n_per_region < 1:
            print(f"Warning: k={k} too large for n_voters={N_VOTERS}, skipping")
            continue
        
        # Calculate actual number of voters this will create
        actual_n_voters = 3 * n_per_region + 2 * k
        
        graph_params_list.append({
            'n': n_per_region, 
            'k': k,
            'actual_n_voters': actual_n_voters
        })
        
        print(f"Tribell config: k={k}, n={n_per_region}, total_nodes={actual_n_voters}")
    
    return run_experiment_suite('tribell', make_tribell, graph_params_list)


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
        if m == 0:
            print(f"Warning: m=0 is invalid for BA graphs, skipping")
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
        if p == 0:
            print(f"Warning: p=0 will create disconnected graph, skipping")
            continue
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
    print(f"pH values: {len(PH_VALUES)} points from {min(PH_VALUES):.2f} to {max(PH_VALUES):.2f}")
    print(f"pL values: {PL_VALUES}")
    print(f"fracH values: {FRACH_VALUES}")
    print(f"Placement methods: {PLACEMENT_METHODS}")
    print(f"\nGraph parameters:")
    print(f"  Tribell k: {TRIBELL_K_VALUES}")
    print(f"  BA m: {BA_M_VALUES}")
    print(f"  ER p: {ER_P_VALUES}")
    
    expected = {
        'tribell': len(TRIBELL_K_VALUES) * len(PL_VALUES) * len(FRACH_VALUES),
        'BA': len(BA_M_VALUES) * len(PL_VALUES) * len(FRACH_VALUES),
        'ER': len(ER_P_VALUES) * len(PL_VALUES) * len(FRACH_VALUES)
    }
    expected['total'] = sum(expected.values())
    
    print(f"\nExpected plots:")
    print(f"  Tribell: {expected['tribell']}")
    print(f"  BA: {expected['BA']}")
    print(f"  ER: {expected['ER']}")
    print(f"  TOTAL: {expected['total']}")
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
        N_TRIALS = 5
        N_ITERATIONS = 10
        PH_VALUES = [0.5, 0.7, 0.9]
        PL_VALUES = [0.0, 0.5]
        FRACH_VALUES = [0.25, 0.75]
        TRIBELL_K_VALUES = [1, 10]
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
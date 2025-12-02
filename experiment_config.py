"""
Configuration and parameter settings for systematic experiments.
Modify this file to adjust experiment parameters without changing the main script.
"""

import numpy as np

# =============================================================================
# EXPERIMENTAL PARAMETERS
# =============================================================================

# Fixed parameters
N_VOTERS = 1000
N_TRIALS = 100  # Trials per configuration (increase for more accuracy, decreases speed)
N_ITERATIONS = 20  # Convergence iterations per trial

# Variable parameters (these define the x-axis and different plots)
PH_VALUES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]  # x-axis
# PH_VALUES = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]  # x-axis
PL_VALUES = [0.25, 0.50, 0.75]  # Each pL produces separate set of plots
FRACH_VALUES = [0.25, 0.50]  # Each fracH produces separate plot

# Placement methods to compare (appear as different lines on same plot)
# Added 'optimized' to the list
PLACEMENT_METHODS = ['random', 'central', 'peripheral', 'optimized']

# =============================================================================
# GRAPH-SPECIFIC PARAMETERS
# =============================================================================

# TRIBELL PARAMETERS
# k = connector length between regions
TRIBELL_K_VALUES = [0, 10]

# BARABÁSI-ALBERT PARAMETERS
# m = edges to attach from new node to existing nodes
# Removed 0 because m must be >= 1
BA_M_VALUES = [10, 20, 30]

# ERDŐS-RÉNYI PARAMETERS  
# p = edge probability
# Removed 0 to ensure graph connectivity and prevent centrality errors
ER_P_VALUES = [0.25, 0.50, 0.75]

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

OUTPUT_DIR = 'redone_results'
DPI = 200  # Resolution of saved plots
FIGURE_SIZE = (10, 6)  # Width, height in inches

# =============================================================================
# SKIP SETTINGS
# =============================================================================

# Skip edge cases where all voters are high-repute or all are low-repute
SKIP_EDGE_CASES = False  # Set to False to include fracH = 0.0 and fracH = 1.0

# =============================================================================
# EXPECTED PLOT COUNTS
# =============================================================================

def calculate_expected_plots():
    """Calculate expected number of plots for each graph type."""
    n_pL = len(PL_VALUES)
    n_fracH = len(FRACH_VALUES)
    if SKIP_EDGE_CASES:
        n_fracH -= 2  # Remove 0.0 and 1.0
    
    tribell_plots = len(TRIBELL_K_VALUES) * n_pL * n_fracH
    ba_plots = len(BA_M_VALUES) * n_pL * n_fracH
    er_plots = len(ER_P_VALUES) * n_pL * n_fracH
    total_plots = tribell_plots + ba_plots + er_plots
    
    return {
        'tribell': tribell_plots,
        'BA': ba_plots,
        'ER': er_plots,
        'total': total_plots
    }

# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================

def print_configuration():
    """Print current configuration."""
    print("\n" + "="*80)
    print("EXPERIMENT CONFIGURATION")
    print("="*80)
    print(f"\nFixed Parameters:")
    print(f"  N_VOTERS: {N_VOTERS}")
    print(f"  N_TRIALS: {N_TRIALS}")
    print(f"  N_ITERATIONS: {N_ITERATIONS}")
    
    print(f"\nVariable Parameters:")
    print(f"  pH values: {PH_VALUES}")
    print(f"  pL values: {PL_VALUES}")
    print(f"  fracH values: {FRACH_VALUES}")
    print(f"  Placement methods: {PLACEMENT_METHODS}")
    
    print(f"\nGraph Parameters:")
    print(f"  Tribell k values: {TRIBELL_K_VALUES}")
    print(f"  BA m values: {BA_M_VALUES}")
    print(f"  ER p values: {ER_P_VALUES}")
    
    print(f"\nOutput Settings:")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  DPI: {DPI}")
    print(f"  Figure size: {FIGURE_SIZE}")
    print(f"  Skip edge cases: {SKIP_EDGE_CASES}")
    
    expected = calculate_expected_plots()
    print(f"\nExpected Plots:")
    print(f"  Tribell: {expected['tribell']} (3 k-values × 3 pL × 5 fracH)")
    print(f"  BA: {expected['BA']} (10 m-values × 3 pL × 5 fracH)")
    print(f"  ER: {expected['ER']} (9 p-values × 3 pL × 5 fracH)")
    print(f"  TOTAL: {expected['total']}")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    print_configuration()
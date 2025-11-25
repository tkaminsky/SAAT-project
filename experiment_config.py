"""
Configuration and parameter settings for systematic experiments.
Modify this file to adjust experiment parameters without changing the main script.
"""

# =============================================================================
# EXPERIMENTAL PARAMETERS
# =============================================================================

# Fixed parameters
N_VOTERS = 1000
N_TRIALS = 50  # Trials per configuration (increase for more accuracy, decreases speed)
N_ITERATIONS = 25  # Convergence iterations per trial

# Variable parameters (these define the x-axis and different plots)
PH_VALUES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]  # x-axis
PL_VALUES = [0.0, 0.25, 0.5]  # Each pL produces separate set of plots
FRACH_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]  # Each fracH produces separate plot

# Placement methods to compare (appear as different lines on same plot)
PLACEMENT_METHODS = ['random', 'central', 'peripheral']

# =============================================================================
# GRAPH-SPECIFIC PARAMETERS
# =============================================================================

# TRIBELL PARAMETERS
# k = connector length between regions
TRIBELL_K_VALUES = [1, 50, 250]

# BARABÁSI-ALBERT PARAMETERS
# m = edges to attach from new node to existing nodes
# Note: m must be < n_voters
# Original request was range(1000, 1000000, 100000) but this doesn't make sense for BA
# Interpreting as: test different m values to vary graph density
BA_M_VALUES = [1, 11, 22, 33, 44, 55, 66, 77, 88, 99]

# ERDŐS-RÉNYI PARAMETERS  
# p = edge probability
ER_P_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

OUTPUT_DIR = 'systematic_results'
DPI = 200  # Resolution of saved plots
FIGURE_SIZE = (10, 6)  # Width, height in inches

# =============================================================================
# SKIP SETTINGS
# =============================================================================

# Skip edge cases where all voters are high-repute or all are low-repute
SKIP_EDGE_CASES = True  # If True, skip fracH = 0.0 and fracH = 1.0

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
    print(f"  Tribell: {expected['tribell']}")
    print(f"  BA: {expected['BA']}")
    print(f"  ER: {expected['ER']}")
    print(f"  TOTAL: {expected['total']}")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    print_configuration()
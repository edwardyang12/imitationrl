import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
import os
import glob
import argparse

# ---------------------------------------------------------
# Configuration & Styling
# ---------------------------------------------------------
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

MODEL_COLORS = {
    'MLP': '#d62728',         # Red
    'T': '#1f77b4',           # Blue (Transformer)
    'GAT': '#2ca02c',         # Green
    'PTN': "#a02c89"          # Purple (PointNet)
}

# High-Contrast Colormap for Success Rate (Higher is better: Red -> Yellow -> Green)
c_nodes = [0.0, 0.3, 0.5, 0.8, 1.0]
c_colors = ["#d62728", "#ff9896", "#ffc107", "#98df8a", "#2ca02c"]
SUCCESS_CMAP = LinearSegmentedColormap.from_list("success_phase", list(zip(c_nodes, c_colors)))

# ---------------------------------------------------------
# Data Ingestion
# ---------------------------------------------------------
def load_and_aggregate_data(data_dir):
    all_dataframes = []
    csv_files = glob.glob(os.path.join(data_dir, "*transport_metrics.csv"))
    if not csv_files:
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}'")
        
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        # Expected format: MODEL_NTRAIN_NMAX_NPACKAGES__transport_metrics.csv
        prefix_part = filename.split('__')[0]
        parts = prefix_part.split('_')
        
        if len(parts) >= 3:
            # Handle cases where model name might be missing (e.g., 30_8_1)
            if parts[0].isdigit():
                model_name = "Unknown"
                n_train = int(parts[0])
                n_max = int(parts[1])
                n_packages = int(parts[2])
            else:
                model_name = parts[0]
                n_train = int(parts[1])
                # Handle potential '_mod_' tag in filenames like T_30_mod_8_1
                if parts[2] == "mod":
                    n_max = int(parts[3])
                    n_packages = int(parts[4])
                    model_name += "_mod"
                else:
                    n_max = int(parts[2])
                    n_packages = int(parts[3])
                
            df = pd.read_csv(file_path)
            df['Model'] = model_name
            df['N_train'] = n_train
            df['n_max'] = n_max
            
            # Use package count from filename if missing from CSV
            if 'Packages' not in df.columns:
                df['Packages'] = n_packages
                
            all_dataframes.append(df)

    if not all_dataframes:
        raise ValueError("No valid data could be aggregated.")
        
    return pd.concat(all_dataframes, ignore_index=True)

# ---------------------------------------------------------
# Filter Helper
# ---------------------------------------------------------
def get_optimal_nmax_df(df, n_train, n_packages):
    """Selects the best n_max configuration per model based on max success rate."""
    subset = df[(df['N_train'] == n_train) & (df['Packages'] == n_packages)]
    if subset.empty: return subset
    
    # Find the n_max that produced the highest overall success rate across all N_test for each model
    best_nmax = subset.groupby(['Model', 'n_max'])['Overall_Success_Rate'].mean().reset_index()
    best_configs = best_nmax.loc[best_nmax.groupby('Model')['Overall_Success_Rate'].idxmax()]
    
    optimal_dfs = []
    for _, row in best_configs.iterrows():
        opt_df = subset[(subset['Model'] == row['Model']) & (subset['n_max'] == row['n_max'])]
        optimal_dfs.append(opt_df)
        
    return pd.concat(optimal_dfs, ignore_index=True)

# ---------------------------------------------------------
# Graph 1: Zero-Shot Scaling (Success & Time)
# ---------------------------------------------------------
def plot_scaling_curves(df, n_train, n_packages, output_dir="plots_transport"):
    df_base = get_optimal_nmax_df(df, n_train, n_packages)
    if df_base.empty: return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    current_palette = {k: v for k, v in MODEL_COLORS.items() if k in df_base['Model'].unique()}
    
    # Add _mod variants to palette dynamically if they exist
    for m in df_base['Model'].unique():
        if m not in current_palette:
            base_m = m.replace('_mod', '')
            current_palette[m] = MODEL_COLORS.get(base_m, '#000000')

    # Plot 1: Success Rate
    sns.lineplot(data=df_base, x='N_test', y='Overall_Success_Rate', 
                 hue='Model', palette=current_palette, marker='o', linewidth=2.5, ax=ax1)
    ax1.set_title(f'Zero-Shot Scaling: Success Rate (P={n_packages})')
    ax1.set_xlabel('Test Population Density (N_test)')
    ax1.set_ylabel('Overall Success Rate (Higher is Better)')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot 2: Time to Completion
    sns.lineplot(data=df_base, x='N_test', y='Mean_Time_To_Completion', 
                 hue='Model', palette=current_palette, marker='s', linewidth=2.5, ax=ax2)
    ax2.set_title(f'Efficiency: Mean Time to Completion')
    ax2.set_xlabel('Test Population Density (N_test)')
    ax2.set_ylabel('Cycles (Lower is Better)')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'1_Scaling_Curves_P{n_packages}.pdf'))
    plt.close()

# ---------------------------------------------------------
# Graph 2: Phase Diagram Heatmaps (Success Rate)
# ---------------------------------------------------------
def plot_phase_diagrams(df, n_packages, output_dir="plots_transport"):
    max_test = df['N_test'].max()
    df_max = df[(df['N_test'] == max_test) & (df['Packages'] == n_packages)].copy()
    
    if df_max.empty: return
        
    models = df_max['Model'].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    if len(models) == 1: axes = [axes]
    
    full_n_train = sorted(df_max['N_train'].unique())
    full_n_max = sorted(df_max['n_max'].unique())
    
    for i, model in enumerate(models):
        model_data = df_max[df_max['Model'] == model]
        pivot = model_data.pivot_table(index='N_train', columns='n_max', values='Overall_Success_Rate', aggfunc='mean')
        
        idx = pd.Index(full_n_train, name='N_train')
        cols = pd.Index(full_n_max, name='n_max')
        pivot = pivot.reindex(index=idx, columns=cols).fillna(0.0) # Fill missing with 0 success
        
        sns.heatmap(
            pivot, ax=axes[i], cmap=SUCCESS_CMAP, vmin=0.0, vmax=1.0, 
            annot=True, fmt=".2f", cbar=(i == len(models)-1), cbar_kws={'label': 'Success Rate'}
        )
        axes[i].set_title(f'{model} Robustness (N={max_test}, P={n_packages})')
        axes[i].invert_yaxis()
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'2_Phase_Diagrams_P{n_packages}.pdf'))
    plt.close()

# ---------------------------------------------------------
# Graph 3: Behavioral Breakdowns
# ---------------------------------------------------------
def plot_behavioral_breakdowns(df, n_train, n_packages, output_dir="plots_transport"):
    df_base = get_optimal_nmax_df(df, n_train, n_packages)
    if df_base.empty: return

    metrics = [
        ('Active_Transport_Error_Mean', 'Mean Distance to Goal'),
        ('Active_Transport_Velocity_Mean', 'Transport Velocity (Goodput)'),
        ('Active_Collision_Rate_Mean', 'Active Collision Rate'),
        ('Active_Engagement_Rate_Mean', 'Agent Engagement Rate')
    ]
    
    metrics = [(m, label) for m, label in metrics if m in df_base.columns]
    if not metrics: return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    
    # Setup Palette
    current_palette = {}
    for m in df_base['Model'].unique():
        base_m = m.replace('_mod', '')
        current_palette[m] = MODEL_COLORS.get(base_m, '#000000')
    
    for idx, (metric_col, y_label) in enumerate(metrics):
        ax = axes[idx]
        sns.lineplot(
            data=df_base, x='N_test', y=metric_col, hue='Model', 
            palette=current_palette, marker='s', ax=ax, legend=(idx==0)
        )
        ax.set_title(y_label)
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle='--', alpha=0.5)
        
    axes[2].set_xlabel('Test Population Density (N_test)')
    axes[3].set_xlabel('Test Population Density (N_test)')
    
    plt.suptitle(f'Transport Mechanics Breakdown (Trained N={n_train}, P={n_packages})', fontsize=16, y=1.02)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'3_Behavioral_Breakdowns_P{n_packages}.pdf'))
    plt.close()

# ---------------------------------------------------------
# Graph 4: 1 vs 2 Package Scaling Comparison
# ---------------------------------------------------------
def plot_package_comparison(df, n_train, output_dir="plots_transport"):
    """Compares the degradation in performance when scaling from 1 to 2 packages."""
    # Combine best configs for both P=1 and P=2
    df_p1 = get_optimal_nmax_df(df, n_train, 1)
    df_p2 = get_optimal_nmax_df(df, n_train, 2)
    df_combined = pd.concat([df_p1, df_p2], ignore_index=True)
    
    if df_combined.empty or df_combined['Packages'].nunique() < 2:
        print("[Info] Need both P=1 and P=2 data for the Package Comparison graph. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    current_palette = {}
    for m in df_combined['Model'].unique():
        base_m = m.replace('_mod', '')
        current_palette[m] = MODEL_COLORS.get(base_m, '#000000')

    # Lineplot with style mapping to packages
    sns.lineplot(data=df_combined, x='N_test', y='Overall_Success_Rate', 
                 hue='Model', style='Packages', palette=current_palette, 
                 markers=['o', 'X'], dashes=True, linewidth=2.5, ax=ax)
                 
    ax.set_title(f'Multi-Object Degradation: 1 vs 2 Packages (N_train={n_train})')
    ax.set_xlabel('Test Population Density (N_test)')
    ax.set_ylabel('Overall Success Rate')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '4_Package_Degradation_Comparison.pdf'))
    plt.close()

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=".", help="Directory containing CSVs")
    parser.add_argument("--output-dir", type=str, default="plots_transport", help="Directory for PDFs")
    parser.add_argument("--baseline-ntrain", type=int, default=30, help="N_train value for line graphs")
    parser.add_argument("--baseline-packages", type=int, default=1, help="Package count for base line graphs")
    
    args = parser.parse_args()
    
    print(f"Scanning '{args.data_dir}' for CSV files...")
    try:
        metrics_df = load_and_aggregate_data(args.data_dir)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()
        
    print(f"Generating Graph 1: Scaling / Survival Curves (P={args.baseline_packages})...")
    plot_scaling_curves(metrics_df, args.baseline_ntrain, args.baseline_packages, args.output_dir)
    
    print(f"Generating Graph 2: Phase Diagrams (P={args.baseline_packages})...")
    plot_phase_diagrams(metrics_df, args.baseline_packages, args.output_dir)
    
    print(f"Generating Graph 3: Behavioral Breakdown Curves (P={args.baseline_packages})...")
    plot_behavioral_breakdowns(metrics_df, args.baseline_ntrain, args.baseline_packages, args.output_dir)

    print("Generating Graph 4: 1 vs 2 Package Scaling Comparison...")
    plot_package_comparison(metrics_df, args.baseline_ntrain, args.output_dir)
    
    print("\n[Success] All Transport graphs generated and saved.")
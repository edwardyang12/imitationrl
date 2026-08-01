import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import argparse

# ---------------------------------------------------------
# Configuration & Styling for Academic Publication
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

# Define a consistent color palette for your architectures
MODEL_COLORS = {
    'MLP': '#d62728',         # Red
    'Transformer': '#1f77b4', # Blue
    'GAT': '#2ca02c'          # Green
}

# ---------------------------------------------------------
# Data Ingestion Hook
# ---------------------------------------------------------
def load_and_aggregate_data(data_dir):
    """
    Scans the specified directory for CSV files. 
    Parses the filename expecting the format: {Model}_{N_train}_{n_max}__...csv
    Example: GAT_6_8__inference_sweep.csv -> Model: GAT, N_train: 6, n_max: 8
    """
    all_dataframes = []
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the directory: '{data_dir}'")
        
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        
        # Parse the filename. Assuming the prefix is split from the rest by '__'
        prefix_part = filename.split('__')[0]
        parts = prefix_part.split('_')
        
        if len(parts) >= 3:
            model_name = parts[0]
            
            try:
                n_train = int(parts[1])
                n_max = int(parts[2])
            except ValueError:
                print(f"[Warning] Could not parse integers for N_train/n_max from file: {filename}. Skipping.")
                continue
                
            # Load the CSV
            df = pd.read_csv(file_path)
            
            # Inject the parsed hyperparameters as new columns
            df['Model'] = model_name
            df['N_train'] = n_train
            df['n_max'] = n_max
            
            all_dataframes.append(df)
        else:
            print(f"[Warning] Filename '{filename}' does not match expected format Model_Ntrain_nmax__... Skipping.")

    if not all_dataframes:
        raise ValueError("No valid data could be aggregated. Check your file names.")
        
    master_df = pd.concat(all_dataframes, ignore_index=True)
    return master_df

# ---------------------------------------------------------
# Graph 1: OOD Survival Curves
# ---------------------------------------------------------
def plot_survival_curves(df, output_dir="plots"):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.lineplot(
        data=df, x='N_test', y='S_rate (Final Goal Retention Rate %)', 
        hue='Model', palette=MODEL_COLORS, marker='o', linewidth=2.5, ax=ax
    )
    
    ax.axhline(80, ls='--', color='gray', alpha=0.7, label='80% Success Threshold')
    ax.set_title('Zero-Shot OOD Scaling Survival')
    ax.set_xlabel('Test Population Density (N_test)')
    ax.set_ylabel('Final Goal Retention Rate (%)')
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '1_Survival_Curves.pdf'))
    plt.close()

# ---------------------------------------------------------
# Graph 2: 2D Phase Diagrams (Heatmaps at N=105)
# ---------------------------------------------------------
def plot_phase_diagrams(df, output_dir="plots"):
    # Filter for the extreme density stress test
    df_105 = df[df['N_test'] == 105].copy()
    
    if df_105.empty:
        print("[Note] No data found for N_test=105. Skipping Graph 2 (Phase Diagrams).")
        return
        
    # Check if there is enough variance to plot a heatmap
    if df_105['n_max'].nunique() < 2 and df_105['N_train'].nunique() < 2:
        print("[Note] Not enough hyperparameter variance for Phase Diagrams. Skipping Graph 2.")
        return
        
    models = df_105['Model'].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    
    # Handle case where only 1 model is found (make axes iterable)
    if len(models) == 1:
        axes = [axes]
    
    for i, model in enumerate(models):
        model_data = df_105[df_105['Model'] == model]
        if model_data.empty:
            continue
            
        pivot = model_data.pivot_table(
            index='N_train', 
            columns='n_max', 
            values='S_rate (Final Goal Retention Rate %)', 
            aggfunc='mean'
        )
        
        sns.heatmap(
            pivot, ax=axes[i], cmap='viridis', vmin=0, vmax=105, 
            annot=True, fmt=".1f", cbar=(i == len(models)-1), cbar_kws={'label': 'Success Rate (%)'}
        )
        axes[i].set_title(f'{model} Scaling Robustness')
        axes[i].invert_yaxis()
        
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '2_Phase_Diagrams.pdf'))
    plt.close()

# ---------------------------------------------------------
# Graph 3: Common Critical Density Benchmark
# ---------------------------------------------------------
def plot_critical_density_bars(df, n_crit=50, output_dir="plots"):
    df_crit = df[df['N_test'] == n_crit].copy()
    
    if df_crit.empty:
        print(f"[Error] No data found for N_crit = {n_crit}. Adjust parameter in script execution to plot Graph 3.")
        return

    # Calculate the Cooperation Ratio
    df_crit['Cooperation Ratio'] = df_crit['Yields_Cooperative'] / (df_crit['Yields_Forced_Displacement'] + 1e-5)
    
    metrics_to_plot = [
        'C_rate (Active Collision Frequency %)',
        'F_rate (Active Deadlock Frequency %)',
        'E_active (Active Mean Energy)',
        'Cooperation Ratio',
        'E_settled_max (Peak Yielding Force)'
    ]
    
    # Normalize metrics using Min-Max scaling for the grouped bar chart
    df_normalized = df_crit.copy()
    for col in metrics_to_plot:
        max_val = df_crit[col].max()
        min_val = df_crit[col].min()
        if max_val != min_val:
            df_normalized[col] = (df_crit[col] - min_val) / (max_val - min_val)
        else:
            df_normalized[col] = 0.0

    df_melted = pd.melt(df_normalized, id_vars=['Model'], value_vars=metrics_to_plot, var_name='Metric', value_name='Normalized Score')
    
    # Format labels to be cleaner for the chart
    df_melted['Metric'] = df_melted['Metric'].apply(lambda x: x.split(' (')[0].replace('_', ' '))

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter palette to only include models present in the data
    current_palette = {k: v for k, v in MODEL_COLORS.items() if k in df_melted['Model'].unique()}
    
    sns.barplot(data=df_melted, x='Metric', y='Normalized Score', hue='Model', palette=current_palette, ax=ax)
    
    ax.set_title(f'Behavioral Strategies at Common Critical Density (N={n_crit})')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
    ax.set_ylabel('Min-Max Normalized Score')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'3_Critical_Density_N{n_crit}.pdf'))
    plt.close()

# ---------------------------------------------------------
# Graph 4: Behavioral Breakdown Curves
# ---------------------------------------------------------
def plot_behavioral_breakdowns(df, output_dir="plots"):
    metrics = [
        ('C_rate (Active Collision Frequency %)', 'Collision Rate (%)'),
        ('F_rate (Active Deadlock Frequency %)', 'Deadlock / Freeze Rate (%)'),
        ('T_conv (Mean Steps to First Arrival)', 'Steps to First Arrival'),
        ('Yields_Cooperative', 'Total Cooperative Yields')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    
    for idx, (metric_col, y_label) in enumerate(metrics):
        ax = axes[idx]
        
        current_palette = {k: v for k, v in MODEL_COLORS.items() if k in df['Model'].unique()}
        
        # Plot the lines
        sns.lineplot(
            data=df, x='N_test', y=metric_col, hue='Model', 
            palette=current_palette, marker='s', ax=ax, legend=(idx==0)
        )
        
        # Methodology Trick: Shade the background where S_rate < 80%
        for model in df['Model'].unique():
            if model in MODEL_COLORS:
                color = MODEL_COLORS[model]
                model_df = df[df['Model'] == model].sort_values('N_test')
                failing_points = model_df[model_df['S_rate (Final Goal Retention Rate %)'] < 80]
                
                if not failing_points.empty:
                    failure_onset_n = failing_points['N_test'].min()
                    ax.axvspan(failure_onset_n, model_df['N_test'].max(), color=color, alpha=0.1)
                    ax.axvline(failure_onset_n, color=color, linestyle=':', alpha=0.8)

        ax.set_title(y_label)
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle='--', alpha=0.5)
        
    axes[2].set_xlabel('Test Population Density (N_test)')
    axes[3].set_xlabel('Test Population Density (N_test)')
    
    plt.suptitle('Pathological Breakdown Mechanics Across Scaling Limits', fontsize=16, y=1.02)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '4_Behavioral_Breakdowns.pdf'))
    plt.close()

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark plots from inference CSVs.")
    parser.add_argument("--data-dir", type=str, default=".", help="Directory containing the CSV files.")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory to save the generated PDF plots.")
    parser.add_argument("--n-crit", type=int, default=50, help="The critical density N to use for Graph 3.")
    
    args = parser.parse_args()
    
    print(f"Scanning '{args.data_dir}' for CSV files...")
    metrics_df = load_and_aggregate_data(args.data_dir)
    print(f"Aggregated {len(metrics_df)} rows of data.")
    
    print("Generating Graph 1: Survival Curves...")
    plot_survival_curves(metrics_df, output_dir=args.output_dir)
    
    print("Generating Graph 2: Phase Diagrams...")
    plot_phase_diagrams(metrics_df, output_dir=args.output_dir)
    
    print(f"Generating Graph 3: Critical Density Comparisons (N={args.n_crit})...")
    plot_critical_density_bars(metrics_df, n_crit=args.n_crit, output_dir=args.output_dir)
    
    print("Generating Graph 4: Behavioral Breakdown Curves...")
    plot_behavioral_breakdowns(metrics_df, output_dir=args.output_dir)
    
    print(f"\n[Success] All graphs generated and saved to the '{args.output_dir}' directory as high-res PDFs.")
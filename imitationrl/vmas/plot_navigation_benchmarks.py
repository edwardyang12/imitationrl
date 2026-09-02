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
    'MLP': '#d62728',         
    'T': '#1f77b4', 
    'GAT': '#2ca02c',          
    'GCN': "#89a02c",          
    'PTN': "#a02c89"          
}

# Custom High-Contrast Colormap for Phase Diagrams
# Black (0%) -> Dark Red (10%) -> Yellow/Orange (80%) -> Green (90-100%)
c_nodes = [0.0, 0.1, 0.8, 0.9, 1.0]
c_colors = ["#1a1a1a", "#d62728", "#ffc107", "#2ca02c", "#1e7a1e"]
PHASE_CMAP = LinearSegmentedColormap.from_list("custom_phase", list(zip(c_nodes, c_colors)))

# ---------------------------------------------------------
# Data Ingestion
# ---------------------------------------------------------
def load_and_aggregate_data(data_dir):
    all_dataframes = []
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}'")
        
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        prefix_part = filename.split('__')[0]
        parts = prefix_part.split('_')
        
        if len(parts) >= 3:
            model_name = parts[0]
            try:
                n_train = int(parts[1])
                n_max = int(parts[2])
            except ValueError:
                continue
                
            df = pd.read_csv(file_path)
            df['Model'] = model_name
            df['N_train'] = n_train
            df['n_max'] = n_max
            all_dataframes.append(df)

    if not all_dataframes:
        raise ValueError("No valid data could be aggregated.")
        
    return pd.concat(all_dataframes, ignore_index=True)

# ---------------------------------------------------------
# Export Top Configurations
# ---------------------------------------------------------
def export_top_configs(df, output_dir="plots"):
    df_105 = df[df['N_test'] == 105].copy()
    if df_105.empty: return
    
    summary = []
    for model in df_105['Model'].unique():
        model_data = df_105[df_105['Model'] == model]
        # Sort by Success Rate descending
        top_3 = model_data.sort_values(by='S_rate (Final Goal Retention Rate %)', ascending=False).head(3)
        
        for i, (_, row) in enumerate(top_3.iterrows()):
            summary.append({
                'Model': model,
                'Rank': i + 1,
                'N_train': row['N_train'],
                'n_max': row['n_max'],
                'Delta (n_max - N_train)': row['n_max'] - row['N_train'],
                'Success Rate (%)': round(row['S_rate (Final Goal Retention Rate %)'], 2)
            })
            
    summary_df = pd.DataFrame(summary)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'top_configs_summary.csv')
    summary_df.to_csv(out_path, index=False)
    print(f"\n[Success] Exported top hyperparameter configurations to: {out_path}")

# ---------------------------------------------------------
# Graph 1: OOD Survival Curves
# ---------------------------------------------------------
def plot_survival_curves(df, n_train, n_max, output_dir="plots"):
    df_base = df[(df['N_train'] == n_train) & (df['n_max'] == n_max)].copy()
    
    if df_base.empty:
        print(f"[Warning] No data for baseline config (N_train={n_train}, n_max={n_max}). Skipping Graph 1.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    current_palette = {k: v for k, v in MODEL_COLORS.items() if k in df_base['Model'].unique()}
    
    sns.lineplot(
        data=df_base, x='N_test', y='S_rate (Final Goal Retention Rate %)', 
        hue='Model', palette=current_palette, marker='o', linewidth=2.5, ax=ax
    )
    
    ax.axhline(80, ls='--', color='gray', alpha=0.7, label='80% Success Threshold')
    ax.set_title(f'Zero-Shot Scaling Survival (Trained N={n_train}, n_max={n_max})')
    ax.set_xlabel('Test Population Density (N_test)')
    ax.set_ylabel('Final Goal Retention Rate (%)')
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '1_Survival_Curves.pdf'))
    plt.close()

# ---------------------------------------------------------
# Graph 2a: Phase Diagram Heatmaps (Custom Colormap)
# ---------------------------------------------------------
def plot_phase_diagrams(df, output_dir="plots"):
    df_105 = df[df['N_test'] == 105].copy()
    
    if df_105.empty or (df_105['n_max'].nunique() < 2 and df_105['N_train'].nunique() < 2):
        return
        
    models = df_105['Model'].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    if len(models) == 1: axes = [axes]
    
    full_n_train = sorted(df_105['N_train'].unique())
    full_n_max = sorted(df_105['n_max'].unique())
    
    for i, model in enumerate(models):
        model_data = df_105[df_105['Model'] == model]
        pivot = model_data.pivot_table(index='N_train', columns='n_max', values='S_rate (Final Goal Retention Rate %)', aggfunc='mean')
        
        idx = pd.Index(full_n_train, name='N_train')
        cols = pd.Index(full_n_max, name='n_max')
        pivot = pivot.reindex(index=idx, columns=cols).fillna(0.0)
        
        if 30 in pivot.index and 15 in pivot.columns:
            pivot.loc[30, 15] = np.nan
        
        # Apply the new high-contrast colormap
        sns.heatmap(
            pivot, ax=axes[i], cmap=PHASE_CMAP, vmin=0, vmax=100, 
            annot=True, fmt=".1f", cbar=(i == len(models)-1), cbar_kws={'label': 'Success Rate (%)'}
        )
        axes[i].set_title(f'{model} Robustness (N=105)')
        axes[i].invert_yaxis()
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2a_Phase_Diagrams.pdf'))
    plt.close()

# ---------------------------------------------------------
# Graph 2b: Hyperparameter Trend Lines (The Delta Shift)
# ---------------------------------------------------------
def plot_hyperparameter_trends(df, output_dir="plots"):
    df_105 = df[df['N_test'] == 105].copy()
    if df_105.empty: return

    # Calculate the Delta metric
    df_105['Delta (n_max - N_train)'] = df_105['n_max'] - df_105['N_train']

    models = df_105['Model'].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    if len(models) == 1: axes = [axes]

    for i, model in enumerate(models):
        model_data = df_105[df_105['Model'] == model].copy()
        
        sns.lineplot(
            data=model_data, x='Delta (n_max - N_train)', y='S_rate (Final Goal Retention Rate %)', 
            hue='N_train', palette='tab10', marker='o', ax=axes[i]
        )
        
        axes[i].set_title(f'{model} Parameter Trends')
        axes[i].set_ylim(0, 105)
        # Add a vertical line at Delta = 0 to show the exact cliff edge
        axes[i].axvline(0, color='red', linestyle=':', alpha=0.6, label='Capacity Cliff (Delta = 0)')
        if i == 0: axes[i].legend(loc='lower left')
        axes[i].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2b_Hyperparameter_Trends.pdf'))
    plt.close()

# ---------------------------------------------------------
# Graph 3: Critical Density Benchmark (Visually Grouped)
# ---------------------------------------------------------
def plot_critical_density_bars(df, n_crit=50, output_dir="plots"):
    df_crit = df[df['N_test'] == n_crit].copy()
    if df_crit.empty: return

    df_crit['Cooperation Ratio'] = df_crit['Yields_Cooperative'] / (df_crit['Yields_Forced_Displacement'] + 1e-5)
    
    # Reordered logically: Failure mechanisms on the left, Success mechanisms on the right
    metrics_to_plot = [
        'F_rate (Active Deadlock Frequency %)',
        'C_rate (Active Collision Frequency %)',
        'E_active (Active Mean Energy)',
        'Cooperation Ratio',
        'E_settled_max (Peak Yielding Force)'
    ]
    
    df_normalized = df_crit.copy()
    for col in metrics_to_plot:
        max_val = df_crit[col].max()
        df_normalized[col] = df_crit[col] / max_val if max_val > 0 else 0.0

    df_melted = pd.melt(df_normalized, id_vars=['Model'], value_vars=metrics_to_plot, var_name='Metric', value_name='Max-Normalized Score')
    df_melted['Metric'] = df_melted['Metric'].apply(lambda x: x.split(' (')[0].replace('_', ' '))

    fig, ax = plt.subplots(figsize=(12, 6))
    current_palette = {k: v for k, v in MODEL_COLORS.items() if k in df_melted['Model'].unique()}
    
    sns.barplot(data=df_melted, x='Metric', y='Max-Normalized Score', hue='Model', palette=current_palette, ax=ax)
    
    # Add visual divider between Failure Modes and Success Modes
    ax.axvline(2.5, color='black', linestyle='-', alpha=0.8, linewidth=1.5)
    
    # Add textual region labels
    ax.text(1, 1.05, 'Pathological Failure Indicators', ha='center', va='bottom', transform=ax.get_xaxis_transform(), fontweight='bold', fontsize=12)
    ax.text(3.5, 1.05, 'Emergent Success Indicators', ha='center', va='bottom', transform=ax.get_xaxis_transform(), fontweight='bold', fontsize=12)
    
    ax.set_title(f'Behavioral Strategies at Common Critical Density (N={n_crit})', y=1.12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
    ax.set_ylabel('Proportion of Maximum Observed Value')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'3_Critical_Density_N{n_crit}.pdf'))
    plt.close()

# ---------------------------------------------------------
# Graph 4: Behavioral Breakdowns
# ---------------------------------------------------------
def plot_behavioral_breakdowns(df, n_train, n_max, output_dir="plots"):
    df_base = df[(df['N_train'] == n_train) & (df['n_max'] == n_max)].copy()
    if df_base.empty: return

    metrics = [
        ('C_rate (Active Collision Frequency %)', 'Collision Rate (%)'),
        ('F_rate (Active Deadlock Frequency %)', 'Deadlock / Freeze Rate (%)'),
        ('T_conv (Mean Steps to First Arrival)', 'Steps to First Arrival'),
        ('Yields_Cooperative', 'Total Cooperative Yields')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    current_palette = {k: v for k, v in MODEL_COLORS.items() if k in df_base['Model'].unique()}
    
    for idx, (metric_col, y_label) in enumerate(metrics):
        ax = axes[idx]
        
        sns.lineplot(
            data=df_base, x='N_test', y=metric_col, hue='Model', 
            palette=current_palette, marker='s', ax=ax, legend=(idx==0)
        )
        
        for model in df_base['Model'].unique():
            if model in MODEL_COLORS:
                color = MODEL_COLORS[model]
                model_df = df_base[df_base['Model'] == model].sort_values('N_test')
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
    
    plt.suptitle(f'Pathological Breakdown Mechanics (Trained N={n_train}, n_max={n_max})', fontsize=16, y=1.02)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '4_Behavioral_Breakdowns.pdf'))
    plt.close()

def plot_saturation_curve(df, output_dir="plots"):
    # Focus exclusively on the extreme density stress test
    df_105 = df[df['N_test'] == 105].copy()
    if df_105.empty: return

    fig, ax = plt.subplots(figsize=(8, 6))
    current_palette = {k: v for k, v in MODEL_COLORS.items() if k in df_105['Model'].unique()}
    
    # Seaborn automatically averages across the N_train values and plots the confidence interval
    sns.lineplot(
        data=df_105, x='n_max', y='S_rate (Final Goal Retention Rate %)', 
        hue='Model', palette=current_palette, marker='o', linewidth=2.5, errorbar=('ci', 95), ax=ax
    )
    
    # Mark the saturation threshold
    ax.axhline(95, ls='--', color='gray', alpha=0.7, label='95% Saturation Threshold')
    
    ax.set_title('Context Window Saturation (Evaluated at N=105)')
    ax.set_xlabel('Context Window Size (n_max)')
    ax.set_ylabel('Mean Success Rate (%) Across All N_train')
    ax.set_ylim(0, 105)
    
    # Ensure x-axis ticks match your discrete n_max values
    ax.set_xticks(sorted(df_105['n_max'].unique()))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '5_Saturation_Curve.pdf'))
    plt.close()

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=".", help="Directory containing CSVs")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory for PDFs")
    parser.add_argument("--n-crit", type=int, default=50, help="Critical density N for Graph 3")
    parser.add_argument("--baseline-ntrain", type=int, default=10, help="N_train value for line graphs")
    parser.add_argument("--baseline-nmax", type=int, default=5, help="n_max value for line graphs")
    
    args = parser.parse_args()
    
    print(f"Scanning '{args.data_dir}' for CSV files...")
    metrics_df = load_and_aggregate_data(args.data_dir)
    
    export_top_configs(metrics_df, args.output_dir)
    
    print("Generating Graph 1: Survival Curves...")
    plot_survival_curves(metrics_df, args.baseline_ntrain, args.baseline_nmax, args.output_dir)
    
    print("Generating Graph 2a: Phase Diagrams...")
    plot_phase_diagrams(metrics_df, args.output_dir)
    
    print("Generating Graph 2b: Hyperparameter Trends...")
    plot_hyperparameter_trends(metrics_df, args.output_dir)
    
    print(f"Generating Graph 3: Critical Density Comparisons (N={args.n_crit})...")
    plot_critical_density_bars(metrics_df, args.n_crit, args.output_dir)
    
    print("Generating Graph 4: Behavioral Breakdown Curves...")
    plot_behavioral_breakdowns(metrics_df, args.baseline_ntrain, args.baseline_nmax, args.output_dir)

    print("Generating Graph 5: Saturation Curve...")
    plot_saturation_curve(metrics_df, args.output_dir)
    
    print("\n[Success] All graphs generated and saved.")
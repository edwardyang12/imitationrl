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

# High-Contrast Colormap for Error (Lower is better: Green -> Red -> Black)
c_nodes = [0.0, 0.3, 0.5, 0.8, 1.0]
c_colors = ["#1e7a1e", "#2ca02c", "#ffc107", "#d62728", "#1a1a1a"]
ERROR_CMAP = LinearSegmentedColormap.from_list("error_phase", list(zip(c_nodes, c_colors)))

# ---------------------------------------------------------
# Data Ingestion
# ---------------------------------------------------------
def load_and_aggregate_data(data_dir):
    all_dataframes = []
    csv_files = glob.glob(os.path.join(data_dir, "*flocking_metrics.csv"))
    if not csv_files:
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
            
            gr_col = [c for c in df.columns if c.startswith('g(r)_Peak')]
            if gr_col:
                df['Lattice_Structure'] = df[gr_col[0]]
                
            all_dataframes.append(df)

    if not all_dataframes:
        raise ValueError("No valid data could be aggregated.")
        
    return pd.concat(all_dataframes, ignore_index=True)

# ---------------------------------------------------------
# Export Top Configurations
# ---------------------------------------------------------
def export_top_configs(df, output_dir="plots_flocking"):
    max_test = df['N_test'].max()
    df_max = df[df['N_test'] == max_test].copy()
    if df_max.empty: return
    
    summary = []
    for model in df_max['Model'].unique():
        model_data = df_max[df_max['Model'] == model]
        # Sort by Tracking Error ascending (Primary emergent metric)
        top_3 = model_data.sort_values(by='Mean_Tracking_Error', ascending=True).head(3)
        
        for i, (_, row) in enumerate(top_3.iterrows()):
            summary.append({
                'Model': model,
                'Rank': i + 1,
                'N_train': row['N_train'],
                'n_max': row['n_max'],
                'Delta (n_max - N_train)': row['n_max'] - row['N_train'],
                'Mean Tracking Error': round(row['Mean_Tracking_Error'], 4),
                'Mean Polarization': round(row['Mean_Polarization'], 4)
            })
            
    summary_df = pd.DataFrame(summary)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'top_configs_summary.csv')
    summary_df.to_csv(out_path, index=False)
    print(f"\n[Success] Exported top hyperparameter configurations to: {out_path}")

# ---------------------------------------------------------
# Graph 1: Scaling / Survival Curves
# ---------------------------------------------------------
def plot_scaling_curves(df, n_train, n_max, output_dir="plots_flocking"):
    df_base = df[(df['N_train'] == n_train) & (df['n_max'] == n_max)].copy()
    
    if df_base.empty:
        print(f"[Warning] No data for baseline config (N_train={n_train}, n_max={n_max}). Skipping Graph 1.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    current_palette = {k: v for k, v in MODEL_COLORS.items() if k in df_base['Model'].unique()}
    
    sns.lineplot(
        data=df_base, x='N_test', y='Mean_Tracking_Error', 
        hue='Model', palette=current_palette, marker='o', linewidth=2.5, ax=ax
    )
    
    ax.axhline(0.40, ls='--', color='gray', alpha=0.7, label='Optimal Tracking Threshold (0.4)')
    ax.set_title(f'Zero-Shot Scaling: Target Tracking (Trained N={n_train}, n_max={n_max})')
    ax.set_xlabel('Test Population Density (N_test)')
    ax.set_ylabel('Mean Target Tracking Error (Lower is Better)')
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '1_Scaling_Tracking_Error.pdf'))
    plt.close()

# ---------------------------------------------------------
# Graph 2a: Phase Diagram Heatmaps
# ---------------------------------------------------------
def plot_phase_diagrams(df, output_dir="plots_flocking"):
    max_test = df['N_test'].max()
    df_max = df[df['N_test'] == max_test].copy()
    
    if df_max.empty or (df_max['n_max'].nunique() < 2 and df_max['N_train'].nunique() < 2):
        return
        
    models = df_max['Model'].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    if len(models) == 1: axes = [axes]
    
    full_n_train = sorted(df_max['N_train'].unique())
    full_n_max = sorted(df_max['n_max'].unique())
    
    for i, model in enumerate(models):
        model_data = df_max[df_max['Model'] == model]
        pivot = model_data.pivot_table(index='N_train', columns='n_max', values='Mean_Tracking_Error', aggfunc='mean')
        
        idx = pd.Index(full_n_train, name='N_train')
        cols = pd.Index(full_n_max, name='n_max')
        # Fill missing with a high error penalty (e.g., 1.5) for contrast
        pivot = pivot.reindex(index=idx, columns=cols).fillna(1.5)
        
        sns.heatmap(
            pivot, ax=axes[i], cmap=ERROR_CMAP, vmin=0.2, vmax=1.5, 
            annot=True, fmt=".2f", cbar=(i == len(models)-1), cbar_kws={'label': 'Tracking Error'}
        )
        axes[i].set_title(f'{model} Robustness (N={max_test})')
        axes[i].invert_yaxis()
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2a_Phase_Diagrams.pdf'))
    plt.close()

# ---------------------------------------------------------
# Graph 2b: Hyperparameter Trend Lines
# ---------------------------------------------------------
def plot_hyperparameter_trends(df, output_dir="plots_flocking"):
    max_test = df['N_test'].max()
    df_max = df[df['N_test'] == max_test].copy()
    if df_max.empty: return

    df_max['Delta (n_max - N_train)'] = df_max['n_max'] - df_max['N_train']

    models = df_max['Model'].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    if len(models) == 1: axes = [axes]

    for i, model in enumerate(models):
        model_data = df_max[df_max['Model'] == model].copy()
        
        sns.lineplot(
            data=model_data, x='Delta (n_max - N_train)', y='Mean_Tracking_Error', 
            hue='N_train', palette='tab10', marker='o', ax=axes[i]
        )
        
        axes[i].set_title(f'{model} Parameter Trends')
        axes[i].set_ylim(bottom=0.0)
        axes[i].axvline(0, color='red', linestyle=':', alpha=0.6, label='Capacity Cliff (Delta = 0)')
        if i == 0: axes[i].legend(loc='upper right')
        axes[i].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2b_Hyperparameter_Trends.pdf'))
    plt.close()

# ---------------------------------------------------------
# Graph 3: Critical Density Benchmark
# ---------------------------------------------------------
def plot_critical_density_bars(df, n_crit=50, output_dir="plots_flocking"):
    df_crit = df[df['N_test'] == n_crit].copy()
    if df_crit.empty:
        n_crit = df['N_test'].unique()[len(df['N_test'].unique())//2]
        df_crit = df[df['N_test'] == n_crit].copy()

    metrics_to_plot = [
        'Mean_Collision_Rate',
        'Mean_Tracking_Error',
        'Mean_Action_Jitter',
        'Mean_Isolation_Rate',
        'Mean_Polarization',
        'Lattice_Structure'
    ]
    
    metrics_to_plot = [m for m in metrics_to_plot if m in df_crit.columns]
    
    df_normalized = df_crit.copy()
    for col in metrics_to_plot:
        max_val = df_crit[col].max()
        df_normalized[col] = df_crit[col] / max_val if max_val > 0 else 0.0

    df_melted = pd.melt(df_normalized, id_vars=['Model'], value_vars=metrics_to_plot, var_name='Metric', value_name='Max-Normalized Score')
    df_melted['Metric'] = df_melted['Metric'].apply(lambda x: x.replace('Mean_', '').replace('_', ' '))

    fig, ax = plt.subplots(figsize=(12, 6))
    current_palette = {k: v for k, v in MODEL_COLORS.items() if k in df_melted['Model'].unique()}
    
    sns.barplot(data=df_melted, x='Metric', y='Max-Normalized Score', hue='Model', palette=current_palette, ax=ax)
    
    ax.axvline(3.5, color='black', linestyle='-', alpha=0.8, linewidth=1.5)
    ax.text(1.5, 1.05, 'Pathological Failure Indicators', ha='center', va='bottom', transform=ax.get_xaxis_transform(), fontweight='bold', fontsize=12)
    ax.text(4.5, 1.05, 'Emergent Success Indicators', ha='center', va='bottom', transform=ax.get_xaxis_transform(), fontweight='bold', fontsize=12)
    
    ax.set_title(f'Flocking Mechanics at Critical Density (N={n_crit})', y=1.12)
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
def plot_behavioral_breakdowns(df, n_train, n_max, output_dir="plots_flocking"):
    df_base = df[(df['N_train'] == n_train) & (df['n_max'] == n_max)].copy()
    if df_base.empty: return

    metrics = [
        ('Mean_Tracking_Error', 'Target Tracking Error'),
        ('Mean_Collision_Rate', 'Collision Rate'),
        ('Mean_Cohesion_Spread', 'Cohesion Spread'),
        ('Lattice_Structure', 'Lattice Structure (g(r))')
    ]
    
    metrics = [(m, label) for m, label in metrics if m in df_base.columns]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    current_palette = {k: v for k, v in MODEL_COLORS.items() if k in df_base['Model'].unique()}
    
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
    
    plt.suptitle(f'Flocking Breakdown Mechanics (Trained N={n_train}, n_max={n_max})', fontsize=16, y=1.02)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '4_Behavioral_Breakdowns.pdf'))
    plt.close()

# ---------------------------------------------------------
# Graph 5: Pareto Frontier
# ---------------------------------------------------------
def plot_behavioral_pareto_frontier(df, n_train, n_max, output_dir="plots_flocking"):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    df_base = df[(df['N_train'] == n_train) & (df['n_max'] == n_max)].copy()
    
    if df_base.empty:
        print(f"[Warning] No data for Pareto config (N_train={n_train}, n_max={n_max}).")
        return

    df_pareto = df_base.groupby(['Model', 'N_test'])[['Mean_Tracking_Error', 'Mean_Collision_Rate']].mean().reset_index()
    current_palette = {k: v for k, v in MODEL_COLORS.items() if k in df_pareto['Model'].unique()}
    
    sns.scatterplot(
        data=df_pareto, 
        x='Mean_Tracking_Error', 
        y='Mean_Collision_Rate', 
        hue='Model', 
        size='N_test',
        sizes=(50, 250),
        palette=current_palette, 
        alpha=0.8,
        ax=ax
    )
    
    for model in df_pareto['Model'].unique():
        model_data = df_pareto[df_pareto['Model'] == model].sort_values('N_test')
        ax.plot(
            model_data['Mean_Tracking_Error'], 
            model_data['Mean_Collision_Rate'], 
            color=MODEL_COLORS.get(model, 'black'), 
            linestyle='-', 
            alpha=0.4
        )
        
    ax.set_title(f'Efficiency Trade-off (Trained N={n_train}, n_max={n_max})')
    ax.set_xlabel('Mean Target Tracking Error (Lower is Better)')
    ax.set_ylabel('Mean Collision Rate (Lower is Better)')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '5_Pareto_Frontier.pdf'), bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------
# Graph 6: n_max Saturation
# ---------------------------------------------------------
def plot_nmax_saturation(df, n_train=20, n_test=105, output_dir="plots_flocking"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    df_sat = df[(df['N_train'] == n_train) & (df['N_test'] == n_test)].copy()
    
    if df_sat.empty:
        print(f"[Warning] No data for n_max saturation config.")
        return
        
    current_palette = {k: v for k, v in MODEL_COLORS.items() if k in df_sat['Model'].unique()}
    
    sns.lineplot(data=df_sat, x='n_max', y='Mean_Tracking_Error', hue='Model', 
                 palette=current_palette, marker='o', linewidth=2.5, ax=ax1)
    ax1.set_title(f'Tracking Error Saturation (N_train={n_train} → N_test={n_test})')
    ax1.set_xlabel('Maximum Neighbors Observed (n_max)')
    ax1.set_ylabel('Mean Target Tracking Error')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    sns.lineplot(data=df_sat, x='n_max', y='Mean_Collision_Rate', hue='Model', 
                 palette=current_palette, marker='o', linewidth=2.5, ax=ax2)
    ax2.set_title(f'Collision Rate Saturation (N_train={n_train} → N_test={n_test})')
    ax2.set_xlabel('Maximum Neighbors Observed (n_max)')
    ax2.set_ylabel('Mean Collision Rate')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '6_nmax_Saturation.pdf'))
    plt.close()

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=".", help="Directory containing CSVs")
    parser.add_argument("--output-dir", type=str, default="plots_flocking", help="Directory for PDFs")
    parser.add_argument("--n-crit", type=int, default=80, help="Critical density N for Graph 3")
    parser.add_argument("--baseline-ntrain", type=int, default=20, help="N_train value for line graphs")
    parser.add_argument("--baseline-nmax", type=int, default=5, help="n_max value for line graphs")
    
    args = parser.parse_args()
    
    print(f"Scanning '{args.data_dir}' for CSV files...")
    metrics_df = load_and_aggregate_data(args.data_dir)
    
    export_top_configs(metrics_df, args.output_dir)
    
    print("Generating Graph 1: Scaling / Survival Curves...")
    plot_scaling_curves(metrics_df, args.baseline_ntrain, args.baseline_nmax, args.output_dir)
    
    print("Generating Graph 2a: Phase Diagrams...")
    plot_phase_diagrams(metrics_df, args.output_dir)
    
    print("Generating Graph 2b: Hyperparameter Trends...")
    plot_hyperparameter_trends(metrics_df, args.output_dir)
    
    print(f"Generating Graph 3: Critical Density Comparisons (N={args.n_crit})...")
    plot_critical_density_bars(metrics_df, args.n_crit, args.output_dir)
    
    print("Generating Graph 4: Behavioral Breakdown Curves...")
    plot_behavioral_breakdowns(metrics_df, args.baseline_ntrain, args.baseline_nmax, args.output_dir)

    print("Generating Graph 5: Pareto Frontier...")
    plot_behavioral_pareto_frontier(metrics_df, args.baseline_ntrain, args.baseline_nmax, args.output_dir)
    
    print("Generating Graph 6: n_max Saturation...")
    max_test_val = metrics_df['N_test'].max() if not metrics_df.empty else 105
    plot_nmax_saturation(metrics_df, n_train=args.baseline_ntrain, n_test=max_test_val, output_dir=args.output_dir)
    
    print("\n[Success] All flocking graphs generated and saved.")
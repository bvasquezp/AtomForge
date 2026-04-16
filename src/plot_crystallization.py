import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_emergent_crystallization(csv_path="logs/training_metrics.csv"):
    df = pd.read_csv(csv_path)
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Training Step', fontweight='bold')
    ax1.set_ylabel(r'Symmetry Loss $\mathcal{L}_{sym}$ [Å] (Log Scale)', fontweight='bold')
    
    sns.lineplot(data=df, x='step', y='W_Zn', ax=ax1, color='#1f77b4', label=r'$\mathcal{W}_{\epsilon}^{(Zn)}$')
    sns.lineplot(data=df, x='step', y='W_C', ax=ax1, color='#2ca02c', label=r'$\mathcal{W}_{\epsilon}^{(C)}$')
    sns.lineplot(data=df, x='step', y='W_H', ax=ax1, color='#7f7f7f', label=r'$\mathcal{W}_{\epsilon}^{(H)}$')
    
    # Eje Secundario
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x='step', y='gamma', ax=ax2, color='purple', linestyle=':', label=r'Fuerza $\gamma(t)$')
    sns.lineplot(data=df, x='step', y='divergence', ax=ax2, color='darkorange', linestyle='-.', label=r'$-\text{div}(v_\theta)$')
    
    # Make sure folder exists
    Path("logs").mkdir(exist_ok=True)
    plt.savefig("logs/phase_transition_diagram.png", bbox_inches='tight')
    print("Grafico guardado en logs/phase_transition_diagram.png")

if __name__ == "__main__":
    plot_emergent_crystallization()

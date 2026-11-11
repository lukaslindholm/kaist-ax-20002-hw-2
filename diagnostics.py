# AI Usage Declaration for KAIST AX20002:
# Parts of this code were generated with AI assistance.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set up paths
DATA = Path("data")

def run_diagnostics():
    print("Loading data for diagnostics...")
    
    if not DATA.exists():
        print(f"Error: Could not find the directory '{DATA}'.")
        return

    train_df = pd.read_csv(DATA / "train.csv")
    test_df = pd.read_csv(DATA / "test.csv")
    
    z = np.load(DATA / "images.npz", allow_pickle=True)
    Xtr = z["X_train"].astype(np.float64) / 255.0  
    Xte = z["X_test"].astype(np.float64) / 255.0   
    
    test_df['label'] = -1
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    X_all = np.concatenate([Xtr, Xte], axis=0)

    # ---------------------------------------------------------
    # 1. Analyze Label Shift with Annotated Percentages
    # ---------------------------------------------------------
    print("\n--- CLASS DISTRIBUTION PER STAGE (Stages 1-3) ---")
    
    # Calculate percentage of each class per stage
    class_dist = pd.crosstab(train_df['stage'], train_df['label'], normalize='index') * 100
    print(class_dist.round(1))
    
    # Plotting Label Shift
    ax = class_dist.plot(
        kind='bar', stacked=True, figsize=(10, 7), colormap='viridis', rot=0
    )
    plt.title('Defect Class Distribution across Stages 1-3 (Label Shift)')
    plt.xlabel('Stage')
    plt.ylabel('Percentage (%)')
    plt.legend(title='Defect Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add percentage text inside the bars
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy() 
        # Only add text if the segment is large enough to fit the text (e.g. > 3%)
        if height > 3:
            ax.text(x + width/2, 
                    y + height/2, 
                    f'{height:.1f}%', 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    color='black', # White text usually looks good on viridis colors
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                    fontsize=9,
                    fontweight='bold')

    plt.tight_layout()
    plt.savefig('label_shift.png')
    plt.close()
    print("Saved annotated 'label_shift.png' to current directory.")

    # ---------------------------------------------------------
    # 2. Analyze Covariate Shift (Image statistics across stages 1-5)
    # ---------------------------------------------------------
    print("\n--- IMAGE STATISTICS PER STAGE (Stages 1-5) ---")
    
    stats_list = []
    for stage in range(1, 6):
        stage_mask = df_all['stage'] == stage
        images_in_stage = X_all[stage_mask]
        
        mean_brightness = images_in_stage.mean(axis=(1, 2)).mean()
        mean_contrast = images_in_stage.std(axis=(1, 2)).mean()
        
        stats_list.append({
            'Stage': stage,
            'Mean Brightness': mean_brightness,
            'Mean Contrast': mean_contrast
        })

    stats_df = pd.DataFrame(stats_list)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(stats_df['Stage'], stats_df['Mean Brightness'], 'g-o', linewidth=2, label='Mean Brightness')
    ax1.set_xlabel('Stage')
    ax1.set_ylabel('Mean Brightness', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.set_xticks([1, 2, 3, 4, 5])
    
    ax2 = ax1.twinx()
    ax2.plot(stats_df['Stage'], stats_df['Mean Contrast'], 'b-s', linewidth=2, label='Mean Contrast')
    ax2.set_ylabel('Mean Contrast (Standard Deviation)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    plt.title('Image Statistics Trend over Time (Covariate Shift)')
    plt.axvline(x=3.5, color='r', linestyle='--', label='Train/Test Split')
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines = lines_1 + lines_2 + [plt.Line2D([0], [0], color='r', linestyle='--')]
    labels = labels_1 + labels_2 + ['Train/Test Split']
    ax1.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('covariate_shift.png')
    plt.close()
    print("Saved 'covariate_shift.png' to current directory.")
    print("\nDiagnostics complete! Check your PNG files.")

if __name__ == "__main__":
    run_diagnostics()
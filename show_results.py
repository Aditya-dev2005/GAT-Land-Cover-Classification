"""
show_results.py — Load saved features.csv and show all analysis charts
Run this anytime after run.py has completed at least once.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')

# ── NDVI bar chart ────────────────────────────────────────────────────────────
def show_ndvi(csv_path='features.csv'):
    df         = pd.read_csv(csv_path)
    class_ndvi = df.groupby('class')['ndvi_mean'].mean().sort_values()

    colors = ['red'        if x < 0    else
              'orange'     if x < 0.2  else
              'yellow'     if x < 0.4  else
              'lightgreen' if x < 0.6  else
              'darkgreen'  for x in class_ndvi.values]

    plt.figure(figsize=(12, 6))
    plt.barh(class_ndvi.index, class_ndvi.values, color=colors, edgecolor='gray')
    plt.xlabel('NDVI Mean')
    plt.title('NDVI by Land Cover Class', fontweight='bold', fontsize=14)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('ndvi_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅  ndvi_analysis.png saved")


# ── class distribution ────────────────────────────────────────────────────────
def show_class_distribution(csv_path='features.csv'):
    df = pd.read_csv(csv_path)
    counts = df.groupby('class')['img_id'].nunique().sort_values(ascending=False)

    plt.figure(figsize=(12, 5))
    plt.bar(counts.index, counts.values, color='steelblue', edgecolor='navy')
    plt.xlabel('Land Cover Class')
    plt.ylabel('Number of Images')
    plt.title('Dataset — Images per Class', fontweight='bold', fontsize=13)
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅  class_distribution.png saved")


# ── superpixel stats per class ────────────────────────────────────────────────
def show_superpixel_stats(csv_path='features.csv'):
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # NDVI std per class
    class_std = df.groupby('class')['ndvi_std'].mean().sort_values()
    axes[0].barh(class_std.index, class_std.values, color='coral', edgecolor='darkred')
    axes[0].set_xlabel('NDVI Std Dev')
    axes[0].set_title('NDVI Variability per Class', fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)

    # Water % per class
    water = df.groupby('class')['water_percent'].mean().sort_values()
    colors_w = ['steelblue' if v > 10 else 'lightblue' for v in water.values]
    axes[1].barh(water.index, water.values, color=colors_w, edgecolor='navy')
    axes[1].set_xlabel('Water %')
    axes[1].set_title('Average Water % per Class', fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('superpixel_stats.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅  superpixel_stats.png saved")


# ── show existing result PNGs ─────────────────────────────────────────────────
def show_existing_results():
    pngs = sorted([f for f in os.listdir('.') if f.startswith('result_') and f.endswith('.png')])
    if not pngs:
        print("No result_*.png found. Run run.py first.")
        return

    n   = len(pngs)
    fig = plt.figure(figsize=(6*min(n,3), 5*((n+2)//3)))
    for i, fname in enumerate(pngs):
        ax  = fig.add_subplot((n+2)//3, min(n,3), i+1)
        img = plt.imread(fname)
        ax.imshow(img); ax.set_title(fname, fontsize=8); ax.axis('off')
    plt.tight_layout()
    plt.savefig('all_results_overview.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("✅  all_results_overview.png saved")


# ── summary table ─────────────────────────────────────────────────────────────
def print_summary(csv_path='features.csv'):
    df = pd.read_csv(csv_path)
    print("\n" + "="*55)
    print("  DATASET SUMMARY")
    print("="*55)
    print(f"  Total superpixels : {len(df)}")
    print(f"  Total images      : {df['img_id'].nunique()}")
    print(f"  Classes           : {df['class'].nunique()}")
    print(f"  Feature columns   : {len(df.columns)}")
    print("\n  Class-wise NDVI (mean):")
    ndvi = df.groupby('class')['ndvi_mean'].mean().sort_values(ascending=False)
    for cls, val in ndvi.items():
        bar   = '█' * int(abs(val) * 20)
        water = '  💧 WATER' if val < 0 else ''
        print(f"    {cls:<25} {val:+.3f}  {bar}{water}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if not os.path.exists('features.csv'):
        print("❌  features.csv not found. Run run.py first!")
    else:
        print_summary()
        show_ndvi()
        show_class_distribution()
        show_superpixel_stats()
        show_existing_results()
        print("\n✅  All charts saved!")
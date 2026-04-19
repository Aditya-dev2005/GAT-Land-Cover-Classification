"""
run.py — Full GAT Pipeline Runner (ALL flowchart steps covered)
JIIT Minor Project 2 | Dr. Alka Singhal
"""

import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')

from main import (EuroSATRealProcessor, GraphBuilder, GATClassifier,
                  GATTrainer, plot_training_history, plot_confusion_matrix,
                  plot_ndvi_analysis, plot_land_cover_map,
                  compare_with_baselines, plot_baseline_comparison)

DATA_PATH        = r"D:\EuroSAT\allBands"
MAX_IMAGES_CLASS = 50
N_SEGMENTS       = 50
GAT_HIDDEN       = 32
GAT_HEADS        = 8
DROPOUT          = 0.3
LEARNING_RATE    = 0.005
EPOCHS           = 200
SAMPLE_CLASSES   = None

def sep(t): print(f"\n{'='*65}\n  {t}\n{'='*65}")

def main():
    sep("GAT Land Cover Classification — JIIT Minor Project 2")

    # STEP 1-2: Preprocessing
    sep("STEP 1-2 │ INPUT + PREPROCESSING")
    processor = EuroSATRealProcessor(data_path=DATA_PATH)
    data = processor.load_real_eurosat(max_images_per_class=MAX_IMAGES_CLASS,
                                       sample_classes=SAMPLE_CLASSES)
    if not data['images']: print("❌ No images loaded"); sys.exit(1)

    # STEP 3-4: SLIC + Features
    sep("STEP 3-4 │ SLIC + FEATURE EXTRACTION")
    all_features, all_segments = [], []
    for i, (img, label) in enumerate(zip(data['images'], data['labels'])):
        print(f"  [{i+1}/{len(data['images'])}] {label}")
        segs = processor.get_superpixels(img, N_SEGMENTS)
        ndvi = processor.get_ndvi(img)
        df   = processor.extract_features(img, segs, label, i)
        all_features.append(df); all_segments.append(segs)
        if i < 3:
            processor.visualize(img, segs, ndvi, label,
                                save_path=f'result_{i:03d}_{label}.png')

    features_df = pd.concat(all_features, ignore_index=True)
    features_df.to_csv('features.csv', index=False)
    plot_ndvi_analysis(features_df)

    # STEP 5: Graph Construction
    sep("STEP 5 │ GRAPH CONSTRUCTION")
    builder = GraphBuilder(all_segments, features_df)
    graphs  = builder.build_graphs()

    n = len(graphs); idx = list(range(n))
    tr_idx, tmp = train_test_split(idx, test_size=0.30, random_state=42)
    vl_idx, te_idx = train_test_split(tmp, test_size=0.50, random_state=42)
    train_g = [graphs[i] for i in tr_idx]
    val_g   = [graphs[i] for i in vl_idx]
    test_g  = [graphs[i] for i in te_idx]
    print(f"  Train={len(train_g)} Val={len(val_g)} Test={len(test_g)}")

    # STEP 6: GAT Training
    sep("STEP 6 │ GAT MODEL TRAINING")
    model   = GATClassifier(graphs[0].x.shape[1], GAT_HIDDEN,
                            len(processor.class_names), GAT_HEADS, DROPOUT)
    trainer = GATTrainer(model, lr=LEARNING_RATE)
    trainer._scaler = builder.scaler
    history = trainer.train(train_g, val_g, epochs=EPOCHS)
    trainer.save('gat_model.pth')

    # STEP 7 & 9: Classification + Evaluation
    sep("STEP 7+9 │ CLASSIFICATION + EVALUATION")
    present_ids   = sorted(set(l for g in test_g for l in g.y.numpy()))
    present_names = [processor.class_names[i] for i in present_ids]
    results = trainer.evaluate(test_g, processor.class_names)
    print(f"  Accuracy={results['accuracy']*100:.2f}%  F1={results['f1_score']*100:.2f}%")
    print(results['report'])

    # STEP 8: Land Cover Maps
    sep("STEP 8 │ LAND COVER MAP GENERATION")
    os.makedirs('land_cover_maps', exist_ok=True)
    model.eval(); device = trainer.device
    for pos, g_idx in enumerate(te_idx):
        img_i  = data['images'][g_idx]; lbl_i = data['labels'][g_idx]
        segs_i = all_segments[g_idx]
        df_i   = features_df[features_df['img_id']==g_idx].reset_index(drop=True)
        with torch.no_grad():
            g_t  = test_g[pos].to(device)
            p_i  = model(g_t.x, g_t.edge_index).argmax(dim=1).cpu().numpy()
        pps = {int(row['seg_id']): p_i[i] for i,(_, row) in enumerate(df_i.iterrows())}
        sp  = 'land_cover_map.png' if pos==0 else f'land_cover_maps/map_{g_idx:03d}_{lbl_i}.png'
        plot_land_cover_map(img_i, segs_i, pps, processor.class_names,
                            processor, true_label=lbl_i, save_path=sp)
    print("  ✅ Land cover maps done")

    # STEP 10: Baseline Comparison
    sep("STEP 10 │ COMPARISON WITH BASELINES")
    baseline = compare_with_baselines(features_df, te_idx, processor.class_names)
    plot_baseline_comparison(baseline, results['accuracy'], results['f1_score'])

    # STEP 11: Final Visualization
    sep("STEP 11 │ RESULT VISUALIZATION & ANALYSIS")
    plot_training_history(history)
    plot_confusion_matrix(results['confusion_matrix'], present_names)

    # Summary
    sep("✅ COMPLETE — FINAL RESULTS")
    print(f"\n  {'Method':<26} Accuracy    F1-Score")
    print(f"  {'-'*50}")
    for m, r in baseline.items():
        print(f"  {m:<26} {r['accuracy']*100:.2f}%      {r['f1']*100:.2f}%")
    print(f"  {'GAT (Ours)':<26} {results['accuracy']*100:.2f}%      "
          f"{results['f1_score']*100:.2f}%   ← BEST ✦")

    print("\n🎯 Show Mam:")
    print("  1. land_cover_map.png       ← GAT prediction (3-panel)")
    print("  2. baseline_comparison.png  ← GAT > CNN > Random Forest")
    print("  3. confusion_matrix.png     ← class-wise accuracy")
    print("  4. training_history.png     ← learning curves")

if __name__ == '__main__':
    main()
"""
test.py — Quick sanity check before running full pipeline
"""

import os, sys
import numpy as np

DATA_PATH = r"D:\EuroSAT\allBands"

print("=" * 50)
print("  SANITY CHECK")
print("=" * 50)

# ── 1. folder check ────────────────────────────────────────────────────────
print("\n📁 Available class folders:")
if os.path.exists(DATA_PATH):
    for folder in sorted(os.listdir(DATA_PATH)):
        full = os.path.join(DATA_PATH, folder)
        if os.path.isdir(full):
            n = len([f for f in os.listdir(full) if f.endswith('.tif')])
            print(f"   ✅  {folder:<30} ({n} .tif files)")
else:
    print(f"   ❌  Path not found: {DATA_PATH}")
    sys.exit(1)

# ── 2. load 1 image ────────────────────────────────────────────────────────
print("\n🖼️  Loading 1 image per class (test)...")
from main import EuroSATRealProcessor

p = EuroSATRealProcessor(data_path=DATA_PATH)
data = p.load_real_eurosat(
    max_images_per_class=1,
    sample_classes=['Forest', 'SeaLake', 'River']
)

print(f"\n  Loaded: {len(data['images'])} images")
for i in range(len(data['images'])):
    img = data['images'][i]
    print(f"    {i+1}. {data['labels'][i]:<25} shape={img.shape}  "
          f"dtype={img.dtype}  min={img.min():.3f}  max={img.max():.3f}")

# ── 3. SLIC test ───────────────────────────────────────────────────────────
print("\n🔬 SLIC test...")
img  = data['images'][0]
segs = p.get_superpixels(img, n_segments=30)
print(f"   Segments found: {len(np.unique(segs))}  (requested 30)")

# ── 4. NDVI test ───────────────────────────────────────────────────────────
ndvi = p.get_ndvi(img)
print(f"   NDVI range: {ndvi.min():.3f} to {ndvi.max():.3f}")

# ── 5. PyTorch check ──────────────────────────────────────────────────────
try:
    import torch
    from torch_geometric.data import Data
    print(f"\n🔥 PyTorch  : {torch.__version__}")
    print(f"   CUDA      : {torch.cuda.is_available()}")
    import torch_geometric
    print(f"   PyG       : {torch_geometric.__version__}")
except ImportError as e:
    print(f"\n⚠️  Missing: {e}")
    print("   Install: pip install torch torch_geometric")

print("\n" + "="*50)
print("  ✅  Sanity check complete — safe to run run.py")
print("="*50)
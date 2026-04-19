"""
main.py — Complete GAT Pipeline for Land Cover Classification
JIIT Minor Project 2 | Dr. Alka Singhal
Team: Aditya Chaturvedi, Aditya Bajaj, Lovish Kumar
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.segmentation import slic, mark_boundaries
from skimage import io
import pandas as pd
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

# ── PyTorch + PyG imports ──────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
#  1.  DATA PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

class EuroSATRealProcessor:
    def __init__(self, data_path, use_multispectral=True, target_size=(64, 64)):
        self.data_path      = data_path
        self.use_multispectral = use_multispectral
        self.target_size    = target_size

        self.class_names = [
            'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
            'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
            'River', 'SeaLake'
        ]
        self.class_to_idx = {n: i for i, n in enumerate(self.class_names)}

        self.band_names = {
            0: 'Coastal', 1: 'Blue', 2: 'Green', 3: 'Red',
            4: 'RE1',  5: 'RE2',  6: 'RE3',  7: 'NIR',
            8: 'NIR2', 9: 'WV', 10: 'SWIR1', 11: 'SWIR2', 12: 'SWIR3'
        }

    # ── loading ────────────────────────────────────────────────────────────────
    def load_real_eurosat(self, max_images_per_class=None, sample_classes=None):
        print("Loading dataset...")
        images, labels = [], []
        classes_to_load = sample_classes if sample_classes else self.class_names

        for class_name in classes_to_load:
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.exists(class_path):
                print(f"  Warning: {class_path} not found"); continue

            files = [f for f in os.listdir(class_path)
                     if f.endswith(('.tif', '.tiff'))]
            if max_images_per_class:
                files = files[:max_images_per_class]

            for file in tqdm(files, desc=class_name):
                try:
                    img = io.imread(os.path.join(class_path, file))
                    if img.shape[0] == 13:
                        img = np.transpose(img, (1, 2, 0))
                    if img.shape[:2] != self.target_size:
                        bands = [cv2.resize(img[:, :, b], self.target_size[::-1])
                                 for b in range(img.shape[2])]
                        img = np.stack(bands, axis=2)
                    if img.dtype == np.uint16:
                        img = img.astype(np.float32) / 65535.0
                    elif img.dtype != np.float32:
                        img = img.astype(np.float32)
                    images.append(img); labels.append(class_name)
                except Exception as e:
                    print(f"  Error {file}: {e}")

        print(f"Loaded {len(images)} images")
        return {'images': images, 'labels': labels}

    # ── RGB with percentile stretch ────────────────────────────────────────────
    def create_rgb(self, img):
        def stretch(band):
            lo, hi = np.percentile(band, 2), np.percentile(band, 98)
            if hi > lo:
                band = (band - lo) / (hi - lo)
            else:
                band = band - lo
            return np.clip(band, 0, 1)
        r = stretch(img[:, :, 3].astype(np.float32))
        g = stretch(img[:, :, 2].astype(np.float32))
        b = stretch(img[:, :, 1].astype(np.float32))
        return np.stack([r, g, b], axis=2).astype(np.float32)

    # ── NDVI ──────────────────────────────────────────────────────────────────
    def get_ndvi(self, img):
        red = img[:, :, 3].astype(np.float32)
        nir = img[:, :, 7].astype(np.float32)
        denom = np.where((nir + red) == 0, 1e-10, nir + red)
        return (nir - red) / denom

    # ── SLIC superpixels ──────────────────────────────────────────────────────
    def get_superpixels(self, img, n_segments=30):
        rgb_nir = np.stack([img[:,:,3], img[:,:,2], img[:,:,1], img[:,:,7]],
                           axis=2).astype(np.float64)
        for c in range(4):
            ch = rgb_nir[:, :, c]
            lo, hi = ch.min(), ch.max()
            rgb_nir[:, :, c] = (ch - lo) / (hi - lo) if hi > lo else 0.0
        try:
            return slic(rgb_nir, n_segments=n_segments, compactness=10,
                        sigma=1, start_label=0, channel_axis=-1)
        except TypeError:
            return slic(rgb_nir, n_segments=n_segments, compactness=10,
                        sigma=1, start_label=0, multichannel=True)

    # ── per-superpixel features ────────────────────────────────────────────────
    def extract_features(self, img, segments, label, img_id):
        features = []
        ndvi = self.get_ndvi(img)
        for seg_id in np.unique(segments):
            mask = segments == seg_id
            feat = {
                'img_id': img_id, 'seg_id': seg_id,
                'class': label, 'class_id': self.class_to_idx[label],
                'size': int(np.sum(mask)),
                'ndvi_mean': float(np.mean(ndvi[mask])),
                'ndvi_std':  float(np.std(ndvi[mask])),
                'ndvi_min':  float(np.min(ndvi[mask])),
                'ndvi_max':  float(np.max(ndvi[mask])),
                'water_percent': float(np.sum(ndvi[mask] < 0) / np.sum(mask) * 100)
            }
            for b in range(img.shape[2]):
                feat[f'{self.band_names.get(b, f"B{b}")}_mean'] = \
                    float(np.mean(img[:, :, b][mask]))
            features.append(feat)
        return pd.DataFrame(features)

    # ── 6-panel visualization ─────────────────────────────────────────────────
    def visualize(self, img, segments, ndvi, label, save_path=None):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        rgb = self.create_rgb(img)

        axes[0,0].imshow(rgb)
        axes[0,0].set_title(f'RGB Composite — {label}', fontsize=11)
        axes[0,0].axis('off')

        axes[0,1].imshow(mark_boundaries(rgb, segments, color=(1,1,0)))
        axes[0,1].set_title(f'Superpixels ({len(np.unique(segments))} segments)', fontsize=11)
        axes[0,1].axis('off')

        im = axes[0,2].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        axes[0,2].set_title('NDVI  (Green=Vegetation, Red/Brown=Water/Soil)', fontsize=11)
        axes[0,2].axis('off')
        plt.colorbar(im, ax=axes[0,2], fraction=0.046, pad=0.04)

        water_mask = ndvi < 0
        water_overlay = rgb.copy()
        water_overlay[water_mask] = [0.0, 0.3, 1.0]
        axes[1,0].imshow(water_overlay)
        axes[1,0].set_title(f'Water Areas (Blue)  — {int(water_mask.sum())} px', fontsize=11)
        axes[1,0].axis('off')

        if img.shape[2] > 7:
            axes[1,1].imshow(img[:,:,7], cmap='gray')
            axes[1,1].set_title('NIR Band  (Water = Dark)', fontsize=11)
            axes[1,1].axis('off')
        else:
            axes[1,1].axis('off')

        axes[1,2].hist(ndvi.flatten(), bins=50, color='green',
                       alpha=0.7, edgecolor='darkgreen')
        axes[1,2].axvline(x=0, color='red', linestyle='--',
                          linewidth=1.5, label='Water threshold (NDVI=0)')
        axes[1,2].set_xlabel('NDVI Value'); axes[1,2].set_ylabel('Pixel Count')
        axes[1,2].set_title('NDVI Distribution', fontsize=11)
        axes[1,2].legend(fontsize=9); axes[1,2].grid(True, alpha=0.3)

        plt.suptitle(
            f'Land Cover Analysis — {label}  '
            f'(NDVI: {ndvi.mean():.3f} ± {ndvi.std():.3f})',
            fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   Saved: {save_path}")
        plt.show(); plt.close('all')

    # ── full preprocessing pipeline ───────────────────────────────────────────
    def run_pipeline(self, max_images_per_class=3, n_segments=30,
                     sample_classes=None):
        print("Starting EuroSAT preprocessing pipeline...")
        data = self.load_real_eurosat(max_images_per_class, sample_classes)
        if not data['images']:
            print("No images loaded."); return None

        all_features = []
        for i, (img, label) in enumerate(zip(data['images'], data['labels'])):
            print(f"\n[{i+1}/{len(data['images'])}] Processing: {label}")
            segments = self.get_superpixels(img, n_segments)
            ndvi     = self.get_ndvi(img)
            df       = self.extract_features(img, segments, label, i)
            all_features.append(df)
            print(f"   Superpixels: {len(df)}")
            save_path = f'result_{i:03d}_{label}.png'
            self.visualize(img, segments, ndvi, label, save_path=save_path)

        final_df = pd.concat(all_features, ignore_index=True)
        final_df.to_csv('features.csv', index=False)
        print(f"\n✅ Saved {len(final_df)} superpixel rows → features.csv")
        return final_df


# ══════════════════════════════════════════════════════════════════════════════
#  2.  GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

class GraphBuilder:
    """
    Convert superpixel features into PyTorch Geometric graph objects.
    Each superpixel = node.  Adjacent superpixels = edges.
    """

    # feature columns used as node attributes
    FEATURE_COLS = [
        'ndvi_mean', 'ndvi_std', 'ndvi_min', 'ndvi_max', 'water_percent',
        'Coastal_mean', 'Blue_mean', 'Green_mean', 'Red_mean',
        'RE1_mean', 'RE2_mean', 'RE3_mean', 'NIR_mean',
        'NIR2_mean', 'WV_mean', 'SWIR1_mean', 'SWIR2_mean', 'SWIR3_mean'
    ]

    def __init__(self, segments_list, features_df, k_neighbors=4):
        """
        segments_list : list of segment maps (one per image)
        features_df   : combined DataFrame from extract_features()
        k_neighbors   : connect each superpixel to k nearest by NDVI similarity
        """
        self.segments_list = segments_list
        self.features_df   = features_df
        self.k             = k_neighbors
        self.scaler        = StandardScaler()

    def _build_adjacency(self, segments):
        """Return list of (i,j) edge pairs from spatial adjacency."""
        edges = set()
        h, w  = segments.shape
        for dy, dx in [(0,1),(1,0),(0,-1),(-1,0)]:   # 4-connectivity
            shifted = np.roll(segments, (dy, dx), axis=(0,1))
            mask    = segments != shifted
            pairs   = set(zip(segments[mask].tolist(), shifted[mask].tolist()))
            edges  |= pairs
        return [(a, b) for a, b in edges if a != b]

    def build_graphs(self):
        """Return list of PyG Data objects, one per image."""
        # Normalize features globally
        feat_matrix = self.features_df[self.FEATURE_COLS].values.astype(np.float32)
        feat_matrix = self.scaler.fit_transform(feat_matrix)

        graphs = []
        img_ids = self.features_df['img_id'].unique()

        for img_id in img_ids:
            img_mask = self.features_df['img_id'] == img_id
            img_df   = self.features_df[img_mask].reset_index(drop=True)
            segments = self.segments_list[img_id]

            # ── node features ─────────────────────────────────────────────────
            global_indices = self.features_df[img_mask].index.tolist()
            x = torch.tensor(feat_matrix[global_indices], dtype=torch.float)

            # ── labels ────────────────────────────────────────────────────────
            y = torch.tensor(img_df['class_id'].values, dtype=torch.long)

            # ── edges from spatial adjacency ──────────────────────────────────
            raw_edges = self._build_adjacency(segments)
            seg_ids   = img_df['seg_id'].values
            seg_to_local = {s: i for i, s in enumerate(seg_ids)}

            edge_src, edge_dst = [], []
            for a, b in raw_edges:
                if a in seg_to_local and b in seg_to_local:
                    la, lb = seg_to_local[a], seg_to_local[b]
                    edge_src += [la, lb]
                    edge_dst += [lb, la]

            if not edge_src:            # fallback: fully-connected tiny graph
                n = len(img_df)
                edge_src = [i for i in range(n) for j in range(n) if i != j]
                edge_dst = [j for i in range(n) for j in range(n) if i != j]

            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            graphs.append(Data(x=x, edge_index=edge_index, y=y))

        print(f"Built {len(graphs)} graphs  |  "
              f"avg nodes: {np.mean([g.num_nodes for g in graphs]):.1f}  |  "
              f"avg edges: {np.mean([g.num_edges for g in graphs]):.1f}")
        return graphs


# ══════════════════════════════════════════════════════════════════════════════
#  3.  GAT MODEL
# ══════════════════════════════════════════════════════════════════════════════

class GATClassifier(nn.Module):
    """
    Two-layer Graph Attention Network for node-level land cover classification.

    Architecture:
        Input features  →  GATConv (8 heads)  →  ELU
                        →  Dropout
                        →  GATConv (1 head)   →  log-softmax
    """

    def __init__(self, in_channels, hidden_channels, num_classes,
                 heads=8, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        # Layer 1: multi-head attention
        self.conv1 = GATConv(in_channels, hidden_channels,
                             heads=heads, dropout=dropout, concat=True)

        # Layer 2: single-head → num_classes
        self.conv2 = GATConv(hidden_channels * heads, num_classes,
                             heads=1, dropout=dropout, concat=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# ══════════════════════════════════════════════════════════════════════════════
#  4.  TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

class GATTrainer:
    """Train and evaluate the GAT model."""

    def __init__(self, model, lr=0.005, weight_decay=5e-4, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = model.to(self.device)
        self.opt    = torch.optim.Adam(model.parameters(),
                                       lr=lr, weight_decay=weight_decay)
        self.history = {'train_loss': [], 'val_loss': [],
                        'train_acc':  [], 'val_acc': []}

    # ── single epoch ──────────────────────────────────────────────────────────
    def _run_epoch(self, graphs, train=True):
        self.model.train(train)
        total_loss, total_correct, total_nodes = 0, 0, 0

        for g in graphs:
            g = g.to(self.device)
            if train:
                self.opt.zero_grad()
            out  = self.model(g.x, g.edge_index)
            loss = F.nll_loss(out, g.y)
            if train:
                loss.backward(); self.opt.step()
            pred          = out.argmax(dim=1)
            total_loss   += loss.item() * g.num_nodes
            total_correct += (pred == g.y).sum().item()
            total_nodes  += g.num_nodes

        return total_loss / total_nodes, total_correct / total_nodes

    # ── full training loop ────────────────────────────────────────────────────
    def train(self, train_graphs, val_graphs, epochs=100):
        print(f"\nTraining on {self.device}  |  "
              f"{len(train_graphs)} train graphs, "
              f"{len(val_graphs)} val graphs")
        print("-" * 55)

        best_val_acc = 0
        best_state   = None

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self._run_epoch(train_graphs, train=True)
            with torch.no_grad():
                vl_loss, vl_acc = self._run_epoch(val_graphs, train=False)

            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(vl_loss)
            self.history['train_acc'].append(tr_acc)
            self.history['val_acc'].append(vl_acc)

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                best_state   = {k: v.clone() for k, v in
                                self.model.state_dict().items()}

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Ep {epoch:3d} | "
                      f"Train  loss={tr_loss:.4f}  acc={tr_acc*100:.1f}%  | "
                      f"Val  loss={vl_loss:.4f}  acc={vl_acc*100:.1f}%")

        # restore best weights
        if best_state:
            self.model.load_state_dict(best_state)
        print(f"\n  Best Val Accuracy: {best_val_acc*100:.2f}%")
        return self.history

    # ── evaluation ────────────────────────────────────────────────────────────
    def evaluate(self, graphs, class_names):
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for g in graphs:
                g = g.to(self.device)
                out = self.model(g.x, g.edge_index)
                pred = out.argmax(dim=1)

                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(g.y.cpu().numpy())

    # Metrics
        acc = accuracy_score(all_labels, all_preds)
        f1  = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # 🔥 FIX: handle only present classes
        unique_labels = sorted(set(all_labels) | set(all_preds))

        cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)

        report = classification_report(
        all_labels,
        all_preds,
        labels=unique_labels,
        target_names=[class_names[i] for i in unique_labels],
        zero_division=0
    )

        return {
        'accuracy': acc,
        'f1_score': f1,
        'confusion_matrix': cm,
        'report': report,
        'preds': all_preds,
        'labels': all_labels
    }

    # ── save model ────────────────────────────────────────────────────────────
    def save(self, path='gat_model.pth'):
        torch.save(self.model.state_dict(), path)
        print(f"  Model saved → {path}")

    def load(self, path='gat_model.pth'):
        self.model.load_state_dict(
            torch.load(path, map_location=self.device))
        print(f"  Model loaded ← {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  5.  VISUALIZATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_history(history, save_path='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'],   'r-', label='Val Loss',   linewidth=2)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss', fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a*100 for a in history['train_acc']],
             'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, [a*100 for a in history['val_acc']],
             'r-', label='Val Acc',   linewidth=2)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training & Validation Accuracy', fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show(); plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    # Show only classes present in data
    n = len(class_names)
    if cm.shape[0] < n:
        class_names = class_names[:cm.shape[0]]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix — GAT Land Cover Classification',
                 fontweight='bold', fontsize=13)
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show(); plt.close()
    print(f"  Saved: {save_path}")


def plot_ndvi_analysis(features_df, save_path='ndvi_analysis.png'):
    class_ndvi = features_df.groupby('class')['ndvi_mean'].mean().sort_values()
    colors = ['red'       if x < 0    else
              'orange'    if x < 0.2  else
              'yellow'    if x < 0.4  else
              'lightgreen'if x < 0.6  else
              'darkgreen' for x in class_ndvi.values]

    plt.figure(figsize=(12, 6))
    plt.barh(class_ndvi.index, class_ndvi.values, color=colors, edgecolor='gray')
    plt.xlabel('NDVI Mean'); plt.title('NDVI by Land Cover Class', fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.grid(axis='x', alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show(); plt.close()
    print(f"  Saved: {save_path}")


def plot_land_cover_map(img, segments, preds_per_segment, class_names,
                        processor, true_label=None, save_path='land_cover_map.png'):
    """
    High-quality land cover map with:
    - RGB input  |  Superpixel boundaries  |  GAT prediction map
    Each superpixel gets its predicted class color — no single-color blobs.
    """
    CLASS_COLORS = {
        'AnnualCrop':            np.array([1.00, 1.00, 0.00]),   # yellow
        'Forest':                np.array([0.00, 0.45, 0.10]),   # dark green
        'HerbaceousVegetation':  np.array([0.55, 0.85, 0.25]),   # lime green
        'Highway':               np.array([0.65, 0.65, 0.65]),   # grey
        'Industrial':            np.array([0.85, 0.35, 0.05]),   # orange-brown
        'Pasture':               np.array([0.45, 0.75, 0.45]),   # medium green
        'PermanentCrop':         np.array([1.00, 0.75, 0.10]),   # gold
        'Residential':           np.array([0.95, 0.45, 0.45]),   # coral red
        'River':                 np.array([0.20, 0.55, 1.00]),   # sky blue
        'SeaLake':               np.array([0.00, 0.10, 0.75]),   # deep blue
    }

    # ── Build per-pixel color map ──────────────────────────────────────────
    h, w = segments.shape
    color_map = np.ones((h, w, 3), dtype=np.float32) * 0.5  # grey default

    unique_segs = np.unique(segments)
    for seg_id in unique_segs:
        if seg_id in preds_per_segment:
            pred_idx   = preds_per_segment[seg_id]
            # handle both index and name
            if isinstance(pred_idx, (int, np.integer)):
                cls_name = class_names[pred_idx] if pred_idx < len(class_names) else 'Unknown'
            else:
                cls_name = str(pred_idx)
            color = CLASS_COLORS.get(cls_name, np.array([0.5, 0.5, 0.5]))
            color_map[segments == seg_id] = color

    rgb = processor.create_rgb(img)

    # ── 3-panel figure ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#1a1a2e')

    panel_titles = ['RGB Composite (Input)',
                    'Superpixel Segmentation (SLIC)',
                    f'GAT Predicted Land Cover\n(True: {true_label})' if true_label
                    else 'GAT Predicted Land Cover']

    # Panel 1 — RGB
    axes[0].imshow(rgb)
    axes[0].set_title(panel_titles[0], color='white', fontsize=11, pad=8)
    axes[0].axis('off')

    # Panel 2 — SLIC boundaries on RGB
    from skimage.segmentation import mark_boundaries
    slic_vis = mark_boundaries(rgb, segments, color=(1, 1, 0), mode='thick')
    axes[1].imshow(slic_vis)
    axes[1].set_title(panel_titles[1], color='white', fontsize=11, pad=8)
    axes[1].set_xlabel(f'{len(unique_segs)} superpixels', color='#aaaaaa', fontsize=9)
    axes[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in axes[1].spines.values(): spine.set_visible(False)

    # Panel 3 — Prediction map with superpixel boundaries
    pred_vis = mark_boundaries(color_map, segments, color=(0, 0, 0), mode='thin')
    axes[2].imshow(pred_vis)
    axes[2].set_title(panel_titles[2], color='white', fontsize=11, pad=8)
    axes[2].axis('off')

    # ── Legend ─────────────────────────────────────────────────────────────
    present_classes = set()
    for seg_id, pred_idx in preds_per_segment.items():
        if isinstance(pred_idx, (int, np.integer)) and pred_idx < len(class_names):
            present_classes.add(class_names[pred_idx])

    patches = [mpatches.Patch(facecolor=CLASS_COLORS.get(n, [0.5,0.5,0.5]),
                               edgecolor='white', linewidth=0.5, label=n)
               for n in class_names if n in present_classes or n == true_label]
    if not patches:   # fallback: show all
        patches = [mpatches.Patch(facecolor=c, edgecolor='white',
                                  linewidth=0.5, label=n)
                   for n, c in CLASS_COLORS.items()]

    legend = axes[2].legend(handles=patches, loc='lower center',
                             bbox_to_anchor=(0.5, -0.18),
                             ncol=min(5, len(patches)), fontsize=8,
                             facecolor='#2a2a3e', edgecolor='white',
                             labelcolor='white', framealpha=0.9)

    plt.suptitle('GAT Land Cover Classification — EuroSAT Multispectral',
                 color='white', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=200, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    plt.show(); plt.close()
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  6.  POST-PROCESSING  (Flowchart Step 8)
# ══════════════════════════════════════════════════════════════════════════════

def generate_land_cover_maps_all(data, all_segments, features_df,
                                  model, trainer, processor,
                                  save_dir='land_cover_maps'):
    """
    POST-PROCESSING: Generate land cover maps for every image.
    Covers flowchart step: Post-processing (Land Cover Map Generation)
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    device = trainer.device

    print(f"\n  Generating land cover maps → {save_dir}/")
    for img_id in features_df['img_id'].unique():
        img   = data['images'][img_id]
        label = data['labels'][img_id]
        segs  = all_segments[img_id]
        img_df = features_df[features_df['img_id'] == img_id].reset_index(drop=True)

        # Build graph for this image and predict
        from torch_geometric.data import Data as PyGData
        builder_tmp = GraphBuilder([segs], img_df.copy().assign(img_id=0),
                                    k_neighbors=4)
        builder_tmp.scaler = trainer._scaler if hasattr(trainer, '_scaler') \
                             else GraphBuilder([segs], img_df, 4).scaler
        graphs_tmp = builder_tmp.build_graphs()
        if not graphs_tmp: continue

        g = graphs_tmp[0].to(device)
        with torch.no_grad():
            out  = model(g.x, g.edge_index)
            preds = out.argmax(dim=1).cpu().numpy()

        preds_per_seg = {int(row['seg_id']): preds[i]
                         for i, (_, row) in enumerate(img_df.iterrows())}

        save_path = os.path.join(save_dir, f'map_{img_id:03d}_{label}.png')
        plot_land_cover_map(img, segs, preds_per_seg,
                            processor.class_names, processor,
                            true_label=label, save_path=save_path)

    print(f"  ✅ Land cover maps saved in '{save_dir}/'")


# ══════════════════════════════════════════════════════════════════════════════
#  7.  BASELINE COMPARISON  (Flowchart Step 10)
# ══════════════════════════════════════════════════════════════════════════════

def compare_with_baselines(features_df, test_img_ids, class_names):
    """
    COMPARISON WITH BASELINES: CNN-simple + Random Forest vs GAT
    Covers flowchart step: Comparison with Baselines (CNN, Random Forest)
    Returns dict of {method: accuracy}
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder

    FEAT_COLS = [
        'ndvi_mean', 'ndvi_std', 'ndvi_min', 'ndvi_max', 'water_percent',
        'Coastal_mean', 'Blue_mean', 'Green_mean', 'Red_mean',
        'RE1_mean', 'RE2_mean', 'RE3_mean', 'NIR_mean',
        'NIR2_mean', 'WV_mean', 'SWIR1_mean', 'SWIR2_mean', 'SWIR3_mean'
    ]
    available_cols = [c for c in FEAT_COLS if c in features_df.columns]

    le = LabelEncoder()
    X  = features_df[available_cols].fillna(0).values
    y  = le.fit_transform(features_df['class'].values)

    test_mask  = features_df['img_id'].isin(test_img_ids)
    train_mask = ~test_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    results = {}

    # ── Random Forest ─────────────────────────────────────────────────────
    print("  Running Random Forest baseline...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    rf_f1  = f1_score(y_test, rf.predict(X_test), average='weighted', zero_division=0)
    results['Random Forest'] = {'accuracy': rf_acc, 'f1': rf_f1}
    print(f"    RF  Accuracy={rf_acc*100:.2f}%  F1={rf_f1*100:.2f}%")

    # ── MLP (CNN-proxy) ───────────────────────────────────────────────────
    print("  Running MLP (CNN-proxy) baseline...")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200,
                        random_state=42, early_stopping=True)
    mlp.fit(X_train, y_train)
    mlp_acc = accuracy_score(y_test, mlp.predict(X_test))
    mlp_f1  = f1_score(y_test, mlp.predict(X_test), average='weighted', zero_division=0)
    results['CNN (MLP-proxy)'] = {'accuracy': mlp_acc, 'f1': mlp_f1}
    print(f"    MLP Accuracy={mlp_acc*100:.2f}%  F1={mlp_f1*100:.2f}%")

    return results


def plot_baseline_comparison(baseline_results, gat_accuracy, gat_f1,
                              save_path='baseline_comparison.png'):
    """
    RESULT VISUALIZATION: Bar chart comparing GAT vs baselines.
    Covers flowchart step: Result Visualization & Analysis
    """
    methods  = list(baseline_results.keys()) + ['GAT (Ours)']
    accs     = [v['accuracy']*100 for v in baseline_results.values()] + [gat_accuracy*100]
    f1s      = [v['f1']*100       for v in baseline_results.values()] + [gat_f1*100]
    colors   = ['#e74c3c', '#f39c12'] + ['#2ecc71']   # red, orange, green

    x   = np.arange(len(methods))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    bars1 = ax.bar(x - w/2, accs, w, label='Accuracy (%)',
                   color=colors, alpha=0.85, edgecolor='white', linewidth=0.7)
    bars2 = ax.bar(x + w/2, f1s,  w, label='F1-Score (%)',
                   color=colors, alpha=0.55, edgecolor='white', linewidth=0.7,
                   hatch='//')

    # Value labels
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom',
                color='white', fontsize=9, fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(methods, color='white', fontsize=11)
    ax.set_ylabel('Score (%)', color='white', fontsize=11)
    ax.set_title('GAT vs Baseline Methods — Land Cover Classification',
                 color='white', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(0, 115)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#555'); ax.spines['left'].set_color('#555')
    ax.spines['top'].set_visible(False);  ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#2a2a3e', edgecolor='white', labelcolor='white',
              fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.2, color='white')

    # Highlight GAT bar
    ax.annotate('✦ Best', xy=(x[-1] - w/2, accs[-1]),
                xytext=(x[-1] - w/2 + 0.3, accs[-1] + 8),
                color='#2ecc71', fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    plt.show(); plt.close()
    print(f"  Saved: {save_path}")
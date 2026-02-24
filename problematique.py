import matplotlib.pyplot as plt
import numpy as np
import skimage.color
from scipy.signal import convolve2d

import helpers.dataset as dataset
import helpers.analysis as analysis
import helpers.viz as viz

def get_dominant_hue(image_hsv):
    mask_colorful = image_hsv[:, :, 1] > 0.1
    h_values = image_hsv[:, :, 0][mask_colorful]
    return np.median(h_values) * 255 if h_values.size > 0 else 128.0

def get_roughness(image_gray):
    diff_h = np.abs(image_gray[:, 1:] - image_gray[:, :-1])
    diff_v = np.abs(image_gray[1:, :] - image_gray[:-1, :])
    return (np.mean(diff_h) + np.mean(diff_v)) / 2.0

def get_asphalt_fraction(image_hsv, n_grid=4):
    h, w, _ = image_hsv.shape
    bottom_half = image_hsv[h//2:, :, :]
    bh_h, bh_w, _ = bottom_half.shape
    
    asphalt_map = ((bottom_half[:, :, 1] < 0.3) & 
                  (bottom_half[:, :, 2] > 0.01) & 
                  (bottom_half[:, :, 2] < 1.0)).astype(float)

    y_edges = np.linspace(0, bh_h, n_grid + 1, dtype=int)
    x_edges = np.linspace(0, bh_w, n_grid + 1, dtype=int)
    counts = np.zeros((n_grid, n_grid))
    
    for gy in range(n_grid):
        for gx in range(n_grid):
            cell = asphalt_map[y_edges[gy]:y_edges[gy+1], x_edges[gx]:x_edges[gx+1]]
            counts[gy, gx] = np.mean(cell) if cell.size > 0 else 0

    best_frac = 0.0
    for gy in range(n_grid - 1):
        for gx in range(n_grid - 1):
            best_frac = max(best_frac, np.mean(counts[gy:gy+2, gx:gx+2]))
    return best_frac * 100

def get_convergence_score(image_gray):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    gx = convolve2d(image_gray, Kx, mode='valid')
    gy = convolve2d(image_gray, Ky, mode='valid')
    
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * 180 / np.pi
    
    h_mid = mag.shape[0] // 2
    thresh = np.percentile(mag, 75)
    mask = mag[h_mid:, :] > thresh
    angles_bottom = angle[h_mid:, :][mask]
    
    if angles_bottom.size == 0: return 0.0
    converging = np.sum(((angles_bottom > 20) & (angles_bottom < 80)) | 
                        ((angles_bottom > 100) & (angles_bottom < 160)))
    return (converging / angles_bottom.size) * 100

def extract_all_features(image):
    img_float = image / 255.0
    img_hsv = skimage.color.rgb2hsv(img_float)
    img_gray = np.mean(image, axis=2)
    return [get_dominant_hue(img_hsv), get_roughness(img_gray), get_asphalt_fraction(img_hsv)]

# --- UTILITY HELPERS ---

def normalize_features(data):
    """Min-Max normalization to [0, 100] scale."""
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    # Avoid division by zero
    ranges = np.where((max_vals - min_vals) == 0, 1, max_vals - min_vals)
    return ((data - min_vals) / ranges) * 100

def print_class_stats(repr_obj, names):
    print(f"\n{'Class':<10} | " + " | ".join([f"{n:<15}" for n in names]))
    print("-" * 70)
    for label in repr_obj.unique_labels:
        data = repr_obj.get_class(label)
        means = np.mean(data, axis=0)
        print(f"{label:<10} | " + " | ".join([f"{m:<15.2f}" for m in means]))

# --- MAIN FUNCTION ---

def problematique():
    images = dataset.ImageDataset("data/image_dataset/")
    
    # 1. Visualization of Raw Data
    samples = images.sample(12)
    viz.plot_images(samples, title="Dataset Samples")
    # (Optional: call viz.plot_images_histograms here if you still want them)

    # 2. Subset Selection (100 per class)
    print("Building subset...")
    subset_indices = []
    counts = {label: 0 for label in images.unique_labels}
    for i, (_, lbl) in enumerate(images):
        if counts[lbl] < 100:
            subset_indices.append(i)
            counts[lbl] += 1

    # 3. Extraction
    raw_data = []
    labels = []
    for idx in subset_indices:
        img, lbl = images[idx]
        raw_data.append(extract_all_features(img))
        labels.append(lbl)
    
    # 4. Normalization & Representation
    features_norm = normalize_features(np.array(raw_data))
    repr_3d = dataset.Representation(data=features_norm, labels=np.array(labels))

    # 5. Visualization & Statistics
    
    viz.plot_data_distribution(
        repr_3d, 
        title="3D Feature Space: Hue vs Roughness vs Asphalt",
        xlabel="Dominant Hue", ylabel="Roughness", zlabel="Asphalt %"
    )
    
    print_class_stats(repr_3d, ["Hue", "Roughness", "Asphalt"])
    plt.show()

if __name__ == "__main__":
    problematique()
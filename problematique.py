import matplotlib.pyplot as plt
import numpy as np

import helpers.dataset as dataset
import helpers.analysis as analysis
import helpers.viz as viz
import helpers.representation as rep
import skimage
from scipy.signal import convolve2d


def get_structural_regularity(image_gray):
    # Use your existing Sobel logic to get gx and gy
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gx = convolve2d(image_gray, sobel_x, mode="same")
    gy = convolve2d(image_gray, sobel_y, mode="same")

    # Calculate magnitude and orientation
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180

    # Mask to keep only strong edges (man-made structures)
    strong_edges = magnitude > np.percentile(magnitude, 80)
    relevant_orientations = orientation[strong_edges]

    # Count edges that are roughly 0, 90, or 180 degrees
    # (Allowing a 5-degree tolerance for perspective)
    cardinal_mask = (
        (relevant_orientations < 5)
        | (relevant_orientations > 175)
        | ((relevant_orientations > 85) & (relevant_orientations < 95))
    )

    if len(relevant_orientations) == 0:
        return 0
    return (np.sum(cardinal_mask) / len(relevant_orientations)) * 100


def get_mean_saturation(image_hsv):
    # Streets are gray/muted; Coasts/Forests are vivid
    return np.mean(image_hsv[:, :, 1]) * 100


def get_sky_smoothness(image_gray):
    # Coasts have empty skies (low variance at top)
    # Streets have buildings/power lines at the top (high variance)
    h = image_gray.shape[0]
    return np.var(image_gray[: h // 4, :]) * 100


def extract_all_features(image):
    img_float = image / 255.0
    img_hsv = skimage.color.rgb2hsv(img_float)
    img_gray = skimage.color.rgb2gray(img_float)

    return [
        get_structural_regularity(img_gray),
        get_mean_saturation(img_hsv),
        get_sky_smoothness(img_gray),
    ]


# --- UTILITY HELPERS ---


def normalize_features(data):
    """Min-Max normalization to [0, 100] scale."""
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    # Avoid division by zero
    ranges = np.where((max_vals - min_vals) == 0, 1, max_vals - min_vals)
    return ((data - min_vals) / ranges) * 100


def print_class_stats(repr_obj, names):
    # 1. Print Means Table
    print(f"\n{'Class':<10} | " + " | ".join([f"{n:<15}" for n in names]))
    print("-" * 75)

    for label in repr_obj.unique_labels:
        data = repr_obj.get_class(label)
        means = np.mean(data, axis=0)
        print(f"{label:<10} | " + " | ".join([f"{m:<15.2f}" for m in means]))

    # 2. Print Detailed Stats (Variance & Correlation) per Class
    print("\n" + "=" * 75)
    print("DETAILED ANALYSIS: VARIANCES & CORRELATIONS")
    print("=" * 75)

    for label in repr_obj.unique_labels:
        data = repr_obj.get_class(label)

        # Calculate stats
        variances = np.var(data, axis=0)
        # rowvar=False because variables are in columns
        correlations = np.corrcoef(data, rowvar=False)

        print(f"\n>>> Class: {label.upper()}")
        print(f"Variances: {np.array2string(variances, precision=2, separator=', ')}")
        print("Correlation Matrix:")
        # Pretty print the matrix with feature names for context
        header = " " * 12 + "  ".join([f"{n[:10]:>10}" for n in names])
        print(header)
        for i, row in enumerate(correlations):
            row_str = "  ".join([f"{val:>10.2f}" for val in row])
            print(f"{names[i][:10]:>10} | {row_str}")
        print("-" * 40)


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
        raw_data.append(rep.extract_all_features(img))
        labels.append(lbl)

    # 4. Normalization & Representation
    features_norm = rep.normalize_features(np.array(raw_data))
    repr = dataset.Representation(data=features_norm, labels=np.array(labels))

    # 5. Visualization & Statistics

    repr_3d = dataset.Representation(data=features_norm[:, :3], labels=np.array(labels))

    feature1_name = "Structural Regularity"
    feature2_name = "Mean Saturation"
    feature3_name = "Sky Smoothness"

    viz.plot_data_distribution(
        repr_3d,
        title=f"3D Feature Space: {feature1_name} vs {feature2_name} vs {feature3_name}",
        xlabel=feature1_name,
        ylabel=feature2_name,
        zlabel=feature3_name,
    )

    viz.plot_features_distribution(
        repr,
        n_bins=32,
        title="Histogrammes des moyennes des fatures",
        features_names=[feature1_name, feature2_name, feature3_name],
        xlabel="Valeur moyenne",
        ylabel="Nombre d'images",
    )

    rep.print_class_stats(repr, [feature1_name, feature2_name, feature3_name])
    plt.show()

    # TODO: Problématique: Visualisez cette représentation
    # -------------------------------------------------------------------------
    #
    # -------------------------------------------------------------------------

    # TODO: Problématique: Comparez différents classificateurs sur cette
    # représentation, comme dans le laboratoire 2 et 3.
    # -------------------------------------------------------------------------
    #
    # -------------------------------------------------------------------------


if __name__ == "__main__":
    problematique()

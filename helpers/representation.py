import numpy as np
import skimage.color
from scipy.signal import convolve2d

import helpers.viz as viz


def get_dominant_hue(image_hsv):
    mask_colorful = image_hsv[:, :, 1] > 0.2
    h_values = image_hsv[:, :, 0][mask_colorful]
    return np.median(h_values) * 255 if h_values.size > 0 else 128.0


def get_roughness(image_gray):
    diff_horizontal = np.abs(image_gray[:, 1:] - image_gray[:, :-1])
    diff_vertical = np.abs(image_gray[1:, :] - image_gray[:-1, :])
    roughness = (np.mean(diff_horizontal) + np.mean(diff_vertical)) / 2.0
    return roughness


def get_orientation_entropy(image_gray, n_bins=18):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gx = convolve2d(image_gray, sobel_x, mode="same", boundary="symm")
    gy = convolve2d(image_gray, sobel_y, mode="same", boundary="symm")

    magnitude = np.sqrt(gx**2 + gy**2) + 1e-8
    orientation = np.arctan2(gy, gx)  # [-pi, pi]

    # Keep strong edges only (structure, not noise)
    mask = magnitude > np.percentile(magnitude, 75)

    orientations = orientation[mask]

    # Map to [0, pi]
    orientations = np.abs(orientations)

    hist, _ = np.histogram(orientations, bins=n_bins, range=(0, np.pi))
    hist = hist / (np.sum(hist) + 1e-8)

    entropy = -np.sum(hist * np.log(hist + 1e-8))

    return entropy * 100


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
    return np.mean(image_hsv[:, :, 1]) * 100


def get_sky_smoothness(image_gray):
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
        get_dominant_hue(img_hsv),
        get_orientation_entropy(img_gray),
        get_roughness(img_gray),
    ]


def normalize_features(data):
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


def plot_feature_histograms(
    repr_obj, feature_names, show_histograms: bool = True, n_bins: int = 32
):
    if not show_histograms:
        return

    viz.plot_features_distribution(
        repr_obj,
        n_bins=n_bins,
        title="Histogrammes des caractéristiques (toutes les dimensions)",
        features_names=feature_names,
        xlabel="Valeur normalisée",
        ylabel="Nombre d'images",
    )

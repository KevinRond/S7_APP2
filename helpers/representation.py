import numpy as np
import skimage.color
from scipy.signal import convolve2d

import helpers.viz as viz


def get_dominant_hue(image_hsv):
    mask_colorful = image_hsv[:, :, 1] > 0.2
    h_values = image_hsv[:, :, 0][mask_colorful]
    return np.median(h_values) * 255 if h_values.size > 0 else 128.0


def get_sky_blue_ratio(image_hsv):
    """Proportion de pixels bleu-ciel dans le tiers supérieur de l'image.
    Discrimine coast (ciel bleu) vs street (bâtiments, peu de ciel)."""
    h = image_hsv.shape[0]
    top_third = image_hsv[: h // 3, :, :]  # tiers supérieur seulement

    hue = top_third[:, :, 0]  # teinte [0, 1]
    sat = top_third[:, :, 1]  # saturation [0, 1]
    val = top_third[:, :, 2]  # valeur/luminosité [0, 1]

    # Masque : bleu-cyan pâle (typique du ciel)
    blue_sky_mask = (
        (hue >= 0.50)
        & (hue <= 0.75)  # teinte bleu à cyan
        & (sat < 0.50)  # peu saturé (ciel pâle)
        & (val > 0.40)  # assez lumineux
    )

    return np.mean(blue_sky_mask) * 100  # en pourcentage


def get_asphalt_fraction_bottom(image_hsv):
    """Fraction maximale de pixels "asphalte" dans un bloc 2x2
    de la moitié inférieure de l'image, en pourcentage.

    Approche inspirée de exploration_exercice4: on découpe la moitié
    inférieure en une grille 4x4 de cellules, on calcule la fraction
    d'asphalte dans chaque cellule (faible saturation, valeur moyenne),
    puis on prend le meilleur bloc 2x2 de cellules adjacentes.
    """

    # On ne garde que la moitié inférieure de l'image
    h = image_hsv.shape[0]
    bottom_half = image_hsv[h // 2 :, :, :]

    s_low = 0.3
    v_low, v_high = 0.01, 1.0

    h_bh, w_bh, _ = bottom_half.shape
    n_rows = 4
    n_cols = 4
    y_edges = np.linspace(0, h_bh, n_rows + 1, dtype=int)
    x_edges = np.linspace(0, w_bh, n_cols + 1, dtype=int)

    asphalt_counts = np.zeros((n_rows, n_cols), dtype=float)
    total_counts = np.zeros((n_rows, n_cols), dtype=float)

    # Fraction d'asphalte par cellule de la grille 4x4
    for gy in range(n_rows):
        for gx in range(n_cols):
            y0, y1 = y_edges[gy], y_edges[gy + 1]
            x0, x1 = x_edges[gx], x_edges[gx + 1]
            if y1 <= y0 or x1 <= x0:
                continue

            cell = bottom_half[y0:y1, x0:x1, :]
            s_cell = cell[:, :, 1].ravel()
            v_cell = cell[:, :, 2].ravel()

            mask_asphalt_cell = (s_cell < s_low) & (v_cell > v_low) & (v_cell < v_high)

            asphalt_counts[gy, gx] = np.sum(mask_asphalt_cell.astype(float))
            total_counts[gy, gx] = s_cell.size

    # Meilleure fraction sur tous les blocs 2x2 de cellules adjacentes
    best_fraction = 0.0
    for gy in range(n_rows - 1):
        for gx in range(n_cols - 1):
            asphalt_block = asphalt_counts[gy : gy + 2, gx : gx + 2].sum()
            total_block = total_counts[gy : gy + 2, gx : gx + 2].sum()
            if total_block == 0:
                continue
            frac_block = asphalt_block / total_block
            if frac_block > best_fraction:
                best_fraction = frac_block

    return best_fraction * 100.0


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


def get_hue_diversity(image_hsv):
    """Diversité des teintes sur les pixels colorés.
    Street = gris uniforme → faible diversité
    Coast = eau + sable + ciel → plus diversifié
    Forest = vert dominant → diversité modérée"""
    sat = image_hsv[:, :, 1]
    hue = image_hsv[:, :, 0]

    # Seulement les pixels avec assez de couleur
    colorful = hue[sat > 0.15]

    if colorful.size < 10:
        return 0.0

    return np.std(colorful) * 100


def get_horizontal_dominance(image_gray):
    """Ratio gradients verticaux / gradients horizontaux.
    Coast : grande surface uniforme → gradients horizontaux faibles
    Street : bâtiments, bords → gradients horizontaux forts
    Forest : texture partout → ratio intermédiaire"""

    # Gradient horizontal (variation gauche-droite)
    grad_h = np.abs(image_gray[:, 1:] - image_gray[:, :-1])

    # Gradient vertical (variation haut-bas)
    grad_v = np.abs(image_gray[1:, :] - image_gray[:-1, :])

    mean_h = np.mean(grad_h)
    mean_v = np.mean(grad_v)

    # Ratio : élevé si plus de variation verticale que horizontale
    if mean_h < 1e-8:
        return 0.0
    return (mean_v / (mean_h + 1e-8)) * 100


def get_gray_pixel_ratio(image_hsv):
    """Proportion de pixels neutres/gris (faible saturation).
    Street : asphalte, béton → beaucoup de gris
    Coast/Forest : eau, sable, végétation → moins de gris"""
    sat = image_hsv[:, :, 1]
    val = image_hsv[:, :, 2]

    # Pixel gris : peu saturé ET luminosité moyenne (pas noir, pas blanc)
    gray_mask = (sat < 0.15) & (val > 0.2) & (val < 0.85)

    return np.mean(gray_mask) * 100


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
        # get_sky_blue_ratio retiré de la représentation principale
        get_hue_diversity(img_hsv),
        get_horizontal_dominance(img_gray),
        get_asphalt_fraction_bottom(img_hsv),
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

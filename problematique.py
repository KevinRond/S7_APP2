import matplotlib.pyplot as plt
import numpy as np

import helpers.dataset as dataset
import helpers.viz as viz
import helpers.representation as rep


# Toggle to show all feature histograms (one window with subplots) or none
SHOW_FEATURE_HISTOGRAMS = True

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
        all_features = rep.extract_all_features(img)
        raw_data.append([all_features[0], all_features[3], all_features[2]])
        labels.append(lbl)

    # 4. Normalization & Representation
    features_norm = rep.normalize_features(np.array(raw_data))
    repr = dataset.Representation(data=features_norm, labels=np.array(labels))

    # 5. Visualization & Statistics

    repr_3d = dataset.Representation(data=features_norm[:, :3], labels=np.array(labels))

    feature1_name = "Structural Regularity"
    # feature2_name = "Mean Saturation"
    feature2_name = "Roughness"
    feature3_name = "Sky Smoothness"
    feature4_name = "Dominant Hue"
    feature5_name = "Orientation Entropy"
    # feature_names_3D = [feature1_name, feature2_name, feature3_name]
    feature_names_all = [
        feature1_name,
        feature2_name,
        feature3_name,
        feature4_name,
        feature5_name,
    ]
    viz.plot_data_distribution(
        repr_3d,
        title=f"3D Feature Space: {feature1_name} vs {feature2_name} vs {feature3_name}",
        xlabel=feature1_name,
        ylabel=feature2_name,
        zlabel=feature3_name,
    )

    # Optionally show all feature histograms in a single window
    rep.plot_feature_histograms(
        repr_obj=repr,
        feature_names=feature_names_all,
        show_histograms=SHOW_FEATURE_HISTOGRAMS,
        n_bins=32,
    )

    rep.print_class_stats(repr, feature_names_all)
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

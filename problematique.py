import matplotlib.pyplot as plt
import numpy as np

import helpers.dataset as dataset
import helpers.analysis as analysis
import helpers.viz as viz
import helpers.representation as rep


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
    
    repr_3d = dataset.Representation(
        data=features_norm[:, :3],
        labels=np.array(labels)
    )

    feature1_name = "Structural Regularity"
    feature2_name = "Mean Saturation"
    feature3_name = "Sky Smoothness"

    viz.plot_data_distribution(
        repr_3d,
        title=f"3D Feature Space: {feature1_name} vs {feature2_name} vs {feature3_name}",
        xlabel=feature1_name,
        ylabel=feature2_name,
        zlabel=feature3_name
    )

    viz.plot_features_distribution(repr, n_bins=32,
                                  title="Histogrammes des moyennes des fatures",
                                  features_names=[feature1_name, feature2_name, feature3_name],
                                  xlabel="Valeur moyenne",
                                  ylabel="Nombre d'images")
    
    rep.print_class_stats(repr, [feature1_name, feature2_name, feature3_name])
    plt.show()

if __name__ == "__main__":
    problematique()
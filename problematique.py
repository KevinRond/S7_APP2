import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import helpers.dataset as dataset
import helpers.viz as viz
import helpers.representation as rep
import helpers.analysis as analysis
import helpers.classifier as classifier


# Toggle to show all feature histograms (one window with subplots) or none
SHOW_FEATURE_HISTOGRAMS = True


def train_val_split_pca4(raw_data, labels):
    """Train/validation split + normalization + PCA(5D) representation."""

    rng = np.random.default_rng(0)
    train_indices = []
    val_indices = []

    for lbl in np.unique(labels):
        class_idx = np.where(labels == lbl)[0]
        rng.shuffle(class_idx)
        n_class = len(class_idx)
        n_train_class = int(0.7 * n_class)
        train_indices.extend(class_idx[:n_train_class])
        val_indices.extend(class_idx[n_train_class:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    raw_train = raw_data[train_indices]
    raw_val = raw_data[val_indices]
    labels_train = labels[train_indices]
    labels_val = labels[val_indices]

    # Standardisation (z-score) basée uniquement sur l'ensemble d'entraînement
    scaler = StandardScaler().fit(raw_train)
    features_train = scaler.transform(raw_train)
    features_val = scaler.transform(raw_val)

    # PCA ajustée sur l'ensemble d'entraînement uniquement
    pca_clf = PCA(n_components=5)
    decor_train = pca_clf.fit_transform(features_train)
    decor_val = pca_clf.transform(features_val)

    repr_train = dataset.Representation(data=decor_train, labels=labels_train)
    repr_val = dataset.Representation(data=decor_val, labels=labels_val)

    return (
        raw_train,
        raw_val,
        labels_train,
        labels_val,
        features_train,
        features_val,
        decor_train,
        decor_val,
        repr_train,
        repr_val,
    )


def run_feature_importance(
    features_train, features_val, labels_train, labels_val, feature_names
):
    """Permutation importance on normalized features before PCA."""
    print("\n===== Permutation Feature Importance (k=5 KNN, no PCA) =====")

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features_train, labels_train)

    baseline = knn.score(features_val, labels_val)
    print(f"Baseline validation accuracy (5-NN, no PCA): {baseline * 100:.2f}%")

    result = permutation_importance(
        knn, features_val, labels_val, n_repeats=30, random_state=0
    )

    sorted_idx = result.importances_mean.argsort()[::-1]

    print("\nFeature importances (mean accuracy drop when feature is shuffled):")
    for i in sorted_idx:
        print(
            f"  {feature_names[i]:<30}: "
            f"{result.importances_mean[i]:+.4f} ± {result.importances_std[i]:.4f}"
        )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        range(len(feature_names)),
        result.importances_mean[sorted_idx],
        yerr=result.importances_std[sorted_idx],
        align="center",
        capsize=4,
        color=[
            "tomato" if v > 0 else "steelblue"
            for v in result.importances_mean[sorted_idx]
        ],
    )
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in sorted_idx], rotation=45, ha="right")
    ax.set_title("Permutation Feature Importance (5-NN, no PCA)")
    ax.set_ylabel("Mean accuracy decrease when shuffled")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    fig.tight_layout()


def run_bayesian_classifier(repr_train, repr_val, decor_train, labels_train):
    """Stage 8: Bayesian Gaussian classifier + 2D decision regions."""

    print("\n===== Bayesian Gaussian classifier on PCA(3) representation =====")

    bayes_classifier = classifier.BayesClassifier()
    bayes_classifier.fit(repr_train)

    # Erreur sur l'ensemble d'entraînement
    bayes_pred_train_idx = bayes_classifier.predict(repr_train.data)
    bayes_pred_train = np.array(
        [repr_train.unique_labels[p] for p in bayes_pred_train_idx]
    )
    bayes_train_error, bayes_train_err_idx = analysis.compute_error_rate(
        repr_train.labels, bayes_pred_train
    )

    # Erreur sur l'ensemble de validation
    bayes_pred_val_idx = bayes_classifier.predict(repr_val.data)
    bayes_pred_val = np.array([repr_train.unique_labels[p] for p in bayes_pred_val_idx])
    bayes_val_error, bayes_val_err_idx = analysis.compute_error_rate(
        repr_val.labels, bayes_pred_val
    )

    print(
        f"\nBayes (Gaussian) training error: {len(bayes_train_err_idx)} / "
        f"{len(repr_train.labels)} ({bayes_train_error * 100:.2f} %)"
    )
    print(
        f"Bayes (Gaussian) validation error: {len(bayes_val_err_idx)} / "
        f"{len(repr_val.labels)} ({bayes_val_error * 100:.2f} %)"
    )

    viz.show_confusion_matrix(
        repr_val.labels,
        bayes_pred_val,
        repr_val.unique_labels,
        plot=False,
    )

    # Frontières de décision numériques en 2D sur (PC1, PC2) (apprises sur train)
    repr_train_2d = dataset.Representation(data=decor_train[:, :2], labels=labels_train)
    bayes_classifier_2d = classifier.BayesClassifier()
    bayes_classifier_2d.fit(repr_train_2d)
    viz.plot_numerical_decision_regions(bayes_classifier_2d, repr_train_2d)


def run_knn_classifier(repr_train, repr_val, decor_train, labels_train):
    """Stage 9: standard K-NN classifier + 2D regions."""

    print("\n--- Recherche du k optimal pour K-NN ---")
    best_k, best_k_err = 1, 1.0
    for k in [1, 3, 5, 7, 9]:
        knn_test = classifier.KNNClassifier(n_neighbors=k)
        knn_test.fit(repr_train)
        pred = knn_test.predict(repr_val.data)
        err, _ = analysis.compute_error_rate(repr_val.labels, pred)
        print(f"k={k}: {err * 100:.2f}%")
        if err < best_k_err:
            best_k, best_k_err = k, err

    print(f"→ k optimal: {best_k} ({best_k_err * 100:.2f}%)")

    print(f"\n===== K-NN classifier (k={best_k}) on PCA(5) representation =====")
    knn_classifier = classifier.KNNClassifier(n_neighbors=best_k)
    knn_classifier.fit(repr_train)

    knn_pred_train = knn_classifier.predict(repr_train.data)
    knn_train_error, knn_train_err_idx = analysis.compute_error_rate(
        repr_train.labels, knn_pred_train
    )

    knn_pred_val = knn_classifier.predict(repr_val.data)
    knn_val_error, knn_val_err_idx = analysis.compute_error_rate(
        repr_val.labels, knn_pred_val
    )

    print(
        f"\nK-NN (1-NN) training error: {len(knn_train_err_idx)} / "
        f"{len(repr_train.labels)} ({knn_train_error * 100:.2f} %)"
    )
    print(
        f"K-NN (1-NN) validation error: {len(knn_val_err_idx)} / "
        f"{len(repr_val.labels)} ({knn_val_error * 100:.2f} %)"
    )

    viz.show_confusion_matrix(
        repr_val.labels,
        knn_pred_val,
        repr_val.unique_labels,
        plot=False,
    )

    # Frontières de décision numériques 2D (PC1, PC2) pour 1-NN (apprises sur train)
    repr_train_2d = dataset.Representation(data=decor_train[:, :2], labels=labels_train)
    knn_classifier_2d = classifier.KNNClassifier(n_neighbors=1)
    knn_classifier_2d.fit(repr_train_2d)
    viz.plot_numerical_decision_regions(knn_classifier_2d, repr_train_2d)


def run_knn_kmeans_classifier(repr_train, repr_val, decor_train, labels_train):
    """Stage 10: K-NN with k-means representatives + 2D regions."""

    print("\n--- Recherche du n_representatives optimal pour K-NN k-moyennes ---")
    best_rep, best_rep_err = 1, 1.0
    for n_rep in [1, 2, 3, 5, 7, 9]:
        knn_test = classifier.KNNClassifier(
            n_neighbors=1, use_kmeans=True, n_representatives=n_rep
        )
        knn_test.fit(repr_train)
        pred = knn_test.predict(repr_val.data)
        err, _ = analysis.compute_error_rate(repr_val.labels, pred)
        print(f"n_representatives={n_rep}: {err * 100:.2f}%")
        if err < best_rep_err:
            best_rep, best_rep_err = n_rep, err

    print(f"→ n_representatives optimal: {best_rep} ({best_rep_err * 100:.2f}%)")

    print(f"\n===== K-NN k-moyennes ({best_rep} reps) on PCA(5) representation =====")
    knn_kmeans_classifier = classifier.KNNClassifier(
        n_neighbors=1, use_kmeans=True, n_representatives=best_rep
    )
    knn_kmeans_classifier.fit(repr_train)

    knn_kmeans_pred_train = knn_kmeans_classifier.predict(repr_train.data)
    knn_kmeans_train_error, knn_kmeans_train_err_idx = analysis.compute_error_rate(
        repr_train.labels, knn_kmeans_pred_train
    )

    knn_kmeans_pred_val = knn_kmeans_classifier.predict(repr_val.data)
    knn_kmeans_val_error, knn_kmeans_val_err_idx = analysis.compute_error_rate(
        repr_val.labels, knn_kmeans_pred_val
    )

    print(
        f"\nK-NN (k-means reps) training error: {len(knn_kmeans_train_err_idx)} / "
        f"{len(repr_train.labels)} ({knn_kmeans_train_error * 100:.2f} %)"
    )
    print(
        f"K-NN (k-means reps) validation error: {len(knn_kmeans_val_err_idx)} / "
        f"{len(repr_val.labels)} ({knn_kmeans_val_error * 100:.2f} %)"
    )

    viz.show_confusion_matrix(
        repr_val.labels,
        knn_kmeans_pred_val,
        repr_val.unique_labels,
        plot=False,
    )

    # Frontières de décision numériques 2D (PC1, PC2) pour K-NN avec k-moyennes
    repr_train_2d = dataset.Representation(data=decor_train[:, :2], labels=labels_train)
    knn_kmeans_classifier_2d = classifier.KNNClassifier(
        n_neighbors=1, use_kmeans=True, n_representatives=best_rep
    )
    knn_kmeans_classifier_2d.fit(repr_train_2d)
    viz.plot_numerical_decision_regions(knn_kmeans_classifier_2d, repr_train_2d)


def run_neural_network_classifier(
    decor_train,
    decor_val,
    labels_train,
    labels_val,
    decorrelated,
    labels,
    repr_val,
):
    """Stage 11: Neural network classifier on PCA(5) + 2D regions."""

    # Utiliser l'ensemble complet (train + val) pour le RN,
    # le découpage train/validation est géré à l'intérieur de
    # NeuralNetworkClassifier.prepare_datasets.
    decor_all = np.vstack([decor_train, decor_val])
    labels_all = np.concatenate([labels_train, labels_val])
    repr_all = dataset.Representation(data=decor_all, labels=labels_all)

    print("\n===== Neural network classifier on PCA(5) representation =====")

    nn_classifier = classifier.NeuralNetworkClassifier(
        input_dim=repr_all.data.shape[1],  # 3 dimensions automatiquement
        output_dim=len(repr_all.unique_labels),
        n_hidden=2,
        n_neurons=32,
        lr=0.005,
        n_epochs=200,
        batch_size=16,
    )
    nn_classifier.fit(repr_all)

    viz.plot_metric_history(nn_classifier.history)

    nn_pred_idx = nn_classifier.predict(repr_val.data)
    nn_predictions = np.array([repr_val.unique_labels[i] for i in nn_pred_idx])

    nn_error_rate, nn_error_indices = analysis.compute_error_rate(
        repr_val.labels, nn_predictions
    )
    print(
        f"\nNeural network classification errors: {len(nn_error_indices)} / "
        f"{len(repr_val.labels)} ({nn_error_rate * 100:.2f} %)"
    )
    viz.show_confusion_matrix(
        repr_val.labels, nn_predictions, repr_all.unique_labels, plot=False
    )

    # Frontières de décision numériques 2D (PC1, PC2) pour le réseau de neurones
    nn_repr_2d = dataset.Representation(data=decorrelated[:, :2], labels=labels)
    nn_classifier_2d = classifier.NeuralNetworkClassifier(
        input_dim=2,
        output_dim=len(nn_repr_2d.unique_labels),
        n_hidden=2,
        n_neurons=32,
        lr=0.005,
        n_epochs=200,
        batch_size=16,
    )
    nn_classifier_2d.fit(nn_repr_2d)
    viz.plot_numerical_decision_regions(nn_classifier_2d, nn_repr_2d)


def problematique():
    images = dataset.ImageDataset("data/image_dataset/")

    # 1. Visualization of Raw Data
    # samples = images.sample(12)
    # viz.plot_images(samples, title="Dataset Samples")
    # (Optional: call viz.plot_images_histograms here if you still want them)

    # 2. Gather images
    subset_indices = list(range(len(images)))

    # 3. Extraction (all raw features from representation.py)
    raw_data = []
    labels = []
    for idx in subset_indices:
        img, lbl = images[idx]
        raw_data.append(rep.extract_all_features(img))
        labels.append(lbl)

    raw_data = np.array(raw_data)
    labels = np.array(labels)

    # 4. Normalization & Representation (9D)
    features_norm = rep.normalize_features(raw_data)

    repr_raw = dataset.Representation(data=features_norm, labels=labels)

    # 5. Visualization & Statistics in raw feature space

    feature_names_all = [
        "Structural Regularity",  # 0
        "Mean Saturation",  # 1
        "Sky Smoothness",  # 2
        "Dominant Hue",  # 3
        "Orientation Entropy",  # 4
        "Roughness",  # 5
        "Hue Diversity",  # 6
        "Horizontal Dominance",  # 7
        "Asphalt Fraction Bottom",  # 8
    ]

    # Top-5 features from permutation importance analysis.
    # To revert to all 9 features, comment out this line and the slicing below.
    SELECTED_FEATURES = [
        0,
        2,
        4,
        5,
        7,
    ]  # Struct. Reg., Sky Smooth., Orient. Entropy, Roughness, Horiz. Dom.

    # 3D scatter on the first three raw features
    repr_1 = dataset.Representation(data=features_norm[:, :3], labels=labels)
    viz.plot_data_distribution(
        repr_1,
        title=(
            "Raw Feature Space: Structural Regularity vs Mean Saturation vs "
            "Sky Smoothness"
        ),
        xlabel=feature_names_all[0],
        ylabel=feature_names_all[1],
        zlabel=feature_names_all[2],
    )

    repr_2 = dataset.Representation(data=features_norm[:, 3:6], labels=labels)
    viz.plot_data_distribution(
        repr_2,
        title=("Raw Feature Space: Dominant Hue vs Orientation Entropy vs Roughness"),
        xlabel=feature_names_all[3],
        ylabel=feature_names_all[4],
        zlabel=feature_names_all[5],
    )

    # Additional 2D views combining original and newly added features

    # Structural Regularity vs Mean Saturation
    repr_str_sat = dataset.Representation(data=features_norm[:, [0, 1]], labels=labels)
    viz.plot_data_distribution(
        repr_str_sat,
        title="Raw Feature Space: Structural Regularity vs Mean Saturation",
        xlabel=feature_names_all[0],
        ylabel=feature_names_all[1],
    )

    # Mean Saturation vs Hue Diversity
    repr_sat_huediv = dataset.Representation(
        data=features_norm[:, [1, 6]], labels=labels
    )
    viz.plot_data_distribution(
        repr_sat_huediv,
        title="Raw Feature Space: Mean Saturation vs Hue Diversity",
        xlabel=feature_names_all[1],
        ylabel=feature_names_all[6],
    )

    # Dominant Hue vs Hue Diversity
    repr_hue_div = dataset.Representation(data=features_norm[:, [3, 6]], labels=labels)
    viz.plot_data_distribution(
        repr_hue_div,
        title="Raw Feature Space: Dominant Hue vs Hue Diversity",
        xlabel=feature_names_all[3],
        ylabel=feature_names_all[6],
    )

    # Orientation Entropy vs Roughness
    repr_ent_rough = dataset.Representation(
        data=features_norm[:, [4, 5]], labels=labels
    )
    viz.plot_data_distribution(
        repr_ent_rough,
        title="Raw Feature Space: Orientation Entropy vs Roughness",
        xlabel=feature_names_all[4],
        ylabel=feature_names_all[5],
    )

    # Horizontal Dominance vs Asphalt Fraction Bottom
    repr_horiz_asph = dataset.Representation(
        data=features_norm[:, [7, 8]], labels=labels
    )
    viz.plot_data_distribution(
        repr_horiz_asph,
        title="Raw Feature Space: Horizontal Dominance vs Asphalt Fraction Bottom",
        xlabel=feature_names_all[7],
        ylabel=feature_names_all[8],
    )

    # Optional: all feature histograms in a single window
    rep.plot_feature_histograms(
        repr_obj=repr_raw,
        feature_names=feature_names_all,
        show_histograms=SHOW_FEATURE_HISTOGRAMS,
        n_bins=32,
    )

    rep.print_class_stats(repr_raw, feature_names_all)

    # 6. Pretraitement: PCA / Décorrélation globale (via sklearn)

    # PCA with all components so we can inspect full spectrum and eigenvectors
    pca = PCA(n_components=features_norm.shape[1])
    decorrelated = pca.fit_transform(features_norm)

    mean_vec = pca.mean_
    # Recompute covariance in original space (for display)
    cov_mat = np.cov(features_norm, rowvar=False)
    eigenvalues_sorted = pca.explained_variance_
    # sklearn.components_ shape: (n_components, n_features), rows are eigenvectors
    eigenvectors_sorted = pca.components_.T

    print("\nGlobal Gaussian model on normalized features (from PCA):")
    viz.print_gaussian_model(mean_vec, cov_mat, eigenvalues_sorted, eigenvectors_sorted)

    # Visualiser la variance expliquée (scree plot)
    explained_variance_ratio = pca.explained_variance_ratio_
    plt.figure()
    plt.plot(
        np.arange(1, len(eigenvalues_sorted) + 1),
        explained_variance_ratio,
        marker="o",
    )
    plt.title("Explained variance ratio per principal component")
    plt.xlabel("Principal component index")
    plt.ylabel("Variance ratio")
    plt.grid(True, alpha=0.3)

    # Représentations PCA pour aider à choisir le nombre de composantes

    # 3D: PC1, PC2, PC3
    repr_pca_3d = dataset.Representation(data=decorrelated[:, :3], labels=labels)
    viz.plot_data_distribution(
        repr_pca_3d,
        title="PCA space: PC1 vs PC2 vs PC3 (decorrelated)",
        xlabel="PC1",
        ylabel="PC2",
        zlabel="PC3",
    )

    # 2D PCA views to inspect separability more simply

    # 2D: PC1 vs PC2
    repr_pca_12 = dataset.Representation(data=decorrelated[:, [0, 1]], labels=labels)
    viz.plot_data_distribution(
        repr_pca_12,
        title="PCA space: PC1 vs PC2 (decorrelated)",
        xlabel="PC1",
        ylabel="PC2",
    )

    # PCA sur les 5 features sélectionnées uniquement (pour comparaison)
    features_selected_norm = features_norm[:, SELECTED_FEATURES]
    selected_names = [feature_names_all[i] for i in SELECTED_FEATURES]

    pca_sel = PCA(n_components=len(SELECTED_FEATURES))
    decorrelated_sel = pca_sel.fit_transform(features_selected_norm)

    plt.figure()
    plt.plot(
        np.arange(1, len(SELECTED_FEATURES) + 1),
        pca_sel.explained_variance_ratio_,
        marker="o",
    )
    plt.title("Explained variance ratio — selected 5 features")
    plt.xlabel("Principal component index")
    plt.ylabel("Variance ratio")
    plt.grid(True, alpha=0.3)

    repr_sel_3d = dataset.Representation(data=decorrelated_sel[:, :3], labels=labels)
    viz.plot_data_distribution(
        repr_sel_3d,
        title="PCA space (5 features): PC1 vs PC2 vs PC3",
        xlabel="PC1",
        ylabel="PC2",
        zlabel="PC3",
    )

    repr_sel_2d = dataset.Representation(
        data=decorrelated_sel[:, [0, 1]], labels=labels
    )
    viz.plot_data_distribution(
        repr_sel_2d,
        title="PCA space (5 features): PC1 vs PC2",
        xlabel="PC1",
        ylabel="PC2",
    )

    # 7–11. Train/validation split, then classifiers and neural network

    (
        raw_train,
        raw_val,
        labels_train,
        labels_val,
        features_train,
        features_val,
        decor_train,
        decor_val,
        repr_train,
        repr_val,
    ) = train_val_split_pca4(raw_data[:, SELECTED_FEATURES], labels)

    run_feature_importance(
        features_train,
        features_val,
        labels_train,
        labels_val,
        [feature_names_all[i] for i in SELECTED_FEATURES],
    )

    run_bayesian_classifier(repr_train, repr_val, decor_train, labels_train)
    run_knn_classifier(repr_train, repr_val, decor_train, labels_train)
    run_knn_kmeans_classifier(repr_train, repr_val, decor_train, labels_train)
    run_neural_network_classifier(
        decor_train,
        decor_val,
        labels_train,
        labels_val,
        decorrelated,
        labels,
        repr_val,
    )

    plt.show()


if __name__ == "__main__":
    problematique()

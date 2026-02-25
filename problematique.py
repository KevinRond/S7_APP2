import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import helpers.dataset as dataset
import helpers.viz as viz
import helpers.representation as rep
import helpers.analysis as analysis
import helpers.classifier as classifier


# Toggle to show all feature histograms (one window with subplots) or none
SHOW_FEATURE_HISTOGRAMS = True

# --- MAIN FUNCTION ---


def problematique():
    images = dataset.ImageDataset("data/image_dataset/")

    # 1. Visualization of Raw Data
    # samples = images.sample(12)
    # viz.plot_images(samples, title="Dataset Samples")
    # (Optional: call viz.plot_images_histograms here if you still want them)

    # 2. Gather images
    subset_indices = list(range(len(images)))

    # 3. Extraction (all 6 raw features from representation.py)
    raw_data = []
    labels = []
    for idx in subset_indices:
        img, lbl = images[idx]
        raw_data.append(rep.extract_all_features(img))
        labels.append(lbl)

    raw_data = np.array(raw_data)
    labels = np.array(labels)

    # 4. Normalization & Representation (6D)
    features_norm = rep.normalize_features(raw_data)
    repr_raw = dataset.Representation(data=features_norm, labels=labels)

    # 5. Visualization & Statistics in raw feature space

    feature_names_all = [
        "Structural Regularity",
        "Mean Saturation",
        "Sky Smoothness",
        "Dominant Hue",
        "Orientation Entropy",
        "Roughness",
    ]

    # 3D scatter on the first three raw features
    repr_raw_3d = dataset.Representation(data=features_norm[:, :3], labels=labels)
    viz.plot_data_distribution(
        repr_raw_3d,
        title=(
            "Raw Feature Space: Structural Regularity vs Mean Saturation vs "
            "Sky Smoothness"
        ),
        xlabel=feature_names_all[0],
        ylabel=feature_names_all[1],
        zlabel=feature_names_all[2],
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

    # 7. Train/validation split et représentation PCA(3) pour les classificateurs

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

    # Normalisation basée uniquement sur l'ensemble d'entraînement
    feat_min = raw_train.min(axis=0)
    feat_max = raw_train.max(axis=0)
    denom = feat_max - feat_min
    denom[denom == 0] = 1.0

    features_train = (raw_train - feat_min) / denom * 100.0
    features_val = (raw_val - feat_min) / denom * 100.0

    # PCA ajustée sur l'ensemble d'entraînement uniquement
    pca_clf = PCA(n_components=3)
    decor_train = pca_clf.fit_transform(features_train)
    decor_val = pca_clf.transform(features_val)

    repr_train = dataset.Representation(data=decor_train, labels=labels_train)
    repr_val = dataset.Representation(data=decor_val, labels=labels_val)

    # 8. Classificateur bayésien (modèle gaussien) avec train/validation

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
    viz.plot_classification_errors(repr_val, bayes_pred_val)

    # Frontières de décision numériques en 2D sur (PC1, PC2) (apprises sur train)
    repr_train_2d = dataset.Representation(data=decor_train[:, :2], labels=labels_train)
    bayes_classifier_2d = classifier.BayesClassifier()
    bayes_classifier_2d.fit(repr_train_2d)
    viz.plot_numerical_decision_regions(bayes_classifier_2d, repr_train_2d)

    # 9. Classificateur K-NN (k-plus proches voisins) avec train/validation

    print("\n===== K-NN classifier (1-NN) on PCA(3) representation =====")

    knn_classifier = classifier.KNNClassifier(n_neighbors=1)
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
    viz.plot_classification_errors(repr_val, knn_pred_val)

    # Frontières de décision numériques 2D (PC1, PC2) pour 1-NN (apprises sur train)
    knn_classifier_2d = classifier.KNNClassifier(n_neighbors=1)
    knn_classifier_2d.fit(repr_train_2d)
    viz.plot_numerical_decision_regions(knn_classifier_2d, repr_train_2d)

    # 10. Variante : K-NN avec représentants de classe obtenus par k-moyennes

    print("\n===== K-NN with k-means representatives on PCA(3) representation =====")

    knn_kmeans_classifier = classifier.KNNClassifier(
        n_neighbors=1, use_kmeans=True, n_representatives=1
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
    viz.plot_classification_errors(repr_val, knn_kmeans_pred_val)

    # Frontières de décision numériques 2D (PC1, PC2) pour K-NN avec k-moyennes
    knn_kmeans_classifier_2d = classifier.KNNClassifier(
        n_neighbors=1, use_kmeans=True, n_representatives=1
    )
    knn_kmeans_classifier_2d.fit(repr_train_2d)
    viz.plot_numerical_decision_regions(knn_kmeans_classifier_2d, repr_train_2d)

    # 11. Classificateur par réseau de neurones sur PCA(3)

    print("\n===== Neural network classifier on PCA(3) representation =====")

    # Utilise la représentation PCA(3) globale (decorrelated[:, :3])
    nn_repr = repr_pca_3d

    nn_classifier = classifier.NeuralNetworkClassifier(
        input_dim=nn_repr.data.shape[1],
        output_dim=len(nn_repr.unique_labels),
        n_hidden=2,
        n_neurons=32,
        lr=0.005,
        n_epochs=200,
        batch_size=16,
    )

    nn_classifier.fit(nn_repr)

    # Visualiser l'historique d'entraînement (loss / accuracy train & val)
    viz.plot_metric_history(nn_classifier.history)

    # Évaluer sur l'ensemble complet (comme dans le laboratoire 2)
    nn_pred_idx = nn_classifier.predict(nn_repr.data)
    nn_predictions = np.array([nn_repr.unique_labels[i] for i in nn_pred_idx])

    nn_error_rate, nn_error_indices = analysis.compute_error_rate(
        nn_repr.labels, nn_predictions
    )
    print(
        f"\nNeural network classification errors: {len(nn_error_indices)} / "
        f"{len(nn_repr.labels)} ({nn_error_rate * 100:.2f} %)"
    )

    viz.show_confusion_matrix(
        nn_repr.labels,
        nn_predictions,
        nn_repr.unique_labels,
        plot=False,
    )
    viz.plot_classification_errors(nn_repr, nn_predictions)

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

    plt.show()


if __name__ == "__main__":
    problematique()

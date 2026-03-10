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


SHOW_FEATURE_HISTOGRAMS = True

FEATURE_NAMES = [
    "Structural Regularity",  # 0
    "Mean Saturation",        # 1
    "Sky Smoothness",         # 2
    "Dominant Hue",           # 3
    "Orientation Entropy",    # 4
    "Roughness",              # 5
    "Horizontal Dominance",   # 6
]

SELECTED_FEATURES = [0, 2, 4, 6]  # Struct. Reg., Sky Smooth., Orient. Entropy, Horiz. Dom.


def train_val_split_pca(raw_data, labels):
    """Découpage 70/30 stratifié + normalisation z-score + PCA."""
    rng = np.random.default_rng(0)
    train_indices, val_indices = [], []

    for lbl in np.unique(labels):
        class_idx = np.where(labels == lbl)[0]
        rng.shuffle(class_idx)
        n_train = int(0.7 * len(class_idx))
        train_indices.extend(class_idx[:n_train])
        val_indices.extend(class_idx[n_train:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    raw_train = raw_data[train_indices]
    raw_val = raw_data[val_indices]
    labels_train = labels[train_indices]
    labels_val = labels[val_indices]

    # Normalisation basée uniquement sur l'ensemble d'entraînement
    scaler = StandardScaler().fit(raw_train)
    features_train = scaler.transform(raw_train)
    features_val = scaler.transform(raw_val)

    # PCA ajustée sur l'ensemble d'entraînement uniquement
    pca_clf = PCA(n_components=min(len(SELECTED_FEATURES), raw_data.shape[1]))
    decor_train = pca_clf.fit_transform(features_train)
    decor_val = pca_clf.transform(features_val)

    repr_train = dataset.Representation(data=decor_train, labels=labels_train)
    repr_val = dataset.Representation(data=decor_val, labels=labels_val)

    return (
        raw_train, raw_val,
        labels_train, labels_val,
        features_train, features_val,
        decor_train, decor_val,
        repr_train, repr_val,
    )


def run_feature_importance(features_train, features_val, labels_train, labels_val, feature_names):
    """Importance des features par permutation (5-PPV, sans PCA)."""
    print("\n===== Importance des features par permutation (k=5, sans PCA) =====")

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features_train, labels_train)

    baseline = knn.score(features_val, labels_val)
    print(f"Précision de base (validation) : {baseline * 100:.2f}%")

    result = permutation_importance(knn, features_val, labels_val, n_repeats=30, random_state=0)
    sorted_idx = result.importances_mean.argsort()[::-1]

    print("\nImportance (baisse moyenne de précision lors du mélange) :")
    for i in sorted_idx:
        print(f"  {feature_names[i]:<30}: {result.importances_mean[i]:+.4f} ± {result.importances_std[i]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        np.arange(len(feature_names)),
        result.importances_mean[sorted_idx],
        yerr=result.importances_std[sorted_idx],
        align="center",
        capsize=4,
    )
    for bar, v in zip(bars, result.importances_mean[sorted_idx]):
        bar.set_facecolor("tomato" if v > 0 else "steelblue")
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in sorted_idx], rotation=45, ha="right")
    ax.set_title("Importance des features par permutation (5-PPV, sans PCA)")
    ax.set_ylabel("Baisse moyenne de précision lors du mélange")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    fig.tight_layout()


def run_bayesian_classifier(repr_train, repr_val, decor_train, labels_train):
    """Classificateur Bayésien Gaussien + frontières de décision 2D."""
    print("\n===== Classificateur Bayésien Gaussien =====")

    unique_labels, counts = np.unique(repr_train.labels, return_counts=True)
    aprioris = counts / counts.sum()
    print(f"A priori : {dict(zip(unique_labels, aprioris.round(3)))}")

    bayes_clf = classifier.BayesClassifier(aprioris=aprioris)
    bayes_clf.fit(repr_train)

    # Erreur entraînement
    pred_train_idx = bayes_clf.predict(repr_train.data)
    pred_train = np.array([repr_train.unique_labels[p] for p in pred_train_idx])
    train_err, _ = analysis.compute_error_rate(repr_train.labels, pred_train)

    # Erreur validation
    pred_val_idx = bayes_clf.predict(repr_val.data)
    pred_val = np.array([repr_train.unique_labels[p] for p in pred_val_idx])
    val_err, _ = analysis.compute_error_rate(repr_val.labels, pred_val)

    print(f"\nErreur entraînement : {train_err * 100:.2f}%")
    print(f"Erreur validation   : {val_err * 100:.2f}%")

    viz.show_confusion_matrix(
        repr_val.labels, pred_val, repr_val.unique_labels,
        plot=True, title="Matrice de confusion - Classificateur Bayésien",
    )

    # Frontières de décision 2D (PC1 vs PC2) — à des fins de visualisation uniquement
    repr_2d = dataset.Representation(data=decor_train[:, :2], labels=labels_train)
    bayes_clf_2d = classifier.BayesClassifier()
    bayes_clf_2d.fit(repr_2d)
    viz.plot_numerical_decision_regions(bayes_clf_2d, repr_2d)


def run_knn_classifier(repr_train, repr_val, decor_train, labels_train):
    """K-PPV : recherche de la métrique et du k optimal + frontières 2D."""
    print("\n===== Classificateur K-PPV =====")

    # Recherche de la meilleure métrique
    best_metric, best_metric_err = "minkowski", 1.0
    print("\n--- Comparaison des métriques (k=3) ---")
    for metric in ["minkowski", "manhattan", "chebyshev", "sqeuclidean", "cosine", "canberra", "braycurtis"]:
        knn_test = classifier.KNNClassifier(n_neighbors=3, metric=metric)
        knn_test.fit(repr_train)
        err, _ = analysis.compute_error_rate(repr_val.labels, knn_test.predict(repr_val.data))
        print(f"  {metric:<12}: {err * 100:.2f}%")
        if err < best_metric_err:
            best_metric, best_metric_err = metric, err

    # Recherche du k optimal
    best_k, best_k_err = 1, 1.0
    print(f"\n--- Recherche du k optimal (métrique={best_metric}) ---")
    for k in [1, 3, 5, 7, 9, 11, 15, 21]:
        knn_test = classifier.KNNClassifier(n_neighbors=k, metric=best_metric)
        knn_test.fit(repr_train)
        err, _ = analysis.compute_error_rate(repr_val.labels, knn_test.predict(repr_val.data))
        print(f"  k={k}: {err * 100:.2f}%")
        if err < best_k_err:
            best_k, best_k_err = k, err

    print(f"→ k optimal: {best_k} ({best_k_err * 100:.2f}%), métrique: {best_metric}")

    # Entraînement final
    knn_clf = classifier.KNNClassifier(n_neighbors=best_k, metric=best_metric)
    knn_clf.fit(repr_train)

    train_err, _ = analysis.compute_error_rate(repr_train.labels, knn_clf.predict(repr_train.data))
    val_err, _ = analysis.compute_error_rate(repr_val.labels, knn_clf.predict(repr_val.data))

    print(f"\nErreur entraînement : {train_err * 100:.2f}%")
    print(f"Erreur validation   : {val_err * 100:.2f}%")

    viz.show_confusion_matrix(
        repr_val.labels, knn_clf.predict(repr_val.data), repr_val.unique_labels,
        plot=True, title=f"Matrice de confusion - K-PPV (k={best_k}, métrique={best_metric})",
    )

    # Frontières de décision 2D (PC1 vs PC2) — à des fins de visualisation uniquement
    repr_2d = dataset.Representation(data=decor_train[:, :2], labels=labels_train)
    knn_clf_2d = classifier.KNNClassifier(n_neighbors=1)
    knn_clf_2d.fit(repr_2d)
    viz.plot_numerical_decision_regions(knn_clf_2d, repr_2d)


def run_knn_kmeans_gridsearch(repr_train, repr_val, decor_train, labels_train):
    """K-PPV + K-moyennes : recherche en grille sur k et n_représentants + frontières 2D."""
    print("\n===== Classificateur K-PPV + K-moyennes (recherche en grille) =====")

    k_values = [1, 3, 5, 7, 9, 11, 15, 21]
    rep_values = [1, 2, 3, 5, 7, 9, 12, 15, 20]

    print("\n--- Recherche en grille : k × n_représentants ---")
    print(f"{'':>10}", end="")
    for n_rep in rep_values:
        print(f"  rep={n_rep:>2}", end="")
    print()

    best_k, best_rep, best_err = 1, 1, 1.0
    results = {}

    for k in k_values:
        print(f"k={k:<8}", end="")
        for n_rep in rep_values:
            if n_rep < k:
                print(f"  {'N/A':>6}", end="")
                results[(k, n_rep)] = None
                continue

            knn_test = classifier.KNNClassifier(
                n_neighbors=k, use_kmeans=True, n_representatives=n_rep, metric="minkowski"
            )
            knn_test.fit(repr_train)
            err, _ = analysis.compute_error_rate(repr_val.labels, knn_test.predict(repr_val.data))
            results[(k, n_rep)] = err
            print(f"  {err * 100:>5.1f}%", end="")

            if err < best_err:
                best_k, best_rep, best_err = k, n_rep, err
        print()

    print(f"\n→ Meilleure combinaison : k={best_k}, n_représentants={best_rep} ({best_err * 100:.2f}%)")

    # Entraînement final
    knn_kmeans_clf = classifier.KNNClassifier(
        n_neighbors=best_k, use_kmeans=True, n_representatives=best_rep
    )
    knn_kmeans_clf.fit(repr_train)

    train_err, _ = analysis.compute_error_rate(repr_train.labels, knn_kmeans_clf.predict(repr_train.data))
    val_err, _ = analysis.compute_error_rate(repr_val.labels, knn_kmeans_clf.predict(repr_val.data))

    print(f"\nErreur entraînement : {train_err * 100:.2f}%")
    print(f"Erreur validation   : {val_err * 100:.2f}%")

    viz.show_confusion_matrix(
        repr_val.labels, knn_kmeans_clf.predict(repr_val.data), repr_val.unique_labels,
        plot=True, title=f"Matrice de confusion - K-PPV + K-moyennes (k={best_k}, reps={best_rep})",
    )

    # Frontières de décision 2D (PC1 vs PC2) — à des fins de visualisation uniquement
    repr_2d = dataset.Representation(data=decor_train[:, :2], labels=labels_train)
    knn_kmeans_clf_2d = classifier.KNNClassifier(
        n_neighbors=best_k, use_kmeans=True, n_representatives=best_rep
    )
    knn_kmeans_clf_2d.fit(repr_2d)
    viz.plot_numerical_decision_regions(knn_kmeans_clf_2d, repr_2d)


def run_neural_network_classifier(decor_train, decor_val, labels_train, labels_val, decorrelated, labels, repr_val):
    """Réseau de neurones + frontières de décision 2D."""
    print("\n===== Classificateur Réseau de Neurones =====")

    # Le RN gère lui-même le découpage train/validation interne
    decor_all = np.vstack([decor_train, decor_val])
    labels_all = np.concatenate([labels_train, labels_val])
    repr_all = dataset.Representation(data=decor_all, labels=labels_all)

    nn_clf = classifier.NeuralNetworkClassifier(
        input_dim=repr_all.data.shape[1],
        output_dim=len(repr_all.unique_labels),
        n_hidden=2,
        n_neurons=16,
        lr=0.001,
        n_epochs=200,
        batch_size=16,
    )
    nn_clf.fit(repr_all)
    viz.plot_metric_history(nn_clf.history)

    pred_idx = nn_clf.predict(repr_val.data)
    predictions = np.array([repr_val.unique_labels[i] for i in pred_idx])
    val_err, _ = analysis.compute_error_rate(repr_val.labels, predictions)

    print(f"\nErreur validation : {val_err * 100:.2f}%")

    viz.show_confusion_matrix(
        repr_val.labels, predictions, repr_all.unique_labels,
        plot=True, title="Matrice de confusion - Réseau de neurones",
    )

    # Frontières de décision 2D (PC1 vs PC2) — modèle ré-entraîné sur 2D pour visualisation
    repr_2d = dataset.Representation(data=decorrelated[:, :2], labels=labels)
    nn_clf_2d = classifier.NeuralNetworkClassifier(
        input_dim=2,
        output_dim=len(repr_2d.unique_labels),
        n_hidden=2,
        n_neurons=16,
        lr=0.001,
        n_epochs=200,
        batch_size=32,
    )
    nn_clf_2d.fit(repr_2d)
    viz.plot_numerical_decision_regions(nn_clf_2d, repr_2d)


def problematique():
    # 1. Chargement des images
    images = dataset.ImageDataset("data/image_dataset/")

    # 2. Extraction des features
    raw_data, labels = [], []
    for idx in range(len(images)):
        img, lbl = images[idx]
        raw_data.append(rep.extract_all_features(img))
        labels.append(lbl)

    raw_data = np.array(raw_data)
    labels = np.array(labels)

    # 3. Normalisation des features brutes
    features_norm = rep.normalize_features(raw_data)
    repr_raw = dataset.Representation(data=features_norm, labels=labels)

    # -------------------------------------------------------------------------
    # 4. Visualisation de l'espace des features brutes
    # -------------------------------------------------------------------------

    # Nuages 3D — deux groupes de 3 features brutes
    viz.plot_data_distribution(
        dataset.Representation(data=features_norm[:, :3], labels=labels),
        title="Espace brut : Régularité Structurelle vs Saturation Moyenne vs Lissé Ciel",
        xlabel=FEATURE_NAMES[0], ylabel=FEATURE_NAMES[1], zlabel=FEATURE_NAMES[2],
    )
    viz.plot_data_distribution(
        dataset.Representation(data=features_norm[:, 3:6], labels=labels),
        title="Espace brut : Teinte Dominante vs Entropie d'Orientation vs Rugosité",
        xlabel=FEATURE_NAMES[3], ylabel=FEATURE_NAMES[4], zlabel=FEATURE_NAMES[5],
    )

    # Nuages 2D — toutes les paires des 4 features sélectionnées
    selected_pairs = [(0, 2), (0, 4), (0, 6), (2, 4), (2, 6), (4, 6)]
    for i, j in selected_pairs:
        viz.plot_data_distribution(
            dataset.Representation(data=features_norm[:, [i, j]], labels=labels),
            title=f"Espace brut : {FEATURE_NAMES[i]} vs {FEATURE_NAMES[j]}",
            xlabel=FEATURE_NAMES[i], ylabel=FEATURE_NAMES[j],
        )

    # Histogrammes par feature et par classe
    rep.plot_feature_histograms(
        repr_obj=repr_raw, feature_names=FEATURE_NAMES,
        show_histograms=SHOW_FEATURE_HISTOGRAMS, n_bins=32,
    )
    rep.print_class_stats(repr_raw, FEATURE_NAMES)

    # Matrice de corrélation entre les features
    corr_matrix = np.corrcoef(features_norm, rowvar=False)
    fig_corr, ax_corr = plt.subplots(figsize=(8, 7))
    im = ax_corr.imshow(corr_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    fig_corr.colorbar(im, ax=ax_corr)
    ax_corr.set_xticks(range(len(FEATURE_NAMES)))
    ax_corr.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right")
    ax_corr.set_yticks(range(len(FEATURE_NAMES)))
    ax_corr.set_yticklabels(FEATURE_NAMES)
    for i in range(len(FEATURE_NAMES)):
        for j in range(len(FEATURE_NAMES)):
            ax_corr.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax_corr.set_title("Matrice de corrélation entre les features (tous les exemples)")
    fig_corr.tight_layout()

    # -------------------------------------------------------------------------
    # 5. PCA globale (toutes les features)
    # -------------------------------------------------------------------------
    pca = PCA(n_components=features_norm.shape[1])
    decorrelated = pca.fit_transform(features_norm)

    viz.print_gaussian_model(
        pca.mean_,
        np.cov(features_norm, rowvar=False),
        pca.explained_variance_,
        pca.components_.T,
    )

    # Scree plot — toutes les features
    plt.figure()
    plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker="o")
    plt.title("Variance expliquée par composante principale (toutes les features)")
    plt.xlabel("Indice de la composante principale")
    plt.ylabel("Ratio de variance")
    plt.grid(True, alpha=0.3)

    # Nuages dans l'espace PCA global
    viz.plot_data_distribution(
        dataset.Representation(data=decorrelated[:, :3], labels=labels),
        title="Espace PCA : PC1 vs PC2 vs PC3 (décorrélé, toutes features)",
        xlabel="PC1", ylabel="PC2", zlabel="PC3",
    )
    viz.plot_data_distribution(
        dataset.Representation(data=decorrelated[:, [0, 1]], labels=labels),
        title="Espace PCA : PC1 vs PC2 (décorrélé, toutes features)",
        xlabel="PC1", ylabel="PC2",
    )

    # -------------------------------------------------------------------------
    # 6. PCA sur les features sélectionnées uniquement
    # -------------------------------------------------------------------------
    features_sel_norm = features_norm[:, SELECTED_FEATURES]
    selected_names = [FEATURE_NAMES[i] for i in SELECTED_FEATURES]

    pca_sel = PCA(n_components=len(SELECTED_FEATURES))
    decorrelated_sel = pca_sel.fit_transform(features_sel_norm)

    # Scree plot — features sélectionnées
    plt.figure()
    plt.plot(np.arange(1, len(SELECTED_FEATURES) + 1), pca_sel.explained_variance_ratio_, marker="o")
    plt.title(f"Variance expliquée — {len(SELECTED_FEATURES)} features sélectionnées")
    plt.xlabel("Indice de la composante principale")
    plt.ylabel("Ratio de variance")
    plt.grid(True, alpha=0.3)

    # Nuages dans l'espace PCA des features sélectionnées
    viz.plot_data_distribution(
        dataset.Representation(data=decorrelated_sel[:, :3], labels=labels),
        title=f"Espace PCA ({len(SELECTED_FEATURES)} features sélectionnées) : PC1 vs PC2 vs PC3",
        xlabel="PC1", ylabel="PC2", zlabel="PC3",
    )
    viz.plot_data_distribution(
        dataset.Representation(data=decorrelated_sel[:, [0, 1]], labels=labels),
        title=f"Espace PCA ({len(SELECTED_FEATURES)} features sélectionnées) : PC1 vs PC2",
        xlabel="PC1", ylabel="PC2",
    )

    # Heatmap des loadings PCA
    loadings = pca_sel.components_
    fig_load, ax_load = plt.subplots(figsize=(8, 4))
    im_load = ax_load.imshow(loadings, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    fig_load.colorbar(im_load, ax=ax_load)
    ax_load.set_xticks(range(len(SELECTED_FEATURES)))
    ax_load.set_xticklabels(selected_names, rotation=45, ha="right")
    ax_load.set_yticks(range(len(SELECTED_FEATURES)))
    ax_load.set_yticklabels([f"PC{i+1}" for i in range(len(SELECTED_FEATURES))])
    for i in range(len(SELECTED_FEATURES)):
        for j in range(len(SELECTED_FEATURES)):
            ax_load.text(j, i, f"{loadings[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax_load.set_title("Loadings PCA — contribution des features originales aux composantes")
    fig_load.tight_layout()

    # -------------------------------------------------------------------------
    # 7. Importance des features par permutation
    # -------------------------------------------------------------------------

    # Découpage 70/30 pour l'importance des features (sans PCA)
    rng_fi = np.random.default_rng(0)
    train_idx_fi, val_idx_fi = [], []
    for lbl in np.unique(labels):
        class_idx = np.where(labels == lbl)[0]
        rng_fi.shuffle(class_idx)
        n_train = int(0.7 * len(class_idx))
        train_idx_fi.extend(class_idx[:n_train])
        val_idx_fi.extend(class_idx[n_train:])
    train_idx_fi = np.array(train_idx_fi)
    val_idx_fi = np.array(val_idx_fi)

    # Toutes les features
    scaler_all = StandardScaler().fit(raw_data[train_idx_fi])
    run_feature_importance(
        scaler_all.transform(raw_data[train_idx_fi]),
        scaler_all.transform(raw_data[val_idx_fi]),
        labels[train_idx_fi], labels[val_idx_fi],
        FEATURE_NAMES,
    )

    # Features sélectionnées uniquement
    scaler_sel = StandardScaler().fit(raw_data[train_idx_fi][:, SELECTED_FEATURES])
    run_feature_importance(
        scaler_sel.transform(raw_data[train_idx_fi][:, SELECTED_FEATURES]),
        scaler_sel.transform(raw_data[val_idx_fi][:, SELECTED_FEATURES]),
        labels[train_idx_fi], labels[val_idx_fi],
        selected_names,
    )

    # -------------------------------------------------------------------------
    # 8-11. Classifieurs
    # -------------------------------------------------------------------------
    (
        raw_train, raw_val,
        labels_train, labels_val,
        features_train, features_val,
        decor_train, decor_val,
        repr_train, repr_val,
    ) = train_val_split_pca(raw_data[:, SELECTED_FEATURES], labels)

    run_bayesian_classifier(repr_train, repr_val, decor_train, labels_train)
    run_knn_classifier(repr_train, repr_val, decor_train, labels_train)
    run_knn_kmeans_gridsearch(repr_train, repr_val, decor_train, labels_train)
    run_neural_network_classifier(
        decor_train, decor_val,
        labels_train, labels_val,
        decorrelated, labels,
        repr_val,
    )

    plt.show()


if __name__ == "__main__":
    problematique()
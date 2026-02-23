# pylint: disable = missing-function-docstring, missing-module-docstring, wrong-import-position
import os
import pathlib

import matplotlib.pyplot as plt
import numpy
import sklearn
import sklearn.inspection

# Must be call before any other TensorFlow/Keras import
# Suppress oneDNN custom operations info
# Suppress INFO and WARNING messages from TF (0=all, 1=no INFO, 2=no INFO/WARN, 3=no error)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import helpers.analysis as analysis
import helpers.classifier as classifier
import helpers.dataset as dataset
import helpers.viz as viz


def main():
    data3classes = dataset.MultimodalDataset(pathlib.Path(__file__).parent / "data/data_3classes/")
    representation = dataset.Representation(data=data3classes.data, labels=data3classes.labels)

    # L2.E4.1, L2.E4.2, L2.E4.3 et L2.E4.4
    # Complétez la classe NeuralNetworkClassifier dans helpers/classifier.py
    # -------------------------------------------------------------------------
    nn_classifier = classifier.NeuralNetworkClassifier(input_dim=representation.data.shape[1],
                                                       output_dim=len(representation.unique_labels),
                                                       n_hidden=2,
                                                       n_neurons=32,
                                                       lr=0.005,
                                                       n_epochs=200,
                                                       batch_size=16)
    # -------------------------------------------------------------------------
    nn_classifier.fit(representation)

    # Save the model
    nn_classifier.save(pathlib.Path(__file__).parent / "saves/multimodal_classifier.keras")

    # Plot training metrics
    viz.plot_metric_history(nn_classifier.history)

    # Evaluate the model
    # Load the trained model
    nn_classifier.load(pathlib.Path(__file__).parent / "saves/multimodal_classifier.keras")

    # Generate a uniform distribution of samples over the minmax domain of the data
    viz.plot_numerical_decision_regions(nn_classifier, representation)

    # Predict the classes over the whole dataset
    predictions = nn_classifier.predict(representation.data)
    predictions = numpy.array([representation.unique_labels[i] for i in predictions])

    # L2.E4.5 Calculez et commentez les performance au moyen du taux de classification de données
    # et la matrice de confusion du modèle sur l'ensemble des données après l'entraînement.
    # -------------------------------------------------------------------------
    error_rate, indexes_errors = analysis.compute_error_rate(representation.labels, predictions)
    print(f"\n{len(indexes_errors)} erreurs de classification sur {len(representation.labels)} échantillons ({error_rate * 100:.2f}%).")

    viz.show_confusion_matrix(representation.labels, predictions, representation.unique_labels, plot=True)
    # -------------------------------------------------------------------------
    print("\n--- Analyse de l'importance des caractéristiques ---")
    # 1. On prépare les données d'entrées (déjà rescalées par le classifier)
    X = nn_classifier.preprocess_data(representation.data)
    
    # 2. On transforme les labels textuels ('C1', 'C2') en chiffres (0, 1, 2)
    le = sklearn.preprocessing.LabelEncoder()
    y = le.fit_transform(representation.labels)

    # 3. Calcul de l'importance
    # Note: On définit une petite fonction score car sklearn en a besoin
    def score_func(model, X_test, y_test):
        preds = model.predict(X_test)
        # On compare les prédictions (0,1,2) aux vrais labels (0,1,2)
        return numpy.mean(preds == y_test)

    result = sklearn.inspection.permutation_importance(
        nn_classifier, X, y, 
        n_repeats=10, 
        random_state=42,
        scoring=score_func # On lui donne notre fonction de calcul d'accuracy
    )

    # 4. Affichage des résultats dans la console
    for i in result.importances_mean.argsort()[::-1]:
        print(f"Dimension {i} (PC{i+1}): {result.importances_mean[i]:.4f} ± {result.importances_std[i]:.4f}")
    # --------------------------

    viz.plot_classification_errors(representation, predictions)

    plt.show()


if __name__ == "__main__":
    # Décommenter ceci pour rendre le code déterministe et pouvoir déverminer.
    # classifier.set_deterministic(seed=42)

    main()

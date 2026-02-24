# pylint: disable = missing-function-docstring, missing-module-docstring, wrong-import-position
import os
import pathlib

import matplotlib.pyplot as plt
import numpy
import skimage

# Must be call before any other TensorFlow/Keras import
# Suppress oneDNN custom operations info
# Suppress INFO and WARNING messages from TF (0=all, 1=no INFO, 2=no INFO/WARN, 3=no error)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import helpers.analysis as analysis
import helpers.dataset as dataset
import helpers.viz as viz


def explore_representations():
    """Sandbox pour explorer différentes représentations couleur (basé sur L1.E4)."""

    images = dataset.ImageDataset(pathlib.Path(__file__).parent / "data/image_dataset/")

    # Visualiser quelques images du dataset
    N = 18
    samples = images.sample(N)
    viz.plot_images(samples, title="Exemples d'images du dataset (sandbox)")

    # Conversion HSV pour explorer l'espace de couleur (image complète)
    samples_hsv = []
    for image, label in samples:
        image_hsv = skimage.color.rgb2hsv(image / 255.0)
        scaled_hsv = analysis.rescale_hsv(image_hsv, n_bins=256)
        samples_hsv.append((scaled_hsv, label))

    viz.plot_images_histograms(
        samples_hsv,
        n_bins=256,
        title="Histogrammes des intensités de pixels HSV (sandbox)",
        x_label="Valeur",
        y_label="Nombre de pixels",
        channel_names=["H", "S", "V"],
        colors=["m", "g", "y"],
    )

    # Ajout de lignes verticales pour marquer une plage de teinte (par exemple bleu)
    for ax in plt.gcf().get_axes():
        ax.axvline(140, color="k", linestyle="--", linewidth=1)
        ax.axvline(170, color="k", linestyle="--", linewidth=1)

    # -----------------------------------------------------------------
    # Même histogrammes HSV mais en ne prenant que la moitié inférieure
    # -----------------------------------------------------------------
    samples_hsv_bottom = []
    for image, label in samples:
        h_img = image.shape[0]
        bottom_half = image[h_img // 2 :, :, :]  # moitié inférieure
        image_hsv_bottom = skimage.color.rgb2hsv(bottom_half / 255.0)
        scaled_hsv_bottom = analysis.rescale_hsv(image_hsv_bottom, n_bins=256)
        samples_hsv_bottom.append((scaled_hsv_bottom, label))

    viz.plot_images_histograms(
        samples_hsv_bottom,
        n_bins=256,
        title="Histogrammes HSV (moitié inférieure des images)",
        x_label="Valeur",
        y_label="Nombre de pixels",
        channel_names=["H", "S", "V"],
        colors=["m", "g", "y"],
    )

    for ax in plt.gcf().get_axes()[-3:]:  # ne marque que les 3 derniers sous-graphiques
        ax.axvline(140, color="k", linestyle="--", linewidth=1)
        ax.axvline(170, color="k", linestyle="--", linewidth=1)

    # =====================================================================
    # Exemple de nouvelle dimension : teinte dominante par image (HSV)
    # =====================================================================
    # On calcule, pour chaque image, la teinte (H) la plus fréquente parmi
    # les pixels suffisamment saturés. Cela donne un scalaire par image.

    n_bins_h = 64
    dominant_hues = numpy.zeros((len(images), 1))
    mean_saturations = numpy.zeros((len(images), 1))
    mean_b_lab = numpy.zeros((len(images), 1))

    # Les mêmes quantités mais calculées en enlevant le tiers supérieur
    dominant_hues_wo_top = numpy.zeros((len(images), 1))
    mean_saturations_wo_top = numpy.zeros((len(images), 1))
    mean_b_lab_wo_top = numpy.zeros((len(images), 1))

    # Réservé si on veut revenir à des stats LAB sur la moitié inférieure
    asphalt_fraction_bottom = numpy.zeros((len(images), 1))
    asphalt_fraction_bottom_vertical = numpy.zeros((len(images), 1))

    for i, (image, _) in enumerate(images):
        # Image complète normalisée
        rgb_full = image / 255.0
        hsv = skimage.color.rgb2hsv(rgb_full)
        h = hsv[:, :, 0].ravel()  # H dans [0, 1]
        s = hsv[:, :, 1].ravel()  # S dans [0, 1]

        # On ignore les pixels peu saturés (proches du gris) pour la teinte dominante
        saturation_threshold = 0.2
        mask = s > saturation_threshold
        if numpy.any(mask):
            h_use = h[mask]
        else:
            h_use = h

        counts, bin_edges = numpy.histogram(h_use, bins=n_bins_h, range=(0.0, 1.0))
        idx_max = numpy.argmax(counts)
        # Centre du bin dominant
        dominant_hues[i, 0] = (bin_edges[idx_max] + bin_edges[idx_max + 1]) / 2.0

        # Saturation moyenne (on garde tous les pixels ici)
        mean_saturations[i, 0] = numpy.mean(s)

        # Moyenne du canal b* en LAB pour cette image (complète)
        lab = skimage.color.rgb2lab(rgb_full)
        b_channel = lab[:, :, 2]
        mean_b_lab[i, 0] = numpy.mean(b_channel)

        # ===============================
        # Même chose sans le tiers supérieur
        # ===============================
        h_full = rgb_full.shape[0]
        start_row = h_full // 3  # on enlève le tiers supérieur
        rgb_no_top = rgb_full[start_row:, :, :]

        hsv_no_top = skimage.color.rgb2hsv(rgb_no_top)
        h_nt = hsv_no_top[:, :, 0].ravel()
        s_nt = hsv_no_top[:, :, 1].ravel()

        mask_nt = s_nt > saturation_threshold
        if numpy.any(mask_nt):
            h_use_nt = h_nt[mask_nt]
        else:
            h_use_nt = h_nt

        counts_nt, bin_edges_nt = numpy.histogram(
            h_use_nt, bins=n_bins_h, range=(0.0, 1.0)
        )
        idx_max_nt = numpy.argmax(counts_nt)
        dominant_hues_wo_top[i, 0] = (
            bin_edges_nt[idx_max_nt] + bin_edges_nt[idx_max_nt + 1]
        ) / 2.0

        mean_saturations_wo_top[i, 0] = numpy.mean(s_nt)

        lab_no_top = skimage.color.rgb2lab(rgb_no_top)
        b_channel_no_top = lab_no_top[:, :, 2]
        mean_b_lab_wo_top[i, 0] = numpy.mean(b_channel_no_top)

        # ------------------------------
        # Moitié inférieure uniquement
        # ------------------------------
        h_img = image.shape[0]
        bottom_half = rgb_full[h_img // 2 :, :, :]
        hsv_bottom = skimage.color.rgb2hsv(bottom_half)

        # mask_b = s_b > saturation_threshold
        # if numpy.any(mask_b):
        #     h_use_b = h_b[mask_b]
        # else:
        #     h_use_b = h_b

        # counts_b, bin_edges_b = numpy.histogram(
        #     h_use_b, bins=n_bins_h, range=(0.0, 1.0)
        # )
        # idx_max_b = numpy.argmax(counts_b)
        # dominant_hues_bottom[i, 0] = (
        #     bin_edges_b[idx_max_b] + bin_edges_b[idx_max_b + 1]
        # ) / 2.0

        # total_counts_b = numpy.sum(counts_b)
        # if total_counts_b > 0:
        #     peak_fraction_b = counts_b[idx_max_b] / total_counts_b
        # else:
        #     peak_fraction_b = 0.0
        # hue_diversities_bottom[i, 0] = 1.0 - peak_fraction_b

        # mean_saturations_bottom[i, 0] = numpy.mean(s_b)

        # lab_bottom = skimage.color.rgb2lab(bottom_half)
        # b_channel_bottom = lab_bottom[:, :, 2]
        # mean_b_lab_bottom[i, 0] = numpy.mean(b_channel_bottom)

        # Fraction de pixels "gris d'asphalte" dans la moitié inférieure
        # On découpe cette moitié en une grille 4x4 de sous-régions et,
        # pour chaque image, on calcule d'abord la fraction d'asphalte
        # dans chaque cellule, puis la fraction d'asphalte dans tous les
        # blocs 2x2 de cellules adjacentes (3x3 positions possibles).
        # La valeur finale est la meilleure fraction d'un bloc 2x2.
        # Critère "asphalte" simple : faible saturation et valeur moyenne.
        s_low = 0.3
        v_low, v_high = 0.01, 1.0

        h_bh, w_bh, _ = hsv_bottom.shape
        n_rows = 4
        n_cols = 4
        y_edges = numpy.linspace(0, h_bh, n_rows + 1, dtype=int)
        x_edges = numpy.linspace(0, w_bh, n_cols + 1, dtype=int)

        # D'abord : compter l'asphalte dans chaque cellule 4x4 de la grille
        asphalt_counts = numpy.zeros((n_rows, n_cols), dtype=float)
        total_counts = numpy.zeros((n_rows, n_cols), dtype=float)

        for gy in range(n_rows):
            for gx in range(n_cols):
                y0, y1 = y_edges[gy], y_edges[gy + 1]
                x0, x1 = x_edges[gx], x_edges[gx + 1]
                if y1 <= y0 or x1 <= x0:
                    continue

                cell = hsv_bottom[y0:y1, x0:x1, :]
                s_cell = cell[:, :, 1].ravel()
                v_cell = cell[:, :, 2].ravel()

                mask_asphalt_cell = (
                    (s_cell < s_low) & (v_cell > v_low) & (v_cell < v_high)
                )

                asphalt_counts[gy, gx] = numpy.sum(mask_asphalt_cell.astype(float))
                total_counts[gy, gx] = s_cell.size

        # Ensuite : meilleures fractions pour tous les blocs 2x2 de cellules adjacentes
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

        asphalt_fraction_bottom[i, 0] = best_fraction

        # Enfin : meilleure fraction sur des bandes verticales (colonnes)
        # On additionne l'asphalte et le total sur toutes les lignes pour
        # chaque colonne de la grille 4x4, puis on garde la colonne ayant
        # la plus grande fraction d'asphalte.
        best_fraction_vertical = 0.0
        for gx in range(n_cols):
            asphalt_col = asphalt_counts[:, gx].sum()
            total_col = total_counts[:, gx].sum()
            if total_col == 0:
                continue
            frac_col = asphalt_col / total_col
            if frac_col > best_fraction_vertical:
                best_fraction_vertical = frac_col

        asphalt_fraction_bottom_vertical[i, 0] = best_fraction_vertical

    # Visualisation de la teinte dominante par classe
    representation_hue = dataset.Representation(
        data=dominant_hues, labels=images.labels
    )

    viz.plot_features_distribution(
        representation_hue,
        n_bins=255,
        title="Sandbox - Distribution de la teinte dominante (H) par classe",
        features_names=["H_dominante"],
        xlabel="Teinte dominante (0-1)",
        ylabel="Nombre d'images",
    )

    # Visualisation de la saturation moyenne par classe
    representation_sat = dataset.Representation(
        data=mean_saturations, labels=images.labels
    )

    viz.plot_features_distribution(
        representation_sat,
        n_bins=32,
        title="Sandbox - Distribution de la saturation moyenne (S) par classe",
        features_names=["S_moyenne"],
        xlabel="Saturation moyenne (0-1)",
        ylabel="Nombre d'images",
    )

    # Visualisation de la moyenne du canal b* (LAB) par classe
    representation_b_lab = dataset.Representation(data=mean_b_lab, labels=images.labels)

    viz.plot_features_distribution(
        representation_b_lab,
        n_bins=255,
        title="Sandbox - Distribution de la moyenne du canal b* (LAB) par classe",
        features_names=["b*_moyen"],
        xlabel="Moyenne du canal b*",
        ylabel="Nombre d'images",
    )

    # ===============================
    # Versions "sans le tiers supérieur"
    # ===============================

    representation_hue_wo_top = dataset.Representation(
        data=dominant_hues_wo_top, labels=images.labels
    )
    viz.plot_features_distribution(
        representation_hue_wo_top,
        n_bins=255,
        title="Sandbox - Teinte dominante (H) sans tiers supérieur",
        features_names=["H_dom_sans_haut"],
        xlabel="Teinte dominante (0-1)",
        ylabel="Nombre d'images",
    )

    representation_sat_wo_top = dataset.Representation(
        data=mean_saturations_wo_top, labels=images.labels
    )
    viz.plot_features_distribution(
        representation_sat_wo_top,
        n_bins=32,
        title="Sandbox - Saturation moyenne (S) sans tiers supérieur",
        features_names=["S_moy_sans_haut"],
        xlabel="Saturation moyenne (0-1)",
        ylabel="Nombre d'images",
    )

    representation_b_lab_wo_top = dataset.Representation(
        data=mean_b_lab_wo_top, labels=images.labels
    )
    viz.plot_features_distribution(
        representation_b_lab_wo_top,
        n_bins=255,
        title="Sandbox - Moyenne du canal b* sans tiers supérieur",
        features_names=["b*_moy_sans_haut"],
        xlabel="Moyenne du canal b*",
        ylabel="Nombre d'images",
    )

    # ===============================
    # Versions "moitié inférieure"
    # ===============================

    # rep_hue_bottom = dataset.Representation(
    #     data=dominant_hues_bottom, labels=images.labels
    # )
    # viz.plot_features_distribution(
    #     rep_hue_bottom,
    #     n_bins=255,
    #     title="Sandbox - Teinte dominante (H) - moitié inférieure",
    #     features_names=["H_dom_bas"],
    #     xlabel="Teinte dominante (0-1)",
    #     ylabel="Nombre d'images",
    # )

    # rep_sat_bottom = dataset.Representation(
    #     data=mean_saturations_bottom, labels=images.labels
    # )
    # viz.plot_features_distribution(
    #     rep_sat_bottom,
    #     n_bins=32,
    #     title="Sandbox - Saturation moyenne (S) - moitié inférieure",
    #     features_names=["S_moyenne_bas"],
    #     xlabel="Saturation moyenne (0-1)",
    #     ylabel="Nombre d'images",
    # )

    # rep_div_bottom = dataset.Representation(
    #     data=hue_diversities_bottom, labels=images.labels
    # )
    # viz.plot_features_distribution(
    #     rep_div_bottom,
    #     n_bins=255,
    #     title="Sandbox - Diversité de teinte (H) - moitié inférieure",
    #     features_names=["Diversite_H_bas"],
    #     xlabel="Diversité de teintes (0-1)",
    #     ylabel="Nombre d'images",
    # )

    # rep_b_lab_bottom = dataset.Representation(
    #     data=mean_b_lab_bottom, labels=images.labels
    # )
    # viz.plot_features_distribution(
    #     rep_b_lab_bottom,
    #     n_bins=255,
    #     title="Sandbox - Moyenne b* (LAB) - moitié inférieure",
    #     features_names=["b*_moyen_bas"],
    #     xlabel="Moyenne du canal b*",
    #     ylabel="Nombre d'images",
    # )

    # Histogramme de la fraction de pixels gris d'asphalte (moitié inférieure)
    rep_asphalt_bottom = dataset.Representation(
        data=asphalt_fraction_bottom, labels=images.labels
    )
    viz.plot_features_distribution(
        rep_asphalt_bottom,
        n_bins=255,
        title="Sandbox - Fraction de pixels gris d'asphalte (meilleure case 4x4, moitié inférieure)",
        features_names=["frac_asphalte_bas"],
        xlabel="Fraction de pixels asphalte (0-1)",
        ylabel="Nombre d'images",
    )

    # Histogramme de la fraction de pixels gris d'asphalte (bandes verticales, moitié inférieure)
    # rep_asphalt_bottom_vertical = dataset.Representation(
    #     data=asphalt_fraction_bottom_vertical, labels=images.labels
    # )
    # viz.plot_features_distribution(
    #     rep_asphalt_bottom_vertical,
    #     n_bins=255,
    #     title=(
    #         "Sandbox - Fraction de pixels gris d'asphalte"
    #         "(meilleure bande verticale, moitié inférieure)"
    #     ),
    #     features_names=["frac_asphalte_verticale_bas"],
    #     xlabel="Fraction de pixels asphalte (0-1)",
    #     ylabel="Nombre d'images",
    # )

    # Affichage des statistiques (moyenne et variance) de la fraction d'asphalte
    # dans la moitié inférieure, pour chaque classe
    print(
        "\nStatistiques de la fraction d'asphalte (blocs 2x2, moitié inférieure) par classe :"
    )
    for class_name in rep_asphalt_bottom.unique_labels:
        class_data = rep_asphalt_bottom.get_class(class_name)
        # class_data est de forme (N_class, 1)
        values = class_data[:, 0]
        mean_val = numpy.mean(values)
        var_val = numpy.var(values)
        print(f"Classe {class_name}: moyenne={mean_val:.4f}, variance={var_val:.4f}")

    # print(
    #     "\nStatistiques de la fraction d'asphalte (bandes verticales, moitié inférieure) par classe :"
    # )
    # for class_name in rep_asphalt_bottom_vertical.unique_labels:
    #     class_data = rep_asphalt_bottom_vertical.get_class(class_name)
    #     values = class_data[:, 0]
    #     mean_val = numpy.mean(values)
    #     var_val = numpy.var(values)
    #     print(f"Classe {class_name}: moyenne={mean_val:.4f}, variance={var_val:.4f}")

    # À partir d'ici, tu peux réutiliser le même schéma pour toute
    # nouvelle dimension scalaire que tu inventes :
    #  - calculer un vecteur (len(images), 1)
    #  - créer un Representation
    #  - appeler viz.plot_features_distribution pour voir la séparation.

    plt.show()


def main():
    explore_representations()


if __name__ == "__main__":
    main()

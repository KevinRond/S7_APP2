
import matplotlib.pyplot as plt
import numpy
import skimage.color
from scipy import ndimage

import helpers.dataset as dataset
import helpers.analysis as analysis
import helpers.viz as viz


def problematique():
    images = dataset.ImageDataset("data/image_dataset/")

    # TODO Problématique: Générez une représentation des images appropriée
    # pour la classification comme dans le laboratoire 1.
    # -------------------------------------------------------------------------

    # Visualiser les images a representer
    N = 12
    samples = images.sample(N)
    viz.plot_images(samples, title="Exemples d'images du dataset")

    # Histogramme de couleur RGB
    viz.plot_images_histograms(samples, n_bins=256, 
                              title="Histogrammes des intensités de pixels RGB",
                              x_label="Valeur",
                              y_label="Nombre de pixels",
                              channel_names=['Red', 'Green', 'Blue'],
                              colors=['r', 'g', 'b'])
    
    # Conversion et visualisation LAB
    samples_lab = []
    for image, label in samples:
        image_lab = skimage.color.rgb2lab(image / 255.0)
        scaled_lab = analysis.rescale_lab(image_lab, n_bins=256)
        samples_lab.append((scaled_lab, label))
    
    viz.plot_images_histograms(samples_lab, n_bins=256,
                              title="Histogrammes des intensités de pixels LAB",
                              x_label="Valeur",
                              y_label="Nombre de pixels",
                              channel_names=['L (Luminosité)', 'A (Vert-Rouge)', 'B (Bleu-Jaune)'],
                              colors=['gray', 'purple', 'orange'])
    
    # Conversion et visualisation HSV
    samples_hsv = []
    for image, label in samples:
        image_hsv = skimage.color.rgb2hsv(image / 255.0)
        scaled_hsv = analysis.rescale_hsv(image_hsv, n_bins=256)
        samples_hsv.append((scaled_hsv, label))
    
    viz.plot_images_histograms(samples_hsv, n_bins=256,
                              title="Histogrammes des intensités de pixels HSV",
                              x_label="Valeur",
                              y_label="Nombre de pixels",
                              channel_names=['H (Teinte)', 'S (Saturation)', 'V (Valeur)'],
                              colors=['red', 'green', 'blue'])
    
    # Créer un subset de 10 images par classe
    print("\nCréation d'un subset: 10 images par classe...")
    subset_indices = []
    class_counts = {}
    
    for i, (_, label) in enumerate(images):
        if label not in class_counts:
            class_counts[label] = 0
        
        if class_counts[label] < 100:
            subset_indices.append(i)
            class_counts[label] += 1
    
    print(f"Subset créé: {len(subset_indices)} images au total")
    for class_name, count in class_counts.items():
        print(f"  - {class_name}: {count} images")
    
    # Extraction de features personnalisées pour discrimination
    # Feature 1: Teinte dominante H (médiane de H pour pixels colorés, 0-255)
    # Feature 2: Rugosité/texture (variance locale - fréquence de changement entre pixels)
    # Feature 3: Fraction d'asphalte dans meilleur bloc 2x2 (grille 4x4 dans moitié inférieure)
    features = numpy.zeros((len(subset_indices), 3), dtype=numpy.float32)
    
    # Features supplémentaires pour analyse
    texture_roughness = numpy.zeros((len(subset_indices), 1), dtype=numpy.float32)
    perspective_convergence = numpy.zeros((len(subset_indices), 1), dtype=numpy.float32)
    
    subset_labels = []
    
    for idx, img_index in enumerate(subset_indices):
        image, label = images[img_index]
        subset_labels.append(label)
        
        # Conversion HSV pour extraire S et H
        image_hsv = skimage.color.rgb2hsv(image / 255.0)
        
        # Calculer la TEINTE DOMINANTE (médiane de H pour pixels colorés)
        # Filtrer les pixels avec S > 0.1 pour éviter les pixels gris où H n'est pas significatif
        mask_colorful = image_hsv[:, :, 1] > 0.1
        H_values = image_hsv[:, :, 0][mask_colorful]
        if len(H_values) > 0:
            dominant_hue = numpy.median(H_values) * 255  # Médiane de H, échelle 0-255
        else:
            dominant_hue = 128.0  # Valeur neutre si image complètement désaturée
        
        # Fraction de pixels "gris d'asphalte" dans la moitié inférieure
        # On découpe cette moitié en une grille 4x4 de sous-régions et,
        # pour chaque image, on calcule d'abord la fraction d'asphalte
        # dans chaque cellule, puis la fraction d'asphalte dans tous les
        # blocs 2x2 de cellules adjacentes (3x3 positions possibles).
        # La valeur finale est la meilleure fraction d'un bloc 2x2.
        # Critère "asphalte" simple : faible saturation et valeur moyenne.
        
        # Diviser l'image horizontalement: prendre 50% inférieur
        height = image_hsv.shape[0]
        hsv_bottom = image_hsv[height//2:, :, :]  # 50% bas de l'image
        
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

        asphalt_fraction = best_fraction * 100  # Convertir en pourcentage
        
        # Calculer la RUGOSITÉ/TEXTURE (variance locale)
        # Mesure la fréquence de changement entre pixels adjacents
        # Forest: haute rugosité (feuilles, ombres, variations)
        # Coast/Street: faible rugosité (zones plus uniformes)
        
        # Convertir en niveaux de gris pour simplifier
        image_gray = numpy.mean(image, axis=2)  # Moyenne RGB pour grayscale
        
        # Calculer les différences absolues entre pixels adjacents
        # Différences horizontales (à droite)
        diff_horizontal = numpy.abs(image_gray[:, 1:] - image_gray[:, :-1])
        # Différences verticales (en bas)
        diff_vertical = numpy.abs(image_gray[1:, :] - image_gray[:-1, :])
        
        # Moyenne des différences = mesure de rugosité
        mean_diff_horizontal = numpy.mean(diff_horizontal)
        mean_diff_vertical = numpy.mean(diff_vertical)
        roughness = (mean_diff_horizontal + mean_diff_vertical) / 2.0
        
        # Stocker aussi dans texture_roughness pour affichage séparé si besoin
        texture_roughness[idx] = roughness
        
        # Calculer la CONVERGENCE PERSPECTIVE (détection de lignes convergentes - trapèze de route)
        # Street: lignes des bords de route convergent vers point de fuite central
        # Forest/Coast: pas de convergence géométrique marquée
        
        # Utiliser les gradients Sobel pour détecter l'orientation des contours
        from scipy import ndimage
        
        # Gradients horizontaux et verticaux avec Sobel
        sobel_x = ndimage.sobel(image_gray, axis=1)  # Gradient horizontal
        sobel_y = ndimage.sobel(image_gray, axis=0)  # Gradient vertical
        
        # Calculer l'angle d'orientation de chaque gradient (en degrés)
        gradient_magnitude = numpy.sqrt(sobel_x**2 + sobel_y**2)
        gradient_angle = numpy.arctan2(sobel_y, sobel_x) * 180 / numpy.pi  # -180 à 180
        
        # Se concentrer sur la moitié inférieure (où se trouve la route)
        h = image_gray.shape[0]
        w = image_gray.shape[1]
        bottom_half_start = h // 2
        
        # Diviser le bas en gauche et droite
        grad_magnitude_bottom = gradient_magnitude[bottom_half_start:, :]
        grad_angle_bottom = gradient_angle[bottom_half_start:, :]
        
        left_half = grad_angle_bottom[:, :w//2]
        right_half = grad_angle_bottom[:, w//2:]
        mag_left = grad_magnitude_bottom[:, :w//2]
        mag_right = grad_magnitude_bottom[:, w//2:]
        
        # Filtrer seulement les gradients forts (contours significatifs)
        threshold = numpy.percentile(gradient_magnitude, 75)
        strong_left = left_half[mag_left > threshold]
        strong_right = right_half[mag_right > threshold]
        
        # Pour une route perspective:
        # - Bords gauches: angles autour de 30-60° (montent vers droite)
        # - Bords droits: angles autour de 120-150° (montent vers gauche)
        convergence_score = 0.0
        
        if len(strong_left) > 0 and len(strong_right) > 0:
            # Proportion de gradients pointant "vers le centre-haut"
            left_converging = numpy.sum((strong_left > 20) & (strong_left < 80))  # Monte vers droite
            right_converging = numpy.sum((strong_right > 100) & (strong_right < 160))  # Monte vers gauche
            
            total_strong = len(strong_left) + len(strong_right)
            convergence_score = (left_converging + right_converging) / total_strong * 100
        
        perspective_convergence[idx] = convergence_score
        
        features[idx] = [dominant_hue, roughness, asphalt_fraction]
    
    features = numpy.array(features)
    
    # Normalisation min-max pour que chaque feature utilise tout l'axe (0-100)
    print("\nNormalisation des features pour meilleure visualisation...")
    print("Plages originales:")
    for i, name in enumerate(["Teinte dominante H (0-255)", "Rugosité/texture", "Fraction asphalte meilleur bloc (0-100)"]):
        print(f"  {name}: [{numpy.min(features[:, i]):.2f}, {numpy.max(features[:, i]):.2f}]")
    
    features_normalized = numpy.zeros_like(features)
    for i in range(features.shape[1]):
        min_val = numpy.min(features[:, i])
        max_val = numpy.max(features[:, i])
        if max_val > min_val:
            features_normalized[:, i] = ((features[:, i] - min_val) / (max_val - min_val)) * 100
        else:
            features_normalized[:, i] = features[:, i]
    
    print("\nPlages après normalisation (0-100):")
    for i, name in enumerate(["Teinte dominante H (0-255)", "Rugosité/texture", "Fraction asphalte meilleur bloc (0-100)"]):
        print(f"  {name}: [{numpy.min(features_normalized[:, i]):.2f}, {numpy.max(features_normalized[:, i]):.2f}]")
    
    subset_labels = numpy.array(subset_labels)
    
    # Créer la représentation avec les features normalisées
    representation = dataset.Representation(data=features_normalized, labels=subset_labels)
    
    # Visualisation 3D comme dans laboratoire_1.py
    viz.plot_data_distribution(representation,
                               title="Distribution: Teinte H vs Rugosité vs Fraction asphalte",
                               xlabel="Teinte dominante H (0-255)",
                               ylabel="Rugosité/texture (variance locale)",
                               zlabel="Fraction asphalte (meilleur bloc 2x2)")
    
    # Afficher les statistiques par classe
    print("\n" + "="*70)
    print("STATISTIQUES PAR CLASSE (valeurs normalisées 0-100)")
    print("="*70)
    print(f"{'Classe':<10} {'Teinte H':<18} {'Rugosité':<15} {'Frac asphalte':<15}")
    print("-"*70)
    for class_name in representation.unique_labels:
        class_data = representation.get_class(class_name)
        mean_features = numpy.mean(class_data, axis=0)
        variance_features = numpy.var(class_data, axis=0)
        print(f"{class_name:<10} {mean_features[0]:<15.4f} {mean_features[1]:<15.4f} {mean_features[2]:<18.4f}")
        print(f"{'Variances:':<10} {variance_features[0]:<15.4f} {variance_features[1]:<15.4f} {variance_features[2]:<18.4f}")
    print("="*70 + "\n")
    
    # Afficher les statistiques détaillées de RUGOSITÉ (maintenant Feature 2)
    print("\n" + "="*70)
    print("DÉTAILS RUGOSITÉ/TEXTURE (Feature 2 - variance locale)")
    print("="*70)
    print("Mesure la fréquence de changement entre pixels adjacents")
    print("Forest → HAUTE rugosité | Coast/Street → BASSE rugosité")
    print("Valeurs NON normalisées (avant normalisation 0-100)")
    print("-"*70)
    print(f"{'Classe':<15} {'Rugosité moyenne':<20} {'Variance':<15}")
    print("-"*70)
    
    # Calculer les statistiques par classe
    for class_name in numpy.unique(subset_labels):
        mask = subset_labels == class_name
        class_roughness = texture_roughness[mask]
        mean_roughness = numpy.mean(class_roughness)
        var_roughness = numpy.var(class_roughness)
        print(f"{class_name:<15} {mean_roughness:<20.4f} {var_roughness:<15.4f}")
    
    print("="*70)
    print(f"Plage globale: [{numpy.min(texture_roughness):.2f}, {numpy.max(texture_roughness):.2f}]")
    print("="*70 + "\n")
    
    # Afficher les statistiques de CONVERGENCE PERSPECTIVE (détection trapèze de route)
    print("\n" + "="*70)
    print("CONVERGENCE PERSPECTIVE (détection trapèze de route)")
    print("="*70)
    print("Mesure si les contours convergent vers un point central (perspective)")
    print("Street → HAUTE convergence (bords de route) | Forest/Coast → BASSE")
    print("Score = % de gradients forts orientés vers convergence")
    print("-"*70)
    print(f"{'Classe':<15} {'Convergence moy':<20} {'Variance':<15}")
    print("-"*70)
    
    # Calculer les statistiques par classe
    for class_name in numpy.unique(subset_labels):
        mask = subset_labels == class_name
        class_convergence = perspective_convergence[mask]
        mean_convergence = numpy.mean(class_convergence)
        var_convergence = numpy.var(class_convergence)
        print(f"{class_name:<15} {mean_convergence:<20.4f} {var_convergence:<15.4f}")
    
    print("="*70)
    print(f"Plage globale: [{numpy.min(perspective_convergence):.2f}, {numpy.max(perspective_convergence):.2f}]")
    print("="*70 + "\n")
    
    
    # -------------------------------------------------------------------------

    # TODO: Problématique: Visualisez cette représentation
    # -------------------------------------------------------------------------
    # 
    # -------------------------------------------------------------------------

    # TODO: Problématique: Comparez différents classificateurs sur cette
    # représentation, comme dans le laboratoire 2 et 3.
    # -------------------------------------------------------------------------
    # 
    # -------------------------------------------------------------------------

    plt.show()


if __name__ == "__main__":
    problematique()

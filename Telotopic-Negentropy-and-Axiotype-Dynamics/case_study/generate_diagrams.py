"""
Telotopic Negentropy Graph Generator - Version Google Colab améliorée
-----------------------------------
Ce script génère des visualisations de champs vectoriels pour l'analyse de la néguentropie télotopique,
en lisant la configuration à partir de fichiers CSV.
Cette version corrige l'ordre de traitement pour suivre la séquence: Phase 1-A, Phase 1-B, Phase 2-A, etc.
Elle intègre également un algorithme avancé pour éviter les chevauchements de texte.

Adapté pour Google Colab à partir du travail original d'Alan Kleden
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration pour Google Colab - à modifier selon vos besoins
CONFIG_FILE = 'config.csv'  # Ajustez selon le nom exact de votre fichier
VECTORS_FILE = 'vectors.csv'  # Ajustez selon le nom exact de votre fichier
OUTPUT_DIR = 'output'
SPECIFIC_PHASE = None  # Mettre 'A1' pour générer uniquement Actor A, Phase 1

def optimize_label_placement(vectors, vector_scale, axis_limit):
    """
    Optimise le placement des étiquettes pour éviter les chevauchements
    Utilise un algorithme de placement avec détection de collision
    """
    # Préparation des positions initiales des étiquettes
    label_positions = []
    vector_endpoints = []
    
    # Définition des zones occupées par les vecteurs
    for vec in vectors:
        theta = np.deg2rad(vec["theta_deg"])
        dx = np.cos(theta) * vec["phi"] * vector_scale
        dy = np.sin(theta) * vec["phi"] * vector_scale
        vector_endpoints.append((dx, dy))
    
    # Fonction pour vérifier si deux rectangles se chevauchent
    def rectangles_overlap(rect1, rect2):
        # rect = (x, y, width, height)
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)
    
    # Fonction pour créer un rectangle autour d'un point avec dimensions données
    def create_rect(x, y, width=2.5, height=1.8):
        return (x - width/2, y - height/2, width, height)
    
    # Estimation des rectangles pour les flèches
    vector_rects = []
    for endpoint in vector_endpoints:
        # Crée un rectangle fin le long de la flèche
        length = np.sqrt(endpoint[0]**2 + endpoint[1]**2)
        if length > 0:
            dx, dy = endpoint
            angle = np.arctan2(dy, dx)
            # Rectangle mince centré sur le milieu du vecteur
            mid_x, mid_y = dx/2, dy/2
            width = length
            height = 0.4  # Largeur de la flèche
            # Rotation du rectangle
            sin_a, cos_a = np.sin(angle), np.cos(angle)
            w_rot = abs(width * cos_a) + abs(height * sin_a)
            h_rot = abs(width * sin_a) + abs(height * cos_a)
            vector_rects.append(create_rect(mid_x, mid_y, w_rot, h_rot))
    
    optimized_positions = []
    
    for i, vec in enumerate(vectors):
        theta = np.deg2rad(vec["theta_deg"])
        # Position initiale suggérée
        angle_offset_options = [15, 25, 35, -15, -25, -35, 45, -45]
        
        best_position = None
        min_overlaps = float('inf')
        
        # Essaie différentes distances d'éloignement
        distance_options = [2.5, 3.0, 3.5, 4.0, 4.5]
        
        for distance in distance_options:
            for angle_offset in angle_offset_options:
                test_angle = theta + np.deg2rad(angle_offset)
                label_distance = vec["phi"] * vector_scale + distance
                
                # Position testée
                test_x = np.cos(test_angle) * label_distance
                test_y = np.sin(test_angle) * label_distance
                
                # Crée un rectangle pour l'étiquette
                test_rect = create_rect(test_x, test_y)
                
                # Compte les chevauchements avec les vecteurs
                overlaps = 0
                for v_rect in vector_rects:
                    if rectangles_overlap(test_rect, v_rect):
                        overlaps += 1
                
                # Vérifie également les chevauchements avec d'autres étiquettes déjà placées
                for other_pos in optimized_positions:
                    other_rect = create_rect(other_pos[0], other_pos[1])
                    if rectangles_overlap(test_rect, other_rect):
                        overlaps += 1
                
                # Vérifie si cette position est meilleure que les précédentes
                if overlaps < min_overlaps:
                    min_overlaps = overlaps
                    best_position = (test_x, test_y, test_angle)
                
                # Si on trouve une position sans chevauchement, on s'arrête
                if overlaps == 0:
                    break
            
            # Si on a trouvé une position sans chevauchement, pas besoin d'essayer d'autres distances
            if min_overlaps == 0:
                break
        
        # Si aucune position idéale n'est trouvée, prend la meilleure disponible
        if best_position is None:
            # Position par défaut
            default_angle = theta + np.deg2rad(25 if i % 2 == 0 else -25)
            label_distance = vec["phi"] * vector_scale + 2.5
            best_position = (np.cos(default_angle) * label_distance, 
                             np.sin(default_angle) * label_distance,
                             default_angle)
        
        optimized_positions.append(best_position)
    
    return optimized_positions

def compute_alignment(group, reference_angle=0, decimals=3):
    """
    Calcule l'alignement des vecteurs par rapport à un angle de référence
    avec un arrondi contrôlé.
    """
    if not group:
        return 0

    num = 0
    denom = 0

    for v in group:
        # Calcule l'angle relatif à la référence
        theta_rel = abs(v["theta_deg"] - reference_angle)
        # Convertit en radians pour le calcul du cosinus
        theta_rad = np.deg2rad(theta_rel)
        # Ajoute à la somme pondérée
        num += v["phi"] * np.cos(theta_rad)**2
        denom += v["phi"]

    # Arrondit au nombre spécifié de décimales
    result = num / denom if denom != 0 else 0
    return round(result, decimals)

def compute_phi_T(fc_group, fi_group, decimals=2):
    """
    Calcule la force conative nette (Φ_T) avec un arrondi contrôlé.
    """
    fc_projection = sum(v["phi"] * np.cos(np.deg2rad(v["theta_deg"])) for v in fc_group)
    fi_projection = sum(v["phi"] * np.cos(np.deg2rad(v["theta_deg"]-180)) for v in fi_group)
    phi_T = fc_projection - fi_projection
    return round(phi_T, decimals)

def generate_graph(config, vectors, output_dir):
    """
    Génère un graphique basé sur la configuration et les données vectorielles fournies
    """
    # Extraction des paramètres de la configuration
    actor_id = config['actor_id']
    phase_id = config['phase_id']
    alpha = float(config['alpha'])
    
    # Création des métadonnées
    title = f"Affective Vector Field – Actor {actor_id}, Phase {phase_id}"
    subtitle = f"(Dual-Negentropy Model, α = {alpha})"
    output_filename = f"{output_dir}/Actor{actor_id}_Phase{phase_id}_VectorField_DualModel_alpha_{int(alpha*10):02d}.png"
    
    # Extraction d'autres paramètres de configuration
    graph_size = (float(config['graph_width']), float(config['graph_height']))
    axis_limit = float(config['axis_limit'])
    circle_radii = [float(r) for r in config['circle_radii'].split(',')]
    vector_scale = float(config['vector_scale'])
    angle_grid = [int(a) for a in config['angle_grid'].split(',')]
    
    # Définition des groupes de couleurs
    fc_colors = config['fc_colors'].split(',')
    fi_colors = config['fi_colors'].split(',')
    
    # Filtrage des vecteurs pour cet acteur et cette phase spécifiques
    phase_vectors = [v for v in vectors if v['actor_id'] == actor_id and str(v['phase_id']) == str(phase_id)]
    
    # Conversion au format requis
    formatted_vectors = []
    for v in phase_vectors:
        formatted_vectors.append({
            "label": v['label'],
            "principle": v['principle'],
            "phi": float(v['phi']),
            "theta_deg": float(v['theta_deg']),
            "color": v['color']
        })
    
    # Classification des vecteurs
    fc_group = [v for v in formatted_vectors if v["color"] in fc_colors]
    fi_group = [v for v in formatted_vectors if v["color"] in fi_colors]

    # Calcul des intensités relatives avec arrondi
    total_phi_fc = sum(v["phi"] for v in fc_group)
    total_phi_fi = sum(v["phi"] for v in fi_group)
    total_phi = total_phi_fc + total_phi_fi
    I_fc = round(total_phi_fc / total_phi, 3) if total_phi != 0 else 0
    I_fi = round(total_phi_fi / total_phi, 3) if total_phi != 0 else 0

    # Calcul de l'alignement avec arrondi contrôlé
    A_fc = compute_alignment(fc_group, 0, 3)
    A_fi = compute_alignment(fi_group, 180, 3)

    # Calcul de la néguentropie pondérée avec arrondi
    N_fc = round(alpha * I_fc + (1 - alpha) * A_fc, 3)
    N_fi = round(alpha * I_fi + (1 - alpha) * A_fi, 3)
    N_net = round(N_fc - N_fi, 3)

    # Calcul de la force conative nette avec arrondi
    phi_T = compute_phi_T(fc_group, fi_group, 2)

    # Création du graphique
    fig, ax = plt.subplots(figsize=graph_size)
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect('equal')
    ax.set_title(f"{title}\n{subtitle}", fontsize=14, pad=20)
    ax.axis('off')

    # Dessine des cercles concentriques et une grille angulaire
    for r in circle_radii:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, linestyle='-', linewidth=0.5, color='lightgray'))
    for angle in angle_grid:
        theta = np.deg2rad(angle)
        x_end = (axis_limit - 0.5) * np.cos(theta)
        y_end = (axis_limit - 0.5) * np.sin(theta)
        ax.plot([0, x_end], [0, y_end], linestyle='-', linewidth=0.5, color='lightgray')
        x_label = axis_limit * 0.98 * np.cos(theta)
        y_label = axis_limit * 0.98 * np.sin(theta)
        ax.text(x_label, y_label, f"{angle}°", fontsize=8, ha='center', va='center', color='gray')

    # Optimisation du placement des étiquettes
    label_positions = optimize_label_placement(formatted_vectors, vector_scale, axis_limit)
    
    # Tracé des vecteurs avec positionnement optimisé
    for i, vec in enumerate(formatted_vectors):
        theta = np.deg2rad(vec["theta_deg"])
        dx = np.cos(theta) * vec["phi"] * vector_scale
        dy = np.sin(theta) * vec["phi"] * vector_scale
        
        # Tracé du vecteur principal
        ax.arrow(0, 0, dx, dy, width=0.2, head_width=0.6, head_length=0.8,
                 length_includes_head=True, color=vec["color"], zorder=5)
        
        # Utilisation des positions optimisées
        label_x, label_y, _ = label_positions[i]
        
        # Ligne en pointillés connectant le vecteur et l'étiquette
        ax.plot([dx, label_x], [dy, label_y], linestyle='dotted', color=vec["color"], zorder=4)
        
        # Texte dans un cadre avec bordure de couleur et fond semi-transparent pour meilleure lisibilité
        text_label = f"{vec['label']}\nφ: {vec['phi']} | θ: {vec['theta_deg']}°\n{vec['principle']}"
        ax.text(label_x, label_y, text_label,
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor=vec["color"], 
                           facecolor='white', alpha=0.95), zorder=7)

    # Dessine le telos
    ax.plot(0, 0, 'ko', markersize=4, zorder=7)
    ax.arrow(0, 0, 8, 0, color='black', width=0.1, head_width=0.3, head_length=0.6,
            length_includes_head=True, linestyle='--', zorder=6)
    ax.text(8.5, 0, "TELOS (0°)", fontsize=10, fontweight='bold', va='center')

    # Affichage des métriques
    metrics_text = (
        f"Dual-Negentropy Model (α = {alpha}):\n"
        f"I_Fc: {I_fc}, A_Fc: {A_fc}, N_Fc: {N_fc}\n"
        f"I_Fi: {I_fi}, A_Fi: {A_fi}, N_Fi: {N_fi}\n"
        f"Net Telotopic Negentropy: {N_net}\n"
        f"Net Conative Force (Φ_T): {phi_T}"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black')
    ax.text(-axis_limit + 0.5, -axis_limit + 1, metrics_text, fontsize=9, verticalalignment='bottom',
            horizontalalignment='left', bbox=props, zorder=10)

    # Sauvegarde et affiche
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    # Correction: fermer la figure pour éviter la duplication
    plt.close(fig)
    
    # Afficher l'image sauvegardée dans Colab
    display_img = plt.figure(figsize=(10, 10))
    img = plt.imread(output_filename)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Actor {actor_id}, Phase {phase_id}", fontsize=12)
    
    # Afficher et fermer
    plt.tight_layout()
    plt.show()
    plt.close(display_img)
    
    print(f"Graphique sauvegardé sous: {output_filename}")
    print(f"Résultats du modèle à double néguentropie (α = {alpha}):")
    print(f"I_Fc: {I_fc}, A_Fc: {A_fc}, N_Fc: {N_fc}")
    print(f"I_Fi: {I_fi}, A_Fi: {A_fi}, N_Fi: {N_fi}")
    print(f"Net Telotopic Negentropy: {N_net}")
    print(f"Net Conative Force (Φ_T): {phi_T}")
    
    return {
        'actor_id': actor_id,
        'phase_id': phase_id,
        'I_Fc': I_fc,
        'A_Fc': A_fc,
        'N_Fc': N_fc,
        'I_Fi': I_fi,
        'A_Fi': A_fi,
        'N_Fi': N_fi,
        'N_net': N_net,
        'phi_T': phi_T
    }

def read_csv_files(config_path, vectors_path):
    """Lit et analyse les fichiers de configuration CSV"""
    
    # Lecture du fichier de configuration
    config_df = pd.read_csv(config_path)
    
    # Lecture du fichier des vecteurs
    vectors_df = pd.read_csv(vectors_path)
    
    # Conversion des DataFrames en listes de dictionnaires pour une manipulation plus facile
    configs = config_df.to_dict('records')
    vectors = vectors_df.to_dict('records')
    
    return configs, vectors

def create_summary_file(results, output_dir):
    """Crée un fichier CSV de synthèse avec tous les résultats des calculs"""
    df = pd.DataFrame(results)
    output_path = f"{output_dir}/summary_results.csv"
    df.to_csv(output_path, index=False)
    print(f"Résultats récapitulatifs enregistrés dans: {output_path}")

# Cette section principale est adaptée pour Colab et remplace le bloc if __name__ == "__main__"
def run_generator(config_file=CONFIG_FILE, vectors_file=VECTORS_FILE, 
                  output_dir=OUTPUT_DIR, specific_phase=SPECIFIC_PHASE):
    """Fonction principale pour exécuter le générateur de graphiques"""
    
    # Assure que le répertoire de sortie existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Lecture des fichiers de configuration
    configs, vectors = read_csv_files(config_file, vectors_file)
    
    # Réorganisation des configurations pour suivre l'ordre: Phase 1-A, Phase 1-B, Phase 2-A, etc.
    sorted_configs = []
    for phase_id in sorted(set(c['phase_id'] for c in configs)):
        for actor_id in sorted(set(c['actor_id'] for c in configs)):
            # Trouver la configuration correspondante
            for config in configs:
                if config['phase_id'] == phase_id and config['actor_id'] == actor_id:
                    sorted_configs.append(config)
                    break
    
    results = []
    
    # Traitement de chaque configuration selon l'ordre établi
    for config in sorted_configs:
        actor_id = config['actor_id']
        phase_id = config['phase_id']
        
        # Si une phase spécifique est demandée, passer les autres
        if specific_phase and specific_phase != f"{actor_id}{phase_id}":
            continue
            
        print(f"\nTraitement de la Phase {phase_id}, Acteur {actor_id}...")
        result = generate_graph(config, vectors, output_dir)
        results.append(result)
    
    # Création du fichier récapitulatif
    create_summary_file(results, output_dir)
    
    print(f"\nTous les graphiques ont été générés avec succès dans: {output_dir}")
    return results

# Exécuter directement - c'est ce qui sera lancé quand vous exécuterez la cellule dans Colab
run_generator()
# 6 cluster (id C1 to C6)
# 6 satellites (id CiS1 to CiS6) par cluster (i) dont 2 noeuds critiques (id CiS5 et CiS6)
# Chaque noeuds critique sont relié  à tous les noeuds critique de même id

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import math

sizeParameters = 2000

def generate_positions(num_clusters=6, num_satellites=6, space_size=6.5, cluster_radius=0.9):
    cluster_centers = np.random.rand(num_clusters, 3) * (space_size - 2.5) + 1.25  

    angles_xy = np.random.rand(num_clusters, num_satellites) * 2 * np.pi
    angles_z = (np.random.rand(num_clusters, num_satellites) - 0.5) * np.pi / 4
    radii = np.random.rand(num_clusters, num_satellites) * cluster_radius
    positions = np.zeros((num_clusters, num_satellites, 3))
    
    positions[..., 0] = cluster_centers[:, None, 0] + radii * np.cos(angles_xy)
    positions[..., 1] = cluster_centers[:, None, 1] + radii * np.sin(angles_xy)
    positions[..., 2] = cluster_centers[:, None, 2] + radii * np.sin(angles_z)
    
    all_positions = []

    for _ in range(sizeParameters):
        random_movements = (np.random.rand(num_clusters, num_satellites, 3) - 0.5) * 0.08
        
        new_positions = positions + random_movements
        
        distances = np.linalg.norm(new_positions - cluster_centers[:, None, :], axis=-1)
        mask = distances > cluster_radius 
        new_positions[mask] = positions[mask]
        
        positions = new_positions
        all_positions.append(positions.copy())

    return all_positions, cluster_centers

def animate_positions(all_positions, cluster_centers):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.clear()
        ax.set_xlim(0, 6.5)
        ax.set_ylim(0, 6.5)
        ax.set_zlim(0, 6.5)
        
        # Afficher les centres des clusters
        #for center in cluster_centers:
        #    ax.scatter(*center, color='red', marker='x', s=100, label="Centre" if center is cluster_centers[0] else "")

        for i in range(len(all_positions[0])):
            x, y, z = all_positions[frame][i].T
            ax.scatter(x, y, z, label=f'Cluster {i+1}')
        
        ax.legend()
    
    ani = animation.FuncAnimation(fig, update, frames=len(all_positions), interval=200)
    plt.show()
    return ani



import numpy as np

def calculate_distances(total_positions):
    """Calcule les distances entre tous les satellites à chaque instant, et retourne également les distances spécifiques à 'C1S1'.""" 
    distances = {}
    distances_known_by_C1S1 = {
        "C1S1_C1S5": [],
        "C1S1_C1S6": [],
        "C1S5_C2S5": [],
        "C1S5_C3S5": [],
        "C1S5_C4S5": [],
        "C1S5_C5S5": [],
        "C1S5_C6S5": [],
        "C1S6_C2S6": [],
        "C1S6_C3S6": [],
        "C1S6_C4S6": [],
        "C1S6_C5S6": [],
        "C1S6_C6S6": [],
    }
    
    for t in range(len(total_positions)):
        for i in range(1, 7):  
            for j in range(1, 7):  
                for n in range(1, 7):  
                    for k in range(1, 7): 

                        link = f"C{i}S{j}_C{n}S{k}"
                        
                        position_Cij = total_positions[t][i-1, j-1]
                        position_Cnk = total_positions[t][n-1, k-1]
                        
                        distance = np.linalg.norm(position_Cij - position_Cnk)
                        
                        if link not in distances:
                            distances[link] = []
                        distances[link].append(distance)
                        
                        if link in distances_known_by_C1S1:
                            distances_known_by_C1S1[link].append(distance)
    
    return distances_known_by_C1S1, distances


def plot_average_distance_over_time(distances):
    """
    Prend un dictionnaire de distances (clé: lien, valeur: liste des distances dans le temps)
    et trace la moyenne des distances à chaque instant t.
    """
    all_distances = np.array(list(distances.values()))  # shape: (nb_links, n_iterations)

    # Calcul de la moyenne à chaque instant t
    average_over_time = np.mean(all_distances, axis=0)

    # Tracé
    plt.figure(figsize=(12, 5))
    plt.plot(average_over_time, label="Moyenne des distances", color="tab:green")
    plt.title("Évolution de la distance moyenne entre satellites au cours du temps")
    plt.xlabel("Itération t")
    plt.ylabel("Distance moyenne")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def generate_congestion_array():
    """Génère un tableau de congestion avec des combinaisons de sinusoïdes déphasées et du bruit."""
    t = np.linspace(0, 2 * np.pi, sizeParameters)

    freq1 = np.random.uniform(0.5, 2)
    freq2 = np.random.uniform(0.5, 3)
    phase1 = np.random.uniform(0, 2 * np.pi)
    phase2 = np.random.uniform(0, 2 * np.pi)
    amp1 = np.random.uniform(1, 5)
    amp2 = np.random.uniform(0.5, 3)

    wave = (
        amp1 * np.sin(freq1 * t + phase1)
        + amp2 * np.cos(freq2 * t + phase2)
    )

    noise = np.random.normal(0, 0.5, sizeParameters)

    peaks = np.zeros(sizeParameters)
    if np.random.rand() > 0.3:
        peak_centers = np.random.choice(sizeParameters, size=int(sizeParameters * 0.05), replace=False)
        for peak in peak_centers:
            smooth_peak = np.exp(-0.5 * ((np.arange(sizeParameters) - peak) / 5) ** 2)
            peaks += smooth_peak * np.random.uniform(3, 8)

    result = wave + noise + peaks

    result -= result.min()
    current_mean = np.mean(result)
    if current_mean > 0:
        result *= (5.0 / current_mean)

    return result

def generate_congestion():
    """Génère toutes les congestions avec méthode uniforme"""
    
    congestions_known_by_C1S1_links = [
        "C1S1_C1S5", "C1S1_C1S6",
        "C1S5_C2S5", "C1S5_C3S5", "C1S5_C4S5", "C1S5_C5S5", "C1S5_C6S5",
        "C1S6_C2S6", "C1S6_C3S6", "C1S6_C4S6", "C1S6_C5S6", "C1S6_C6S6",
    ]

    congestions_known_by_C1S1 = {
        link: generate_congestion_array()
        for link in congestions_known_by_C1S1_links
    }

    congestions = {}

    for i in range(1, 7):
        for j in range(1, 7):
            for k in range(1, 7):
                for n in range(1, 7):
                    link = f"C{i}S{j}_C{k}S{n}"
                    if link in congestions_known_by_C1S1:
                        congestions[link] = congestions_known_by_C1S1[link]
                    else:
                        congestions[link] = generate_congestion_array()

    return congestions_known_by_C1S1, congestions






def plot_congestion(congestions):
    """
    Affiche les congestions.
    - Si un dict est passé, on utilise les clés comme noms.
    - Si une liste de tuples (name, data) est passée, on les utilise directement.
    """
    
    if isinstance(congestions, dict):
        data_items = list(congestions.items())
    elif isinstance(congestions, list):
        if isinstance(congestions[0], (list, np.ndarray)):
            data_items = [(f"Congestion {i+1}", data) for i, data in enumerate(congestions)]
        else:
            data_items = congestions
    else:
        raise ValueError("Format de données non supporté. Attendu dict ou list.")

    total = len(data_items)
    cols = 4
    rows = math.ceil(total / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
    axes = axes.flatten()

    for i, (link, data) in enumerate(data_items):
        ax = axes[i]
        ax.plot(range(len(data)), data, label=link)
        ax.set_title(f"Congestion: {link}")
        ax.set_xlabel("Instant")
        ax.set_ylabel("Niveau")
        ax.grid(True)
        ax.legend()

    for j in range(len(data_items), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_average_congestion_over_time(congestions):
    """
    Prend un dictionnaire de congestions et trace la moyenne
    de toutes les courbes de congestion à chaque instant t.
    """
    all_values = np.array(list(congestions.values())) 

    average_over_time = np.mean(all_values, axis=0)

    plt.figure(figsize=(12, 5))
    plt.plot(average_over_time, label='Moyenne des congestions')
    plt.title("Évolution de la moyenne des congestions au cours du temps")
    plt.xlabel("Itération t")
    plt.ylabel("Congestion moyenne")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


from .parameters import sizeParameters, generate_congestion, generate_positions, calculate_distances
import math

# Generate system state once when module is imported
congestions_known_by_C1S1, congestions = generate_congestion()
total_positions, centers = generate_positions()
distances_known_by_C1S1, distances = calculate_distances(total_positions)

def calculate_cost(path, iteration):
    # Identical body used across project files
    travel_time = sum(congestions[lien][iteration] for lien in path)
    energy_cost = sum((distances[lien][iteration]*2) ** 2 for lien in path) + sum((distances[lien][iteration]) ** 3 for lien in path)
    return travel_time + energy_cost / 4

def moving_average(data, window):
    # Identical body used across project files
    avg = []
    for j in range(len(data)):
        start = max(0, j - window + 1)
        window_vals = data[start:j + 1]
        avg.append(sum(window_vals) / len(window_vals))
    return avg

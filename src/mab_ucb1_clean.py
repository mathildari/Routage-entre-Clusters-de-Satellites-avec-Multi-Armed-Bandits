
import random
import math
import matplotlib.pyplot as plt
from collections import defaultdict

from .parameters import sizeParameters
from .utils import calculate_cost, moving_average

def generate_possible_paths(destination_satellite):
    # Tuple-returning version (from mab_ucb1.py) expected by mab_ucb()
    clusters = [2, 3, 4, 5, 6]
    source = "C1S1"
    possible_paths = []
    for cluster in clusters:
        if cluster == int(destination_satellite[1]):
            path = (f"{source}_C1S5", f"C1S5_C{cluster}S5", f"C{cluster}S5_{destination_satellite}")
            possible_paths.append((path, 'direct'))
            path = (f"{source}_C1S6", f"C1S6_C{cluster}S6", f"C{cluster}S6_{destination_satellite}")
            possible_paths.append((path, 'direct'))
            for inter_cluster in range(2, 7):
                if inter_cluster != cluster:
                    path = (f"{source}_C1S5", f"C1S5_C{inter_cluster}S5", f"C{inter_cluster}S5_C{cluster}S5", f"C{cluster}S5_{destination_satellite}")
                    possible_paths.append((path, 'indirect'))
                    path = (f"{source}_C1S6", f"C1S6_C{inter_cluster}S6", f"C{inter_cluster}S6_C{cluster}S6", f"C{cluster}S6_{destination_satellite}")
                    possible_paths.append((path, 'indirect'))
    return possible_paths

def mab_ucb():
    rewards_mab = []
    rewards_random = []
    rewards_optimal = []
    cumulative_regret_mab = [0]
    window_size = 1000
    n_iterations = 10000

    prefix_mab = defaultdict(lambda: {"rewards": {}, "counts": {}})

    history_optimal = []
    direct_ratio_optimal = []
    direct_ratio_mab = []
    direct_ratio_random = []
    history_mab = []
    history_random = []

    for i in range(1, n_iterations):
        i_dest = random.randint(2, 6)
        j_dest = random.randint(1, 4)
        destination = f"C{i_dest}S{j_dest}"
        possible_paths = generate_possible_paths(destination)

        def path_prefix(path):
            return path[:-1]

        best_path = None
        best_ucb = float('inf')

        for path, _ in possible_paths:
            prefix = path_prefix(path)
            rewards_dict = prefix_mab[prefix]["rewards"]
            counts_dict = prefix_mab[prefix]["counts"]

            if path not in rewards_dict:
                best_path = path
                break
            else:
                avg = rewards_dict[path]
                n = counts_dict[path]
                bonus = math.sqrt(2 * math.log(i + 1) / n)
                score = avg - bonus  # minimization
                if score < best_ucb:
                    best_ucb = score
                    best_path = path

        chosen_path_mab = best_path
        reward_mab = calculate_cost(chosen_path_mab, i % sizeParameters)

        prefix = path_prefix(chosen_path_mab)
        rewards_dict = prefix_mab[prefix]["rewards"]
        counts_dict = prefix_mab[prefix]["counts"]

        if chosen_path_mab not in rewards_dict:
            rewards_dict[chosen_path_mab] = reward_mab
            counts_dict[chosen_path_mab] = 1
        else:
            counts_dict[chosen_path_mab] += 1
            rewards_dict[chosen_path_mab] += (reward_mab - rewards_dict[chosen_path_mab]) / counts_dict[chosen_path_mab]

        chosen_type_mab = next((t for p, t in possible_paths if p == chosen_path_mab), None)
        history_mab.append(chosen_type_mab)
        recent_mab = history_mab[-window_size:] if len(history_mab) >= window_size else history_mab
        direct_ratio_mab.append(recent_mab.count("direct") / len(recent_mab))

        chosen_path_random = random.choice([path for path, _ in possible_paths])
        reward_random = calculate_cost(chosen_path_random, i % sizeParameters)
        rewards_random.append(reward_random)
        chosen_type_random = next((t for p, t in possible_paths if p == chosen_path_random), None)
        history_random.append(chosen_type_random)
        recent_random = history_random[-window_size:] if len(history_random) >= window_size else history_random
        direct_ratio_random.append(recent_random.count("direct") / len(recent_random))

        best_cost_opt = float('inf')
        best_type_opt = None
        for path, typ in possible_paths:
            cost = calculate_cost(path, i % sizeParameters)
            if cost < best_cost_opt:
                best_cost_opt = cost
                best_type_opt = typ

        rewards_mab.append(reward_mab)
        rewards_optimal.append(best_cost_opt)
        instant_regret = reward_mab - best_cost_opt
        cumulative_regret_mab.append(cumulative_regret_mab[-1] + instant_regret)
        history_optimal.append(best_type_opt)
        recent_optimal = history_optimal[-window_size:] if len(history_optimal) >= window_size else history_optimal
        direct_ratio_optimal.append(recent_optimal.count("direct") / len(recent_optimal))

    moving_avg_mab = moving_average(rewards_mab, window_size)
    moving_avg_random = moving_average(rewards_random, window_size)
    moving_avg_optimal = moving_average(rewards_optimal, window_size)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_iterations), moving_avg_mab, label="UCB1", linewidth=2)
    plt.plot(range(1, n_iterations), moving_avg_random, label="Choix aléatoire", linewidth=2)
    plt.plot(range(1, n_iterations), moving_avg_optimal, label="Choix optimal", linewidth=2)
    plt.xlabel("Itération")
    plt.ylabel("Coût observé (récompense)")
    plt.title("Moyenne mobile (UCB1)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_iterations), direct_ratio_mab, label="UCB1 % direct", linewidth=2)
    plt.plot(range(1, n_iterations), direct_ratio_random, label="Random % direct", linewidth=2)
    plt.plot(range(1, n_iterations), direct_ratio_optimal, label="Optimal % direct", linewidth=2)
    plt.xlabel("Itération")
    plt.ylabel("Proportion de chemins directs")
    plt.title("UCB1 : proportion de chemins directs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_iterations), cumulative_regret_mab[1:], label="Regret cumulé UCB1", linewidth=2)
    plt.xlabel("Itération")
    plt.ylabel("Regret cumulé")
    plt.title("Évolution du regret cumulé (UCB1)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

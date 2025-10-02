
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import math

from .parameters import sizeParameters
from .utils import calculate_cost, moving_average

def generate_possible_paths(destination_satellite):
    # Dict-returning version (from mab.py) expected by mab(), allMAB(), run_multiple_mab_runs()
    clusters = [2, 3, 4, 5, 6]
    source = "C1S1"
    possible_paths = []
    for cluster in clusters:
        if cluster == int(destination_satellite[1]):
            path = (f"{source}_C1S5", f"C1S5_C{cluster}S5", f"C{cluster}S5_{destination_satellite}")
            possible_paths.append({
                "complet": path,
                "a_enregistrer": path[:-1],
                "type": "direct"
            })
            path = (f"{source}_C1S6", f"C1S6_C{cluster}S6", f"C{cluster}S6_{destination_satellite}")
            possible_paths.append({
                "complet": path,
                "a_enregistrer": path[:-1],
                "type": "direct"
            })
            for inter_cluster in range(2, 7):
                if inter_cluster != cluster:
                    path = (f"{source}_C1S5", f"C1S5_C{inter_cluster}S5", f"C{inter_cluster}S5_C{cluster}S5", f"C{cluster}S5_{destination_satellite}")
                    possible_paths.append({
                        "complet": path,
                        "a_enregistrer": path[:-1],
                        "type": "indirect"
                    })
                    path = (f"{source}_C1S6", f"C1S6_C{inter_cluster}S6", f"C{inter_cluster}S6_C{cluster}S6", f"C{cluster}S6_{destination_satellite}")
                    possible_paths.append({
                        "complet": path,
                        "a_enregistrer": path[:-1],
                        "type": "indirect"
                    })
    return possible_paths

def mab(strategy="MAB_EPSILON_GREEDY", plot=False, regret=False):
    rewards_mab = []
    rewards_random = []
    rewards_optimal = []
    path_count_mab = defaultdict(int)
    path_count_random = defaultdict(int)
    window_size = 250
    n_iterations = 2000
    epsilon = 1

    cumulative_regret_mab = [0] 
    epsilon_values = []
    previous_rewards = {}
    n_choices = {}
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

        if strategy == "MAB_EPSILON_GREEDY_DECAYING":
            epsilon = 1 * (1 - i / n_iterations)
        elif strategy == "MAB_GREEDY":
            epsilon = 0.0
        elif strategy == "MAB_EPSILON_GREEDY":
            epsilon = 0.1
        epsilon_values.append(epsilon)

        if strategy == "MAB_UCB":
            best_entry = None
            best_ucb_score = float('inf')
            for entry in possible_paths:
                key = entry["a_enregistrer"]
                if key not in previous_rewards:
                    chosen_entry = entry
                    break  # exploration forcée
                avg = previous_rewards[key]
                count = n_choices[key]
                confidence = math.sqrt(2 * math.log(i + 1) / count)
                ucb_score = avg - confidence  # on veut minimiser le coût
                if ucb_score < best_ucb_score:
                    best_ucb_score = ucb_score
                    best_entry = entry
            else:
                chosen_entry = best_entry
        else:
            use_exploration = (random.random() < epsilon)
            if use_exploration:
                chosen_entry = random.choice(possible_paths)
            else:
                best_entry = None
                best_avg_reward = float('inf')
                for entry in possible_paths:
                    key = entry["a_enregistrer"]
                    if key in previous_rewards:
                        avg_reward = previous_rewards[key]
                        if avg_reward < best_avg_reward:
                            best_avg_reward = avg_reward
                            best_entry = entry
                if best_entry is None:
                    chosen_entry = random.choice(possible_paths)
                else:
                    chosen_entry = best_entry

        chosen_path_mab = chosen_entry["complet"]
        path_key_mab = chosen_entry["a_enregistrer"]
        path_type_mab = chosen_entry["type"]

        path_count_mab[chosen_path_mab] += 1

        reward_mab = calculate_cost(chosen_path_mab, i % sizeParameters)
        rewards_mab.append(reward_mab)

        if path_key_mab not in previous_rewards:
            previous_rewards[path_key_mab] = reward_mab
            n_choices[path_key_mab] = 1
        else:
            n_choices[path_key_mab] += 1
            previous_rewards[path_key_mab] += (reward_mab - previous_rewards[path_key_mab]) / n_choices[path_key_mab]

        history_mab.append(path_type_mab)
        recent_mab = history_mab[-window_size:] if len(history_mab) >= window_size else history_mab
        direct_ratio_mab.append(recent_mab.count("direct") / len(recent_mab))

        chosen_entry_random = random.choice(possible_paths)
        chosen_path_random = chosen_entry_random["complet"]
        path_type_random = chosen_entry_random["type"]

        path_count_random[chosen_path_random] += 1

        reward_random = calculate_cost(chosen_path_random, i % sizeParameters)
        rewards_random.append(reward_random)

        history_random.append(path_type_random)
        recent_random = history_random[-window_size:] if len(history_random) >= window_size else history_random
        direct_ratio_random.append(recent_random.count("direct") / len(recent_random))

        best_cost_opt = float('inf')
        best_type_opt = None
        for entry in possible_paths:
            cost = calculate_cost(entry["complet"], i % sizeParameters)
            if cost < best_cost_opt:
                best_cost_opt = cost
                best_type_opt = entry["type"]

        rewards_optimal.append(best_cost_opt)
        history_optimal.append(best_type_opt)
        recent_optimal = history_optimal[-window_size:] if len(history_optimal) >= window_size else history_optimal
        direct_ratio_optimal.append(recent_optimal.count("direct") / len(recent_optimal))

        instant_regret = reward_mab - best_cost_opt
        cumulative_regret_mab.append(cumulative_regret_mab[-1] + instant_regret)

    if (not plot):
        return rewards_mab, rewards_random, rewards_optimal, direct_ratio_mab, direct_ratio_random, direct_ratio_optimal

    avg_regret = [cumulative_regret_mab[i] / i for i in range(1, len(cumulative_regret_mab))]
    if (regret):
        return cumulative_regret_mab, avg_regret

    moving_avg_mab = moving_average(rewards_mab, window_size)
    moving_avg_random = moving_average(rewards_random, window_size)
    moving_avg_optimal = moving_average(rewards_optimal, window_size)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_iterations), moving_avg_mab, label="MAB " + strategy, linewidth=2)
    plt.plot(range(1, n_iterations), moving_avg_random, label="Choix aléatoire", linewidth=2)
    plt.plot(range(1, n_iterations), moving_avg_optimal, label="Choix optimal", linewidth=2)
    plt.xlabel("Itération")
    plt.ylabel("Coût observé (récompense)")
    plt.title(f"Moyenne mobile ({window_size}) du coût à chaque itération : " + strategy)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_iterations), direct_ratio_mab, label="MAB pourcentage de direct", linewidth=2)
    plt.plot(range(1, n_iterations), direct_ratio_random, label="Random pourcentage de direct", linewidth=2)
    plt.plot(range(1, n_iterations), direct_ratio_optimal, label="Optimal pourcentage de direct", linewidth=2)
    plt.plot(range(1, n_iterations), epsilon_values, label="Epsilon", linewidth=1)
    plt.xlabel("Itération")
    plt.ylabel("Proportion de chemins directs")
    plt.title("Évolution des choix de chemins directs au cours du temps : " + strategy)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_iterations), cumulative_regret_mab[1:], label="Regret cumulé MAB", linewidth=2)
    plt.xlabel("Itération")
    plt.ylabel("Regret cumulé")
    plt.title("Évolution du regret cumulé pour la stratégie " + strategy)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(avg_regret)
    plt.xlabel("Itération")
    plt.ylabel("Regret moyen")
    plt.title("Regret moyen par itération " + strategy)
    plt.grid(True)
    plt.show()

def allMAB():
    strategies = [
        "MAB_EPSILON_GREEDY",
        "MAB_GREEDY",
        "MAB_EPSILON_GREEDY_DECAYING",
        "MAB_UCB"
    ]

    results = {}
    window_size = 250
    n_iterations = 1000

    for strategy in strategies:
        rewards_mab = []
        path_count_mab = defaultdict(int)
        epsilon_start = 0.1
        epsilon = epsilon_start

        cumulative_regret_mab = [0] 
        epsilon_values = []
        previous_rewards = {}
        n_choices = {}
        history_mab = []
        direct_ratio_mab = []

        rewards_optimal = []
        history_optimal = []
        direct_ratio_optimal = []

        for i in range(1, n_iterations):
            i_dest = random.randint(2, 6)
            j_dest = random.randint(1, 4)
            destination = f"C{i_dest}S{j_dest}"
            possible_paths = generate_possible_paths(destination)

            if strategy == "MAB_EPSILON_GREEDY_DECAYING":
                epsilon = epsilon_start * (1 - i / n_iterations)
            elif strategy == "MAB_GREEDY":
                epsilon = 0.0
            elif strategy == "MAB_EPSILON_GREEDY":
                epsilon = epsilon_start
            epsilon_values.append(epsilon)

            if strategy == "MAB_UCB":
                best_entry = None
                best_ucb_score = float('inf')
                for entry in possible_paths:
                    key = entry["a_enregistrer"]
                    if key not in previous_rewards:
                        chosen_entry = entry
                        break  # forced exploration
                    avg = previous_rewards[key]
                    count = n_choices[key]
                    confidence = math.sqrt(2 * math.log(i + 1) / count)
                    ucb_score = avg - confidence  # minimization
                    if ucb_score < best_ucb_score:
                        best_ucb_score = ucb_score
                        best_entry = entry
                else:
                    chosen_entry = best_entry
            else:
                use_exploration = (random.random() < epsilon)
                if use_exploration:
                    chosen_entry = random.choice(possible_paths)
                else:
                    best_entry = None
                    best_avg_reward = float('inf')
                    for entry in possible_paths:
                        key = entry["a_enregistrer"]
                        if key in previous_rewards:
                            avg_reward = previous_rewards[key]
                            if avg_reward < best_avg_reward:
                                best_avg_reward = avg_reward
                                best_entry = entry
                    if best_entry is None:
                        chosen_entry = random.choice(possible_paths)
                    else:
                        chosen_entry = best_entry

            chosen_path_mab = chosen_entry["complet"]
            path_key_mab = chosen_entry["a_enregistrer"]
            path_type_mab = chosen_entry["type"]

            path_count_mab[chosen_path_mab] += 1
            reward_mab = calculate_cost(chosen_path_mab, i % sizeParameters)
            rewards_mab.append(reward_mab)

            if path_key_mab not in previous_rewards:
                previous_rewards[path_key_mab] = reward_mab
                n_choices[path_key_mab] = 1
            else:
                n_choices[path_key_mab] += 1
                previous_rewards[path_key_mab] += (reward_mab - previous_rewards[path_key_mab]) / n_choices[path_key_mab]

            history_mab.append(path_type_mab)
            recent_mab = history_mab[-window_size:] if len(history_mab) >= window_size else history_mab
            direct_ratio_mab.append(recent_mab.count("direct") / len(recent_mab))

            best_cost_opt = float('inf')
            best_type_opt = None
            for entry in possible_paths:
                cost = calculate_cost(entry["complet"], i % sizeParameters)
                if cost < best_cost_opt:
                    best_cost_opt = cost
                    best_type_opt = entry["type"]

            rewards_optimal.append(best_cost_opt)
            history_optimal.append(best_type_opt)
            recent_optimal = history_optimal[-window_size:] if len(history_optimal) >= window_size else history_optimal
            direct_ratio_optimal.append(recent_optimal.count("direct") / len(recent_optimal))

            instant_regret = reward_mab - best_cost_opt
            cumulative_regret_mab.append(cumulative_regret_mab[-1] + instant_regret)

        moving_avg = moving_average(rewards_mab, window_size)
        avg_regret = [cumulative_regret_mab[i] / i for i in range(1, len(cumulative_regret_mab))]

        results[strategy] = {
            "moving_avg": moving_avg,
            "direct_ratio": direct_ratio_mab,
            "cumulative_regret": cumulative_regret_mab[1:],
            "avg_regret": avg_regret
        }

    # Plotting
    x = range(1, n_iterations)

    plt.figure(figsize=(10, 5))
    for strategy in strategies:
        plt.plot(x, results[strategy]["moving_avg"], label=strategy)
    plt.xlabel("Itération")
    plt.ylabel("Coût observé (récompense)")
    plt.title("Moyenne mobile des coûts observés")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    for strategy in strategies:
        plt.plot(x, results[strategy]["direct_ratio"], label=strategy)
    plt.xlabel("Itération")
    plt.ylabel("Proportion de chemins directs")
    plt.title("Évolution des choix de chemins directs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    for strategy in strategies:
        plt.plot(x, results[strategy]["cumulative_regret"], label=strategy)
    plt.xlabel("Itération")
    plt.ylabel("Regret cumulé")
    plt.title("Évolution du regret cumulé")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    for strategy in strategies:
        plt.plot(results[strategy]["avg_regret"], label=strategy)
    plt.xlabel("Itération")
    plt.ylabel("Regret moyen")
    plt.title("Regret moyen par itération")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_multiple_mab_runs(n_runs=100):
    from .utils import calculate_cost  # ensure same state across runs
    from .parameters import generate_congestion, generate_positions, calculate_distances

    global congestions_known_by_C1S1, congestions
    global total_positions, centers
    global distances_known_by_C1S1, distances

    strategies = ["MAB_GREEDY", "MAB_EPSILON_GREEDY", "MAB_EPSILON_GREEDY_DECAYING", "MAB_UCB"]
    avg_rewards = {}
    avg_direct_ratios = {}
    reward_window = 50
    direct_ratio_window = 100

    for strategy in strategies:
        avg_rewards[strategy] = {"mab": None, "random": None, "optimal": None}
        avg_direct_ratios[strategy] = {"mab": None, "random": None, "optimal": None}

    import numpy as np

    for i in range(n_runs):
        c_known, c = generate_congestion()
        positions, cts = generate_positions()
        d_known, d = calculate_distances(positions)

        congestions_known_by_C1S1, congestions = c_known, c
        total_positions, centers = positions, cts
        distances_known_by_C1S1, distances = d_known, d

        for strategy in strategies:
            rewards_mab, rewards_random, rewards_optimal, direct_ratio_mab, direct_ratio_random, direct_ratio_optimal = mab(strategy=strategy)

            for kind, data in zip(["mab", "random", "optimal"], [rewards_mab, rewards_random, rewards_optimal]):
                arr = np.array(data, dtype=np.float64)
                avg_rewards[strategy][kind] = arr if avg_rewards[strategy][kind] is None else avg_rewards[strategy][kind] + arr

            for kind, data in zip(["mab", "random", "optimal"], [direct_ratio_mab, direct_ratio_random, direct_ratio_optimal]):
                arr = np.array(data, dtype=np.float64)
                avg_direct_ratios[strategy][kind] = arr if avg_direct_ratios[strategy][kind] is None else avg_direct_ratios[strategy][kind] + arr

    for strategy in strategies:
        for key in avg_rewards[strategy]:
            avg_rewards[strategy][key] /= n_runs
        for key in avg_direct_ratios[strategy]:
            avg_direct_ratios[strategy][key] /= n_runs

    plt.figure(figsize=(12, 6))
    for strategy in strategies:
        y = moving_average(avg_rewards[strategy]["mab"], reward_window)
        plt.plot(range(reward_window, reward_window + len(y)), y, label=f"{strategy} - MAB", linewidth=2)

    y_random = moving_average(avg_rewards[strategy]["random"], reward_window)
    y_optimal = moving_average(avg_rewards[strategy]["optimal"], reward_window)
    plt.plot(range(0, 0 + len(y_random)), y_random, label="Random", linestyle="--")
    plt.plot(range(0, 0 + len(y_optimal)), y_optimal, label="Optimal", linestyle="--")

    plt.title(f"Coût moyen par itération ({n_runs} exécutions)")
    plt.xlabel("Itération")
    plt.ylabel("Coût")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

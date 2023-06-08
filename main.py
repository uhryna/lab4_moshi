import random
import matplotlib.pyplot as plt
import numpy as np
import warnings
from matplotlib import MatplotlibDeprecationWarning

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


def generate_map():
    n = random.randint(25, 35)
    distances = [[random.randint(10, 100) for j in range(n)] for i in range(n)]
    with open("map.txt", "w") as f:
        f.write(str(n) + "\n")
        for i in range(n):
            f.write(" ".join(str(x) for x in distances[i]) + "\n")


def read_map():
    with open("map.txt", "r") as f:
        n = int(f.readline().strip())
        distances = []
        for i in range(n):
            row = [int(x) for x in f.readline().split()]
            distances.append(row)
    return n, distances


def print_map(n, distances):
    print("Кількість міст: ", n)
    print("Матриця відстаней:")
    for i in range(n):
        print(distances[i])


def ant_colony_optimization(distance_matrix, num_ants, alpha, beta, evaporation_rate, q0, pheromone_initial,num_iterations):
    n = len(distance_matrix)
    ant_locations = random.sample(range(n), num_ants)
    pheromone_matrix = [[pheromone_initial for _ in range(n)] for _ in range(n)]
    shortest_distance = float('inf')
    shortest_path = []

    for iter in range(num_iterations):
        for i in range(num_ants):
            current_location = ant_locations[i]
            allowed_locations = [j for j in range(n) if j != current_location]
            next_location = None

            if random.random() < q0:
                # вибір міста з більшою концентрацією ферменту
                pheromone_values = [(j, pheromone_matrix[current_location][j]) for j in allowed_locations]
                pheromone_values.sort(key=lambda x: x[1], reverse=True)
                next_location = pheromone_values[0][0]
            else:

                # Вибрати локацію випадковим чином в залежності від ймовірності, яка залежить від відстані та концентрації ферменту
                probabilities = [(j, (pheromone_matrix[current_location][j]) ** alpha *
                                  (1 / distance_matrix[current_location][j]) ** beta) for j in allowed_locations]
                probabilities_sum = sum(p[1] for p in probabilities)
                probabilities = [(p[0], p[1] / probabilities_sum) for p in probabilities]
                probabilities.sort(key=lambda x: x[1], reverse=True)
                random_value = random.random()
                cumulative_probability = 0
                for p in probabilities:
                    cumulative_probability += p[1]
                    if random_value <= cumulative_probability:
                        next_location = p[0]
                        break

            ant_locations[i] = next_location

            # оновлення значення ферменту
            for j in range(n):
                if j != current_location:
                    pheromone_matrix[current_location][j] *= (1 - evaporation_rate)
                    pheromone_matrix[current_location][j] += evaporation_rate * pheromone_initial
                if j == next_location:
                    pheromone_matrix[current_location][j] += q0 * pheromone_initial + (1 - q0) * \
                                                             pheromone_matrix[current_location][j]
        # випарювання ферменту
        for i in range(n):
            for j in range(n):
                pheromone_matrix[i][j] *= (1 - evaporation_rate)

        # знаходження найкоротшого шляху
        for start in range(n):
            visited = {start}
            distance = 0
            path = [start]

            while len(visited) < n:
                current_location = path[-1]
                distances = [(j, distance_matrix[current_location][j]) for j in allowed_locations if j not in visited]
                if not distances:
                    break
                next_location = min(distances, key=lambda x: x[1])[0]
                visited.add(next_location)
                path.append(next_location)
                distance += distance_matrix[current_location][next_location]

            # add the distance back to the starting point
            distance += distance_matrix[path[-1]][start]

            if distance < shortest_distance:
                shortest_distance = distance
                shortest_path = path + [start]
        print(f"Iteration {iter+1}: shortest path length = {shortest_distance}, shortest path = {shortest_path}")
    return shortest_path, shortest_distance


def plot_map(best_path, cities_amount):
    best_x = []
    best_y = []
    fig, ax = plt.subplots(figsize=(8, 8))

    # Coordinates of center point
    cx, cy = 0.5, 0.5

    # Radius of the circle
    r = 0.4

    # Calculate the angles of the vertices in the polygon
    angles = np.linspace(0, 2 * np.pi, cities_amount, endpoint=False)

    # Calculate the x, y coordinates of the vertices
    x = r * np.cos(angles) + cx
    y = r * np.sin(angles) + cy

    for i in range(cities_amount):
        for j in range(cities_amount):
            if i < j:
                ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', linewidth=0.3, alpha=0.4)

    for i in range(cities_amount):
        if i == best_path[0]:
            circle = plt.Circle((x[i], y[i]), 0.018, color='red', zorder=10)
        else:
            circle = plt.Circle((x[i], y[i]), 0.018, color='green', zorder=10)

        ax.add_artist(circle)
        ax.text(x[i], y[i], str(i), fontsize=11, color='white', fontweight='bold', zorder=10, ha='center', va='center')
        best_x = [x[i] for i in best_path]
        best_y = [y[i] for i in best_path]

    ax.quiver(best_x[:-1], best_y[:-1], np.diff(best_x), np.diff(best_y),
              angles='xy', scale_units='xy', scale=1, color='red', alpha=0.7, width=0.0055,
              headlength=7, headwidth=4, headaxislength=5, )

    ax.plot(best_x + [best_x[0]], best_y + [best_y[0]], 'o-', color='green', linewidth=0.7, alpha=0.7,
            label='Optimal Path')
    ax.scatter(x, y, zorder=6, s=30, color='green', label='Cities')
    ax.legend(loc='best', fontsize=12)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Traveling Salesman Problem', fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    #generate_map()
    n, distances = read_map()
    #print_map(n, distances)
    num_ants = n
    alpha = 1
    beta = 2
    evaporation_rate = 0.5
    q0 = 0.5
    pheromone_initial = 1.0
    num_iterations = 50
    shortest_path, shortest_distance=ant_colony_optimization(distances,num_ants, alpha, beta, evaporation_rate, q0, pheromone_initial, num_iterations)
    plot_map(shortest_path,n)
    print(f"\nShortest path length = {shortest_distance}, shortest path = {shortest_path}")


main()


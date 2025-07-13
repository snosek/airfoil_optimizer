import numpy as np
import matplotlib.pyplot as plt
import random
import neuralfoil as nf
from scipy.special import comb

# NeuralFoil config
ALPHA = 5  # deg
RE = 100e3
MODEL_SIZE = "xxxlarge"
CL_MIN = 0.1

# Bezier configuration
N_CTRL = 10
OPT_IDX = [*range(1,N_CTRL-1)]  # tylko te indeksy są optymalizowane
Y_RANGE = (0.0, 0.1)
POP_SIZE = 100
N_GEN = 50
MUTATION_RATE = 0.4
ELITISM_RATIO = 0.05
DIVERSITY_NUMBER = int(POP_SIZE * 0.4)

Y_RANGE = (0.0, 0.15)  # możliwe wartości wysokości punktów kontrolnych

def bezier_curve(control_points, n_points=100):
    n = len(control_points) - 1
    t = np.linspace(0, 1, n_points)
    curve = np.zeros((n_points, 2))
    for i in range(n + 1):
        bernstein = comb(n, i) * (1 - t) ** (n - i) * t ** i
        curve += np.outer(bernstein, control_points[i])
    return curve

def build_airfoil(y_opt_params):
    full_y = np.zeros(N_CTRL)
    full_y[OPT_IDX] = y_opt_params

    x_fixed = np.linspace(0.0, 1.0, N_CTRL)
    upper_ctrl = np.stack([x_fixed, full_y], axis=1)
    lower_ctrl = np.stack([x_fixed, -full_y], axis=1)

    upper = bezier_curve(upper_ctrl)
    lower = bezier_curve(lower_ctrl)[::-1]

    return np.vstack([upper, lower])

def evaluate(y_params):
    coords = build_airfoil(y_params)
    try:
        aero = nf.get_aero_from_coordinates(coords, alpha=ALPHA, Re=RE, model_size=MODEL_SIZE)
        cl, cd = aero["CL"][0], aero["CD"][0]
        
        if cl < CL_MIN or cd <= 0:
            return 1e3
        
        penalty = 100 * max(0, CL_MIN - cl)**2
        return -cl / cd + penalty
    except Exception as e:
        print(e)
        return 1e3

def random_individual():
    return np.random.uniform(*Y_RANGE, size=len(OPT_IDX))

def mutate(params):
    std_dev = 0.01
    mutated = params + np.random.normal(0, std_dev, size=params.shape)
    return np.clip(mutated, *Y_RANGE)

def crossover(p1, p2):
    alpha = np.random.uniform(0.2, 0.8)
    return alpha * p1 + (1 - alpha) * p2

def tournament_selection(population, fitness, k=3):
    candidates = random.sample(list(zip(population, fitness)), k)
    return min(candidates, key=lambda x: x[1])[0]

# === Główna pętla algorytmu genetycznego ===
population = [random_individual() for _ in range(POP_SIZE)]

for gen in range(N_GEN):
    fitness = [evaluate(ind) for ind in population]
    best_fit = min(fitness)
    print(f"Gen {gen}: best CL/CD = {-best_fit:.6f}")
    
    new_population = []
    num_elites = max(1, int(ELITISM_RATIO * POP_SIZE))
    elite_indices = np.argsort(fitness)[:num_elites]
    new_population.extend(population[i] for i in elite_indices)
    
    if gen % 5 == 0 and gen > 0:
        new_population.extend(random_individual() for _ in range(min(DIVERSITY_NUMBER, POP_SIZE - len(new_population))))
    
    while len(new_population) < POP_SIZE:
        p1 = tournament_selection(population, fitness)
        p2 = tournament_selection(population, fitness)
        child = mutate(crossover(p1, p2))
        new_population.append(child)
    
    population = new_population

# === Wyświetlenie najlepszego rozwiązania ===
final_fitness = [evaluate(ind) for ind in population]
best_idx = np.argmin(final_fitness)
best_params = population[best_idx]
best_coords = build_airfoil(best_params)

best_aero = nf.get_aero_from_coordinates(best_coords, alpha=ALPHA, Re=RE, model_size=MODEL_SIZE)

print("\n=== BEST BEZIER AIRFOIL ===")
print(f"Y control points: {best_params}")
print(f"CL = {best_aero['CL'][0]:.4f}, CD = {best_aero['CD'][0]:.4f}")
print(f"CL/CD = {best_aero['CL'][0] / best_aero['CD'][0]:.2f}")

# === Wizualizacja ===
plt.figure(figsize=(10, 4))
plt.plot(best_coords[:, 0], best_coords[:, 1], 'k', label="Optimized Airfoil")
x_ctrl = np.linspace(0.0, 1.0, N_CTRL)
full_y = np.zeros(N_CTRL)
full_y[OPT_IDX] = best_params

plt.plot(x_ctrl, full_y, 'o--', label="Upper control points")
plt.plot(x_ctrl, -full_y, 'o--', label="Lower control points")


plt.axis("equal")
plt.title("Optimized Symmetric Bezier Airfoil")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

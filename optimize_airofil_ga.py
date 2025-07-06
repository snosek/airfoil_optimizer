import random
import numpy as np
import neuralfoil as nf
import aerosandbox as asb

# genetic algorithm config
POP_SIZE = 60
N_GEN = 50
MUTATION_RATE = 0.5
ELITISM_RATIO = 0.1
DIVERSITY_NUMBER = int(POP_SIZE * 0.4)

# NeuralFoil config
ALPHA = 5  # degrees
RE = 100e3
MODEL_SIZE = "xxxlarge"
CL_MIN = 0.5

# NACA parameter ranges
M_RANGE = (0, 9)  # maximum camber
P_RANGE = (0, 9)    # position of max camber
T_RANGE = (1, 99)  # thickness

def random_individual():
    """Generate random NACA parameters [m, p, t]"""
    return np.array([
        np.random.uniform(*M_RANGE),
        np.random.uniform(*P_RANGE),
        np.random.uniform(*T_RANGE)
    ])

def decode_individual(params):
    """Convert parameters to NACA airfoil"""
    m, p, t = params
    digits = f"{int(m):01d}{int(p):01d}{int(t):02d}"
    return asb.Airfoil(name=f"NACA{digits}")

def evaluate(params):
    """Evaluate fitness: maximize CL/CD ratio with CL constraint"""
    try:
        aero = nf.get_aero_from_airfoil(
            airfoil=decode_individual(params),
            alpha=ALPHA, Re=RE, model_size=MODEL_SIZE
        )
        
        cd, cl, conf = aero["CD"][0], aero["CL"][0], aero["analysis_confidence"]
        
        # Reject unreliable or invalid results
        if conf < 0.9 or cd <= 0 or cl <= 0:
            return 1e3
        
        # Apply penalty for insufficient lift
        penalty = 100 * max(0, CL_MIN - cl)**2
        
        return -cl/cd + penalty  # minimize negative efficiency + penalty
        
    except Exception:
        return 1e3

def crossover(p1, p2):
    """Blend crossover"""
    alpha = np.random.uniform(0.3, 0.7)
    return alpha * p1 + (1 - alpha) * p2

def mutate(params):
    """Gaussian mutation with parameter-specific standard deviations"""
    params = params.copy()
    std_devs = [0.2, 0.1, 5]  # different mutation rates for m, p, t
    ranges = [M_RANGE, P_RANGE, T_RANGE]
    
    for i in range(3):
        if np.random.rand() < MUTATION_RATE:
            params[i] += np.random.normal(0, std_devs[i])
            params[i] = np.clip(params[i], *ranges[i])
    
    return params

def tournament_selection(population, fitness, k=3):
    """Tournament selection"""
    candidates = random.sample(list(zip(population, fitness)), k)
    return min(candidates, key=lambda x: x[1])[0]

# Initialize population
population = [random_individual() for _ in range(POP_SIZE)]

# Genetic algorithm main loop
for gen in range(N_GEN):
    fitness = [evaluate(ind) for ind in population]
    best_fitness = min(fitness)
    print(f"Gen {gen}: best CL/CD = {-best_fitness:.6f}")
    
    # Create next generation
    new_population = []
    
    num_elites = max(1, int(ELITISM_RATIO * POP_SIZE))
    elite_indices = np.argsort(fitness)[:num_elites]
    new_population.extend(population[i] for i in elite_indices)
    
    # Add random individuals every 5 generations for diversity
    if gen % 5 == 0 and gen > 0:
        new_population.extend(random_individual() for _ in range(min(DIVERSITY_NUMBER, POP_SIZE - len(new_population))))
    
    # Fill remaining population with crossover + mutation
    while len(new_population) < POP_SIZE:
        p1 = tournament_selection(population, fitness)
        p2 = tournament_selection(population, fitness)
        child = mutate(crossover(p1, p2))
        new_population.append(child)
    
    population = new_population[:POP_SIZE]

# Display best result
final_fitness = [evaluate(ind) for ind in population]
best_params = population[np.argmin(final_fitness)]
best_airfoil = decode_individual(best_params)

aero = nf.get_aero_from_airfoil(
    airfoil=best_airfoil, alpha=ALPHA, Re=RE, model_size=MODEL_SIZE
)

print("\n=== BEST NACA AIRFOIL ===")
print(f"Parameters: m={best_params[0]:.3f}, p={best_params[1]:.3f}, t={best_params[2]:.3f}")
print(f"CL = {aero['CL'][0]:.4f}, CD = {aero['CD'][0]:.4f}")
print(f"CL/CD = {aero['CL'][0]/aero['CD'][0]:.2f}")

best_airfoil.draw(show=True)
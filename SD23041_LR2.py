import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

POP_SIZE = 300
CHROM_LENGTH = 80
TARGET_ONES = 50
MAX_FITNESS = 80
N_GENERATIONS = 50

TOURNAMENT_K = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 1.0 / CHROM_LENGTH

# Fitness Function
def fitness(individual: np.ndarray) -> float:
    """
    Fitness = MAX_FITNESS - |count(1s) - TARGET_ONES|
    Peaks at 80 when exactly 50 ones are present.
    """
    ones = int(individual.sum())
    return MAX_FITNESS - abs(ones - TARGET_ONES)

# GA Operators
def init_population(pop_size: int, chrom_lenght: int) -> np.array:
    return np.random.randint(0, 2, size=(pop_size, chrom_lenght), dtype=np.int8)

def tournament_selection(population: np.ndarray, fitnesses: np.ndarray, k: int) -> np.ndarray:
    idxs = np.random.randint(0, len(population), size=k)
    best_idx = idxs[np.argmax(fitnesses[idxs])]
    return population[best_idx].copy()

def single_point_crossover(parent1: np.ndarray, parent2: np.ndarray):
    if np.random.rand() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()
    point = np.random.randint(1, CHROM_LENGTH)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent1[:point], parent1[point:]])
    return child1, child2 

def mutate(individual: np.ndarray) -> np.ndarray:
    mask = np.random.rand(CHROM_LENGTH) < MUTATION_RATE
    individual[mask] = 1 - individual[mask]
    return individual

def evolve(population: np.ndarray, generations: int):
    best_fitness_per_gen = []
    best_individual = None
    best_f = -np.inf

    for gen in range(generations):
        fitnesses = np.array([fitness(ind) for ind in population])

        # Track best this generation and overall
        gen_best_idx = np.argmax(fitnesses)
        gen_best_f = fitnesses[gen_best_idx]
        best_fitness_per_gen.append(float(gen_best_f))

        if gen_best_f > best_f:
            best_f = float(gen_best_f)
            best_individual = population[gen_best_idx].copy()

        # Create next generation
        new_population = []
        while len(new_population) < len(population):
            parent1 = tournament_selection(population, fitnesses, TOURNAMENT_K)
            parent2 = tournament_selection(population, fitnesses, TOURNAMENT_K)
            child1, child2 = single_point_crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        population = np.array(new_population[:len(population)], dtype=np.int8)

    return best_individual, best_f, best_fitness_per_gen

# Streamlit UI
st.set_page_config(
    page_title = "GA: 80-bit Pattern (Target 50 ones)",
    page_icon = "🧬",
    layout = "centered"
)

st.title("Genetic Algorithm: Evolving 80-bit Patterns")
st.caption("Population = 300, Length = 80, Target = 50 ones, Max Fitness = 80, Generations = 50")

with st.expander("Problem Requirements (Fixed)", expanded=True):
    st.markdown(f"""
                - **Population Size:** {POP_SIZE}
                - **Chromosome Length:** {CHROM_LENGTH}
                - **Target Number of Ones:** {TARGET_ONES}
                - **Max Fitness:** {MAX_FITNESS}' (achieved when ones = {TARGET_ONES})
                - **Generations:** {N_GENERATIONS}
                - **Selection**: Tournament (k={TOURNAMENT_K})
                - **Crossover**: Single-point, Rate = {CROSSOVER_RATE}
                - **Mutation Rate:** {MUTATION_RATE:.4f}' per bit (~1 flip/chromosome)
                """
                )
    
col1, col2 = st.columns(2)
with col1:
    seed = st.number_input("Random Seed (for reproducibility)", min_value=0, value=123, step=1)
with col2:
    run_button = st.button("Run Genetic Algorithm", type="primary", use_container_width=True)

if run_button:
    with st.spinner("Evolving population across 50 generations..."):
        # Reproducibility
        random.seed(seed)
        np.random.seed(seed)

        # Initialize and evolve
        initial_population = init_population(POP_SIZE, CHROM_LENGTH)
        best_individual, best_fit, fitness_trend = evolve(initial_population, N_GENERATIONS)

        #Analyze best individual
        ones_count = int(best_individual.sum())
        zeros_count = CHROM_LENGTH - ones_count
        bitstring = "".join(map(str, best_individual.tolist()))
        
        # Display results
        st.subheader("Best Individual Evolved")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Best Fitness", f"{best_fit:.0f} / {MAX_FITNESS}")
        with col_b:
            st.metric("Ones Count", f"{ones_count} / {TARGET_ONES}")

        st.write(f"**Zeros**: {zeros_count} / **Total bits**: {CHROM_LENGTH}")
        st.code(bitstring, language="text")

        # Covergence plot
        st.subheader("Fitness Convergence Over Generations")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, N_GENERATIONS + 1), fitness_trend, marker='o', linestyle='-', color='blue')
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness")
        ax.set_title("GA Fitness Convergence (Max = 80)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Success Check
        if best_fit == MAX_FITNESS and ones_count == TARGET_ONES:
            st.success("Perfect match achieved: ones = 50 and fitness = 80 ✅")
        else:
             st.info("GA may reach near-optimal solutions; try another seed to explore further.")

# Footer
st.caption(
    "© 2025 Genetic Algorithm Demo | 80-bit patterns with optimum at 50 ones."
)

    

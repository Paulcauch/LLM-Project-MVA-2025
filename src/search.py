import numpy as np
from tqdm import tqdm

from src.longrope_utils import (initialize_population,
                                evaluate_population,
                                evaluate_individual,
                                mutate,
                                crossover,
                                monotonic_constraint)


def search_lambda_factors(
    model,
    data,
    extension_ratio,
    population_size,
    num_mutations,
    num_crossovers,
    max_iterations,
):
    """
    Evolutionary search to search for lambda as described in the paper

    model: model to be extended.
    data: List of input sequences for evaluation.
    extension_ratio: Ratio of target length to current length s = L'/L.
    population_size: Size of the population for evolutionary search (P).
    num_mutations: Number of mutations per iteration (N1).
    num_crossovers: Number of crossovers per iteration (N2).
    max_iterations: Maximum number of iterations for evolutionary search (\Tau).

    """
    search_space = {
        "lambda_i": (
            1.0,
            extension_ratio * 1.25,
            0.01,
        ),  # Min, max, and step size for lambda_i
        "n_hat": [
            0,
            1,
            2,
            4,
            8,
            12,
            16,
            20,
            24,
            28,
            32,
            64,
            128,
            256,
        ],  # Possible n_hat values
    }

    # Init populations
    population = initialize_population(population_size, search_space, model.d_model)

    for _ in tqdm(range(max_iterations), desc="searching for lambda factors"):
        # Get the perplexities for each configurations (population)
        perplexities = evaluate_population(model, data, population)

        # Select the top-performing individuals as parents
        indices = np.argsort(perplexities)[:population_size//2]
        parents = [population[i] for i in indices]

        # Create new population through mutation and crossover
        mutated = mutate(parents, num_mutations, model.d_model)
        crossed = crossover(parents, num_crossovers, model.d_model)
        population = mutated + crossed  # we combine the two list

        population = [
            monotonic_constraint(individual) for individual in population
        ]

    best_individual = min(population, key=lambda x: evaluate_individual(model, data, x))
    return best_individual["lambda_i"], best_individual["n_hat"]

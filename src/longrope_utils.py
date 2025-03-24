import torch
import random
from src.utils_general import monotonic_constraint, flatten_list
import torch.nn.functional as F


def non_uniform_interpolation(pos_embed, extension_ratio, lambda_factors, n_hat, d_model=256):
    """
    This function implements the two forms of non-uniformities:
    1. Varying RoPE dimensions (lambda_factors)
    2. Token positions (n_hat)

    pos_embed is the positional embedding (what is out of RoPE())
    n_hat is the number of tokens we keep without interpolation
    lambda_factors are the "rescale factors" in the paper
    extension_ratio is s in the paper: L_extended / L_original where L is the input size

    """
    interpolated_pos = pos_embed.clone()

    for i in range(d_model // 2):
        mask = torch.arange(pos_embed.shape[-2], device=pos_embed.device) < n_hat
        # as described in the paper we set to 1 where pos < n_hat, 1/lambda otherwise
        dummy_temp = torch.ones_like(pos_embed[..., 0], device=pos_embed.device)
        scale = torch.where(mask, dummy_temp, 1 / lambda_factors[i] / extension_ratio)

        interpolated_pos[..., 2 * i] *= scale
        interpolated_pos[..., 2 * i + 1] *= scale

    return interpolated_pos


def evaluate_individual(model, data, individual):
    """
    Evaluate an individual lambda factor configuration. We compute the perplexity for a specific
    configuration

    individual (dict): Lambda factor configuration and n_hat.
    """
    lambda_factors, n_hat = individual["lambda_i"], individual["n_hat"]

    # Set the lambda factors and n_hat for the model from the individual configuration since they
    # are directly used when we call the model

    model.lambda_factors["4k"] = lambda_factors
    model.n_hat["4k"] = n_hat

    total_loss = 0
    total_tokens = 0

    model.eval()

    with torch.no_grad():
        for seq in data:
            input_ids = seq.unsqueeze(0)

            output = model(input_ids)
            loss = F.cross_entropy(
                output.view(-1, model.vocab_size)[:-1], seq.view(-1).to(model.device)[1:], reduction="sum"
            )

            total_loss += loss.item()
            total_tokens += seq.numel()

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))

    return perplexity.item()


def evaluate_population(model, data, population):
    """
    Evaluate the population of lambda factor configurations, we test the performance for each
    configuration of n_hat and lambda_factors
    """
    perplexities = []

    for individual in population:
        perplexity = evaluate_individual(model, data, individual)

        perplexities.append(perplexity)

    return perplexities


def initialize_population(population_size, search_space, d_model):
    """
    Initialize the population for evolutionary search.

    as described in the paper, the authors include pi ntk and yarn individuals


    population_size: Number of individuals in the population (P)
    search_space: Dictionary defining the search space for lambda_i and n_hat
    d_model: Dimension of the model (d_k)
    """

    # Initialize population
    population = []

    # Add PI individual
    pi_individual = {
        "lambda_i": [search_space["lambda_i"][1]] * (d_model // 2),
        "n_hat": 0,
    }

    population.append(pi_individual)

    # Add NTK individual
    ntk_individual = {
        "lambda_i": [
            search_space["lambda_i"][1] ** (i / (d_model // 2))
            for i in range(d_model // 2)
        ],
        "n_hat": 0,
    }

    population.append(ntk_individual)

    # Add YaRN individual
    yarn_individual = {
        "lambda_i": [1.0] * (d_model // 6)
        + [
            search_space["lambda_i"][1] ** (i / (d_model // 2))
            for i in range(d_model // 6, d_model // 3)
        ]
        + [search_space["lambda_i"][1]] * (d_model // 2 - d_model // 3),
        "n_hat": 0,
    }

    population.append(yarn_individual)

    for _ in range(population_size):
        individual = {
            "lambda_i": [
                random.uniform(*search_space["lambda_i"][:2]) for _ in range(d_model // 2)
            ],
            "n_hat": random.choice(search_space["n_hat"]),
        }
        population.append(individual)
    return population


def mutate(parents, num_mutations, d_model):
    """
    parents : Parent population.
    num_mutations : Number of mutations to perform (N1).
    d_model : Dimension of the model (d_k).

    """
    mutated_population = []
    for _ in range(num_mutations):
        parent = random.choice(parents)
        child = {"lambda_i": parent["lambda_i"].copy(), "n_hat": parent["n_hat"]}
        for i in range(d_model//2):
            if random.random() < 0.1:
                child["lambda_i"][i] *= random.uniform(0.8, 1.2)

        if random.random() < 0.1:
            child["n_hat"] = random.randint(0, d_model)

        mutated_population.append(child)

    return mutated_population


def crossover(parents, num_crossovers, d_model):
    crossover_population = []
    for _ in range(num_crossovers):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        child = {"lambda_i": parent1["lambda_i"].copy(), "n_hat": parent1["n_hat"]}
        for i in range(d_model//2):
            if random.random() < 0.5:
                child["lambda_i"][i] = parent2["lambda_i"][i]

        if random.random() < 0.5:
            child["n_hat"] = parent2["n_hat"]

        crossover_population.append(child)

    return crossover_population

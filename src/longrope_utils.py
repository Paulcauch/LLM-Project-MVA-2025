import torch
import random


def non_uniform_interpolation(pos_embed, extension_ratio, lambda_factors, n_hat):
    """
    This function implements the two forms of non-uniformities:
    1. Varying RoPE dimensions (lambda_factors)
    2. Token positions (n_hat)

    pos_embed is the positional embedding (what is out of RoPE())
    n_hat is the number of tokens we keep without interpolation
    lambda_factors are the "rescale factors" in the paper
    extension_ratio is s in the paper: L_extended / L_original where L is the input size

    """
    d_model = pos_embed.shape[-1]
    interpolated_pos = pos_embed.clone()

    for i in range(d_model // 2):
        mask = torch.arange(pos_embed.shape[-2], device=pos_embed.device) < n_hat
        # as described in the paper we set to 1 where pos < n_hat, 1/lambda otherwise
        dummy_temp = torch.ones_like(pos_embed[..., 0], device=pos_embed.device)
        scale = torch.where(mask, dummy_temp, 1 / lambda_factors[i] / extension_ratio)

        interpolated_pos[..., 2 * i] *= scale
        interpolated_pos[..., 2 * i + 1] *= scale

    return interpolated_pos


def progressive_extension(
    model,
    data,
    base_length,
    target_length,
    population_size,
    num_mutations,
    num_crossovers,
    max_iterations,
):
    """
    Extend the context with a non fine tuning phase, then a fine tuning phase

    model : Model to be extended.
    data : List of tensors containing input ids
    base_length : Original context window length of the model.
    target_length : Target context window length
    population_size : Size of the population for evolutionary search (P in the paper).
    num_mutations : Number of mutations per iteration in the search (N1 in the paper).
    num_crossovers : Number of crossovers per iteration in the search (N2 in the paper).
    max_iterations : Maximum number of iterations for evolutionary search (\Tau in the paper).
    """
    curr_model = model

    lambda_factors_up, n_hat_up = search_lambda_factors(
        curr_model,
        data,
        128000 / base_length,
        population_size,
        num_mutations,
        num_crossovers,
        max_iterations,
    )

    curr_model = fine_tune(
        curr_model, data, 128000, lambda_factors_up, n_hat_up, steps=400
    )

    curr_model.lambda_factors["128k"] = lambda_factors_up
    curr_model.n_hat["128k"] = n_hat_up

    return (curr_model, lambda_factors_up, n_hat_up,)


def evaluate_individual(model, data, individual):
    """
    Evaluate an individual lambda factor configuration. We compute the perplexity for a specific configuration

    individual (dict): Lambda factor configuration and n_hat.
    """
    lambda_factors, n_hat = individual["lambda_i"], individual["n_hat"]

    # Set the lambda factors and n_hat for the model from the individual configuration since they are
    # directly used when we call the model

    model.lambda_factors = lambda_factors
    model.n_hat = n_hat

    total_loss = 0
    total_tokens = 0

    model.eval()

    with torch.no_grad():
        for seq in data:
            input_ids = seq.unsqueeze(0)

            output = model(input_ids)
            
            loss = F.cross_entropy(
                output.view(-1, model.vocab_size), seq.view(-1), reduction="sum"
            )

            total_loss += loss.item()
            total_tokens += seq.numel()

    perplexity = torch.exp(total_loss / total_tokens)
    
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

    This function implements the optimized initial population generation described in Section 3.2,
    including PI, NTK, and YaRN as initial individuals.

    Args:
        population_size: Number of individuals in the population
        search_space: Dictionary defining the search space for lambda_i and n_hat
        d_model: Dimension of the model

    Returns:
        population: List of individuals, each represented as a dictionary
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

    # Generate the rest of the population randomly
    for _ in range(population_size):
        individual = {
            "lambda_i": [
                random.uniform(*search_space["lambda_i"]) for _ in range(d_model // 2)
            ],
            "n_hat": random.choice(search_space["n_hat"]),
        }
        population.append(apply_monotonic_constraint(individual))
    return population

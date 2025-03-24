from src.search import search_lambda_factors
from src.finetune import fine_tune


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
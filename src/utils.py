def monotonic_constraint(individual):
    """
    Just to assert lambda_i+1 >= lambda_i
    """
    lambda_i = individual["lambda_i"]
    for i in range(1, len(lambda_i)):
        lambda_i[i] = max(lambda_i[i], lambda_i[i - 1])
    return individual

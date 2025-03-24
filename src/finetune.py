import torch
import torch.nn.functional as F
import random
from tqdm import tqdm


def evaluate_perplexity(model, data, target_length):
    """
    evaluate the perplexity of a model, everything is classic in this function
    """
    total_loss = 0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for seq, _ in data:
            seq_len = seq.size(0)
            if seq_len <= target_length:
                input_ids = seq.unsqueeze(0)
            else:
                start_idx = random.randint(0, seq_len - target_length)
                input_ids = seq[start_idx: start_idx + target_length].unsqueeze(0)
                # we truncate the input if it is too long
            output = model(input_ids)
            loss = F.cross_entropy(
                output.view(-1, model.vocab_size)[:-1], input_ids.view(-1).to(model.device)[1:], reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += input_ids.numel()
    return torch.exp(torch.tensor(total_loss / total_tokens))


def fine_tune(model, train_data, val_data, target_length, lambda_factors, n_hat, steps):
    """
    Fine-tune the LongRoPE model.

    Args:
        model (nn.Module): LongRoPE model.
        train_data (list): List of input sequences for training.
        val_data (list): List of input sequences for validation.
        target_length (int): Target context window length.
        lambda_factors (list): Lambda factors for interpolation.
        n_hat (int): Threshold for applying interpolation.
        steps (int): Number of fine-tuning steps, as specified in the paper.

    Returns:
        nn.Module: Fine-tuned LongRoPE model.
    """
    model.lambda_factors[f"{target_length // 1000}k"] = lambda_factors
    model.n_hat[f"{target_length // 1000}k"] = n_hat
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_perplexity = float("inf")
    best_model_state = None

    for step in tqdm(range(steps), desc="fine tuning step"):
        # Training
        model.train()
        optimizer.zero_grad()
        seq = random.choice(train_data)[0]
        seq_len = seq.size(0)
        if seq_len <= target_length:
            input_ids = seq.unsqueeze(0)
        else:
            start_idx = random.randint(0, seq_len - target_length)
            input_ids = seq[start_idx: start_idx + target_length].unsqueeze(0)
        output = model(input_ids)
        loss = F.cross_entropy(output.view(-1, model.vocab_size)[:-1],
                               input_ids.view(-1).to(model.device)[1:])
        loss.backward()
        optimizer.step()

        # Validation (every 50 steps)
        if step % 50 == 0:
            model.eval()
            val_perplexity = evaluate_perplexity(model, val_data, target_length)
            print(f"Step {step}, Validation Perplexity: {val_perplexity}")
            if val_perplexity < best_val_perplexity:
                best_val_perplexity = val_perplexity
                best_model_state = model.state_dict()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model

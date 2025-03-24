import torch
import torch.nn as nn
from .rope import RoPEPositionalEncoding
from .longrope_utils import non_uniform_interpolation
from .extension import progressive_extension
from .utils_general import check_valid_ids, truncate_ids


class LongRoPEModel(nn.Module):
    """
    d_model : Dimension of the model (usually denoted as d_k).
    n_heads : Number of attention heads.
    num_layers : Number of transformer layers.
    vocab_size : Size of the vocabulary.
    max_len : Original context window length of the model.
    """

    def __init__(self, d_model, n_heads, num_layers, vocab_size, max_len, device="cuda"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rope = RoPEPositionalEncoding(d_model, max_len)
        self.device = device
        self.transformers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
                for _ in range(num_layers)
            ]
        )  # here we take a vanilla transformers but actually we can just replace it by
           # a trained model
        self.out = nn.Linear(d_model, vocab_size)

        self.lambda_factors = {
            "4k": None,
            "8k": None,
            "128k": None,
            "256k": None,
            "2048k": None,
        }

        self.n_hat = {"4k": None, "8k": None, "128k": None, "256k": None, "2048k": None}

        self.lambda_factors_base = None
        self.n_hat_base = 0

        self.extension_ratio = None
        self.base_context_length = max_len

        self.to(device)

    def forward(self, input_ids):
        input_ids = input_ids.to(self.device)
        input_embeddings = self.embedding(input_ids)  # convert the embeddings to vectors
        seq_length = input_ids.size(1)  # n tokens
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        pos_embeddings = self.rope(positions)  # get the embeddings of rope
    
        pos_embeddings = self.apply_interpolation(pos_embeddings, "4k")

        if seq_length > self.base_context_length and False:
            # If seq_len > max_len we truncate the data
            pos_embeddings = pos_embeddings[:, : self.base_context_length, :]
            input_embeddings = input_embeddings[:, : self.base_context_length, :]
            seq_length = self.base_context_length

        pos_embeddings = pos_embeddings[:, :seq_length, : self.d_model]  # just to make sure..

        embeddings = input_embeddings + pos_embeddings
        for transformer in self.transformers:
            embeddings = transformer(embeddings)
        
        embeddings = self.out(embeddings)

        return embeddings

    def only_embeddings(self, input_ids):
        input_ids = input_ids.to(self.device)
        input_embeddings = self.embedding(input_ids)
        seq_length = input_ids.size(1)
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        pos_embeddings = self.rope(positions)
    
        pos_embeddings = self.apply_interpolation(pos_embeddings, "4k")

        if seq_length > self.base_context_length and False: # we disable this part
            # If seq_len > max_len we truncate the data
            pos_embeddings = pos_embeddings[:, : self.base_context_length, :]
            input_embeddings = input_embeddings[:, : self.base_context_length, :]
            seq_length = self.base_context_length

        pos_embeddings = pos_embeddings[:, :seq_length, : self.d_model]  # just to make sure..

        embeddings = input_embeddings + pos_embeddings

        return embeddings, (input_embeddings, pos_embeddings)

    def apply_interpolation(self, pos_embed, context_length):
        """Apply non-uniform interpolation to position embeddings."""
        return non_uniform_interpolation(
            pos_embed,
            self.extension_ratio,
            self.lambda_factors[context_length],
            self.n_hat[context_length],
            self.d_model
        )

    def extend_context(
        self,
        dataset,
        target_length,
        max_sequence_length,
        tokenizer,
        population_size,
        num_mutations,
        num_crossovers,
        max_iterations,
    ):
        """
        dataset : tensor dataset, contains the ids of the tokenizer make sure max_length is correct.
        target_length : Target context window length.
        max_sequence_length : Maximum sequence length for input data.
        tokenizer: Tokenizer object for encoding input data.
        population_size : Size of the population for evolutionary search.
        num_mutations : Number of mutations per iteration.
        num_crossovers : Number of crossovers per iteration.
        max_iterations : Maximum number of iterations for evolutionary search.
        """
        self.extension_ratio = target_length / self.rope.max_len
        (
            model,
            lambda_factors,
            n_hat,
        ) = progressive_extension(
            self,
            dataset,
            self.rope.max_len,
            target_length,
            population_size,
            num_mutations,
            num_crossovers,
            max_iterations,
        )

        self.lambda_factors["4k"] = lambda_factors
        #self.lambda_factors_base = lambda_factors_base
        self.n_hat["4k"] = n_hat
        #self.n_hat_base = n_hat_base

        return model

from collections import namedtuple
import numpy as np


Beam = namedtuple(
    "Beam",
    (
        "sequence",
        "log_prob",
    ),
)


def beam_search_decode(
    log_probs: np.ndarray,
    *,
    beam_width: int,
):

    # Get sequence length and number of tokens
    seq_len, num_tokens = log_probs.shape

    # Initialize beams
    beams = [
        Beam(
            sequence=[],
            log_prob=0.0,
        ),
    ]

    for t in range(seq_len):

        # Initialize new beams
        new_beams = []

        for beam in beams:
            for token_index in range(num_tokens):
                new_beam = Beam(
                    sequence=beam.sequence + [token_index],
                    log_prob=beam.log_prob + log_probs[t, token_index],
                )
                new_beams.append(new_beam)

        # Sort new beams by log-probability in descending order
        new_beams.sort(key=lambda x: x.log_prob, reverse=True)

        # Update beams
        # Only keep the top several beams
        beams = new_beams[:beam_width]

    # Get the best beam
    best_beam = beams[0]

    return best_beam


def decode_token_indices(
    token_indices: list[int],
    *,
    tokens: list[str],
    blank_token_index: int = 0,
) -> str:

    # Initialize the token indices to predict
    pred_token_indices = []

    # Last token index
    last_token_index = None

    for index in token_indices:

        if index == last_token_index or index == blank_token_index:
            continue

        # Add the predicted token index
        pred_token_indices.append(index)

        # Update the last token index
        last_token_index = index

    # Map each index to token
    pred_tokens = list(
        map(
            lambda index: tokens[index],
            pred_token_indices,
        )
    )

    # The predicted label
    label = "".join(pred_tokens)

    return label

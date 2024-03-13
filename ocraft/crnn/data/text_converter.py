from typing import Self
import torch
from torch import Tensor


class TextConverter:

    BLANK_TOKEN = "<blank>"
    BLANK_TOKEN_ENCODED_VALUE = 0

    def __init__(
        self,
        tokens: list[str],
        *,
        blank_token_repr: str = "-",
    ) -> None:

        self._token_to_encoded_value: dict[str, int] = {}

        # Add blank token
        self._token_to_encoded_value[self.BLANK_TOKEN] = self.BLANK_TOKEN_ENCODED_VALUE

        # Assign the rest of the tokens with encoded values
        for i, token in enumerate(tokens):
            value = i + 1
            self._token_to_encoded_value[token] = value

    @classmethod
    def from_token_str(
        cls,
        token_str: str,
        *,
        blank_token_repr: str = "-",
    ) -> Self:

        # Get tokens from the token string
        tokens = list(token_str)

        return cls(tokens, blank_token_repr=blank_token_repr)

    def encode(self, text: str) -> Tensor:

        # Encoded values of all tokens in the text
        encoded_values = []
        for token in text:
            value = self._token_to_encoded_value[token]
            encoded_values.append(value)

        # Convert to tensor
        encoded_text = torch.LongTensor(encoded_values)

        return encoded_text

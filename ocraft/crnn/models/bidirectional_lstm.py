from torch import Tensor
from torch import nn


class BidirectionalLSTM(nn.Module):

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
        )

        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, sequence: Tensor) -> Tensor:
        """Forward propagation.

        Parameters
        ----------
        sequence : Tensor
            Shape: (L, N, input_size)

        Returns
        -------
        Tensor
            Shape: (L, N, output_size)
        """

        # Input sequence has shape (L, N, input_size)

        # (L, N, hidden_size * 2)
        lstm_output: Tensor
        lstm_output, _ = self.lstm(sequence)

        seq_len, batch_size, double_hidden_size = lstm_output.shape

        # (L * N, hidden_size * 2)
        x = lstm_output.view(seq_len * batch_size, double_hidden_size)

        # (L * N, output_size)
        output: Tensor = self.embedding(x)

        # (L, N, output_size)
        output = output.view(seq_len, batch_size, -1)

        return output

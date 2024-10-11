from typing import override

from torch import nn, Tensor
from torchvision.models.video import R3D_18_Weights, r3d_18  # type: ignore


class ResNet3D_LSTM(nn.Module):
    """
    Initializes the ResNet3D-18 + LSTM model.

    Args:
        lstm_hidden_size (int): The number of features in the hidden state of LSTM.
        num_classes (int): The number of output classes.
        num_layers (int): Number of LSTM layers.
    """

    def __init__(self, num_classes: int, lstm_hidden_size: int, num_layers: int = 2):
        super(ResNet3D_LSTM, self).__init__()
        self.resnet3d = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.resnet3d.fc = nn.Linear(self.resnet3d.fc.in_features, 512)
        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, channels, depth, height, width).

        Returns:
            torch.Tensor: Output predictions for each class.
        """
        batch_size, seq_length, c, d, h, w = x.size()

        # Reshape input for ResNet3D
        x = x.view(batch_size * seq_length, c, d, h, w)

        # Feature extraction via ResNet3D
        x = self.resnet3d(x)

        # Reshape features for LSTM input
        x = x.view(batch_size, seq_length, -1)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last time step

        # Final classification layer
        out = self.fc(lstm_out)
        return out

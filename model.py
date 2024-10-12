from typing import override

import torch
from torch import Tensor, nn
from torchvision.models.video import R3D_18_Weights, r3d_18  # type: ignore


class ResNet3D_LSTM(nn.Module):
    """
    Initializes the ResNet3D-18 + LSTM model.

    Args:
        lstm_hidden_size (int): The number of features in the hidden state of LSTM.
        num_classes (int): The number of output classes.
        num_layers (int): Number of LSTM layers.
    """

    def __init__(self, num_classes: int, lstm_hidden_size: int, num_layers: int = 1):
        super(ResNet3D_LSTM, self).__init__()
        self.num_layers = num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.resnet3d = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.resnet3d.fc = nn.Linear(self.resnet3d.fc.in_features, 512)
        for param in self.resnet3d.parameters():
            param.requires_grad = False
        for param in self.resnet3d.fc.parameters():
            param.requires_grad = True
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
        # print(f'x size: {x.size()}')

        # Feature extraction via ResNet3D
        x = self.resnet3d(x)
        # print(f'output of resnet: {x.shape}')
        x = x.view(1, 1, x.size(1))

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.lstm_hidden_size).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.lstm_hidden_size).requires_grad_()

        out, (_, _) = self.lstm(x, (h0.detach(), c0.detach()))

        # print(f'LSTM forward pass output: {out.shape}')

        # Take the output of the last time step for each batch
        out = self.fc(out[:, -1, :])

        return out

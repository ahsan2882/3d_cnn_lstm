import argparse
import time
from pathlib import Path
from typing import List

import torch
from torch.optim import Adam, lr_scheduler  # type: ignore
from torchinfo import summary  # type: ignore

from dataloaders import create_dataloaders
from engine import plot_curves, train_step, val_step
from model import ResNet3D_LSTM
from utils import load_model, save_model

NUM_EPOCHS = 100
BATCH_SIZE = 8


def main(source: str, filename: str) -> None:
    """
    Main function to train the model.

    Args:
        source (str): Path to the dataset folder.
        filename (str): Filename for saving the model.
    """
    device: torch.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    video_dir = Path(source).resolve()

    _, train_dataloader, _, val_dataloader, _, _, class_names = create_dataloaders(
        video_dir, BATCH_SIZE)
    model = ResNet3D_LSTM(num_classes=len(class_names),
                          lstm_hidden_size=128, num_layers=1)

    model = load_model(model, Path(__file__).parent.resolve() /
                       'model_data' / f'{filename}.pth').to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3)

    best_train_acc = 0.0
    best_train_loss = float('inf')
    best_val_acc = 0.0
    best_val_loss = float('inf')

    train_losses: List[float] = []
    train_accuracies: List[float] = []
    val_losses: List[float] = []
    val_accuracies: List[float] = []

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        if epoch == 0:
            summary(model, input_size=(1, 128, 3, 112, 112), col_names=[
                    "input_size", "output_size", "num_params", "params_percent", "kernel_size", "trainable"])

        start_time = time.time()
        train_loss, train_acc = train_step(
            model, train_dataloader, loss_fn, optimizer, device, epoch)
        train_losses.append(train_loss)
        torch.cuda.empty_cache()

        # Validation phase
        val_loss, val_acc = val_step(
            model, val_dataloader, loss_fn, device, epoch)
        val_losses.append(val_loss)

        # Update scheduler and cache
        scheduler.step(val_loss)
        torch.cuda.empty_cache()

        # Print training and validation stats
        elapsed_time = time.time() - start_time
        remaining_time = (NUM_EPOCHS - epoch - 1) * elapsed_time
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%,Val Loss: {
              val_loss:.4f}, Val Acc: {val_acc:.2f}%, learning rate: {scheduler.get_last_lr()}, Estimated Remaining Time: {remaining_time // 60:.0f} min {remaining_time % 60:.0f} sec')

        # Save model if metrics improved
        if (train_loss < best_train_loss and
            train_acc > best_train_acc and
            val_loss < best_val_loss and
                val_acc > best_val_acc):
            best_train_acc = train_acc
            best_train_loss = train_loss
            best_val_acc = val_acc
            best_val_loss = val_loss

            print(f"Saving model... (Train Loss: {train_loss:.4f}, Train Acc: {
                  train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%)")
            save_model(model, Path(__file__).parent.resolve() /
                       'model_data', f'{filename}.pth')

    plot_curves(train_losses, val_losses,
                train_accuracies, val_accuracies, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a 3D CNN model on video data.')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to the dataset folder.')
    parser.add_argument('--filename', type=str, required=True,
                        help='Filename for saving model')
    args = parser.parse_args()

    main(args.source, args.filename)

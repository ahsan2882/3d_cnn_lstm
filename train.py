import argparse
import gc
import time
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataloaders import create_dataloaders
from engine import plot_curves, train_step, val_step
from model import ResNet3D_LSTM
from utils import save_model

NUM_EPOCHS = 100
BATCH_SIZE = 1
PATIENCE = 30


def play_video(video_frames: List[np.ndarray[Any,
                                             np.dtype[np.integer[Any] | np.floating[Any]]]], label: str):
    for frame in video_frames:
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Video', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def main(source: str, filename: str):
    device: torch.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    video_dir = Path(source).resolve()

    _, train_dataloader, _, val_dataloader, _, _, class_names = create_dataloaders(
        video_dir, BATCH_SIZE)
    model = ResNet3D_LSTM(num_classes=len(class_names),
                          lstm_hidden_size=128).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(  # type: ignore
        model.parameters(),
        lr=0.01)
    scheduler = ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3)

    best_train_acc = 0.0
    best_train_loss = float('inf')
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    train_losses: List[float] = []
    train_accuracies: List[float] = []
    val_losses: List[float] = []
    val_accuracies: List[float] = []

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        start_time = time.time()
        train_loss, train_acc = train_step(
            model, train_dataloader, loss_fn, optimizer, device, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            gc.collect()
        # Validation phase
        val_loss, val_acc = val_step(model, val_dataloader, loss_fn, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        scheduler.step(val_loss)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            gc.collect()
        # Print training and validation stats
        elapsed_time = time.time() - start_time
        remaining_time = (NUM_EPOCHS - epoch - 1) * elapsed_time
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%,Val Loss: {
              val_loss:.4f}, Val Acc: {val_acc:.2f}%, Estimated Remaining Time: {remaining_time // 60:.0f} min {remaining_time % 60:.0f} sec')

        # Early stopping
        if val_loss < best_val_loss:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered!")
                break
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
            save_model(model, Path(Path(__file__).parent.resolve(),
                       'model_data'), f'{filename}.pth')

    plot_curves(train_losses, val_losses,
                train_accuracies, val_accuracies, filename)
    save_model(model, Path(Path(__file__).parent.resolve(),
               'model_data'), f'{filename}.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a 3D CNN model on video data.')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to the dataset folder.')
    parser.add_argument('--filename', type=str, required=True,
                        help='filename for saving model')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args.source, args.filename)

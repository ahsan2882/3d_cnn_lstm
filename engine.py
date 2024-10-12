from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch.optim import Optimizer  # type: ignore
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_packed_sequence


def train_step(model: torch.nn.Module,
               dataloader: DataLoader[Tuple[torch.Tensor, int]],
               loss_fn: torch.nn.Module,
               optimizer: Optimizer,
               device: torch.device,
               epoch: int) -> Tuple[float, float]:
    """
    Performs a single training step, aggregating predictions over all windows.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for training data.
        loss_fn (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run the model on.
        epoch (int): Current epoch number.

    Returns:
        Tuple[float, float]: Training loss and accuracy.
    """
    model.train()
    model.to(device)

    train_loss, train_acc = 0.0, 0.0
    video_predictions: Dict[int, Dict[str, List[torch.Tensor]]] = {}

    for windows, labels in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}", unit="batch"):
        windows, labels = windows.to(device), labels.to(device)
        optimizer.zero_grad()
        padded_videos, lengths = pad_packed_sequence(
            windows.cpu(), batch_first=True)
        print(lengths)
        padded_videos = padded_videos.to(device)
        lengths = lengths.to(device)
        # padded_videos = padded_videos.permute(0, 2, 1, 3, 4)
        outputs = model(padded_videos)
        for i in range(padded_videos.size(0)):
            # Use the label as video ID for aggregation
            video_id = labels[i].item()
            if video_id not in video_predictions:
                video_predictions[video_id] = {'outputs': [], 'labels': []}
            video_predictions[video_id]['outputs'].append(outputs[i])
            video_predictions[video_id]['labels'].append(labels[i])

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(dataloader)
    for video in video_predictions.values():
        video_outputs = torch.stack(video['outputs'])
        video_labels = torch.tensor(video['labels'], device=device)

        # Aggregate predictions
        predicted_labels = torch.argmax(
            torch.softmax(video_outputs, dim=1), dim=1)
        train_acc += (predicted_labels ==
                      video_labels).sum().item() / len(video_labels)

    train_acc /= len(video_predictions)  # Average over all videos
    return train_loss, train_acc * 100


def val_step(model: torch.nn.Module,
             dataloader: DataLoader[Tuple[torch.Tensor, int]],
             loss_fn: torch.nn.Module,
             device: torch.device, epoch: int) -> Tuple[float, float]:
    """
    Performs a validation step, aggregating predictions over all windows.

    Args:
        model (nn.Module): The model to validate.
        dataloader (DataLoader): DataLoader for validation data.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device to run the model on.

    Returns:
        Tuple[float, float]: Validation loss and accuracy.
    """
    model.to(device)
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    video_predictions: Dict[int, Dict[str, List[torch.Tensor]]] = {}
    with torch.inference_mode():
        for windows, labels in tqdm(dataloader, desc=f"Validation Epoch {epoch + 1}", unit="batch"):
            windows, labels = windows.to(device), labels.to(device)
            padded_videos, lengths = pad_packed_sequence(
                windows.cpu(), batch_first=True)
            print(lengths)
            padded_videos = padded_videos.to(device)
            lengths = lengths.to(device)
            # padded_videos = padded_videos.permute(0, 2, 1, 3, 4)
            outputs = model(padded_videos)
            for i in range(padded_videos.size(0)):
                # Use the label as video ID for aggregation
                video_id = labels[i].item()
                if video_id not in video_predictions:
                    video_predictions[video_id] = {'outputs': [], 'labels': []}
                video_predictions[video_id]['outputs'].append(outputs[i])
                video_predictions[video_id]['labels'].append(labels[i])
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(dataloader)
    for video in video_predictions.values():
        video_outputs = torch.stack(video['outputs'])
        video_labels = torch.tensor(video['labels'], device=device)

        # Aggregate predictions
        predicted_labels = torch.argmax(
            torch.softmax(video_outputs, dim=1), dim=1)
        val_acc += (predicted_labels == video_labels).sum().item() / \
            len(video_labels)

    val_acc /= len(video_predictions)  # Average over all videos
    return val_loss, val_acc * 100  # Return percentage


def plot_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float], filename: str
) -> None:
    """
    Plots the training and validation loss and accuracy curves.

    Args:
        train_losses (List[float]): List of training losses.
        val_losses (List[float]): List of validation losses.
        train_accuracies (List[float]): List of training accuracies.
        val_accuracies (List[float]): List of validation accuracies.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    # You can specify the desired file format (e.g., .png, .jpg, .pdf)
    plt.savefig(f'{filename}_curves.png')

    plt.close()

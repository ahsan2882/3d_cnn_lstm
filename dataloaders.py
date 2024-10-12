import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, override

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms  # type: ignore
from tqdm import tqdm


class VideoTransform:
    def __init__(self, size: Tuple[int, int]):  # type: ignore
        self.resize = transforms.Resize(size)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, video_frames: torch.Tensor) -> torch.Tensor:
        # Assume video_frames is of shape (C, T, H, W)
        # Resize frames
        resized_frames = self.resize(video_frames.permute(1, 0, 2, 3))
        resized_frames = resized_frames.permute(
            1, 0, 2, 3)  # Change back to (C, T, H, W)
        # Normalize frames to [-1, 1] for PyTorch models
        normalized_frames = (resized_frames - 0.5) / 0.5
        return normalized_frames


class CustomVideoDataset(Dataset[Tuple[torch.Tensor, int]]):
    def __init__(self, video_dir: str, transform: Optional[VideoTransform] = None,  # type: ignore
                 window_size: int = 128, stride: int = 32):
        """
        Custom dataset to handle video windows.

        Args:
            video_dir (str): Path to the dataset directory with class folders.
            transform (VideoTransform, optional): Transformations to be applied. Defaults to None.
            window_size (int, optional): Number of frames per window. Defaults to 128.
            stride (int, optional): Step size for sliding window. Defaults to 32.
            mode (str, optional): Mode of dataset (train/validate/test). Defaults to "train".
        """
        self.video_dir: str = video_dir
        self.transform: VideoTransform | None = transform
        self.window_size: int = window_size
        self.stride: int = stride
        self.video_paths, self.labels, self.classes = self._prepare_dataset()

    def _prepare_dataset(self) -> Tuple[List[str], List[int], List[str]]:
        video_paths: List[str] = []
        labels: List[int] = []
        classes: List[str] = []
        class_folders: List[str] = os.listdir(self.video_dir)

        for label, class_folder in tqdm(enumerate(class_folders), desc="Preparing dataset", unit="class"):
            class_path: str = os.path.join(self.video_dir, class_folder)
            if os.path.isdir(class_path):
                classes.append(class_folder)
                for video_file in os.listdir(class_path):
                    video_paths.append(os.path.join(class_path, video_file))
                    labels.append(label)
        return video_paths, labels, classes

    def _get_video_frames(self, video_path: str):
        """Extract frames from a video file."""
        cap = cv2.VideoCapture(video_path)
        frames: List[np.ndarray[Any,
                                np.dtype[np.integer[Any] | np.floating[Any]]]] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def _extract_windows(self,
                         frames: List[np.ndarray[Any,
                                                 np.dtype[np.integer[Any] | np.floating[Any]]]]
                         ) -> List[List[np.ndarray[Any,
                                                   np.dtype[np.integer[Any] | np.floating[Any]]]]]:
        """Extract sliding windows from video frames."""
        windows: List[List[np.ndarray[Any,
                                      np.dtype[np.integer[Any] | np.floating[Any]]]]] = []
        total_frames = len(frames)
        for start in range(0, total_frames-self.window_size+1, self.stride):
            windows.append(frames[start:start+self.window_size])
        return windows

    def __len__(self) -> int:
        return len(self.video_paths)

    @override
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = self._get_video_frames(video_path)

        while len(frames) < self.window_size:
            frames.append(np.zeros_like(frames[0]))
        windows = self._extract_windows(frames)

        # Randomly select a single window (or loop over windows if desired)
        window_idx = np.random.randint(len(windows))
        selected_window = windows[window_idx]

        # Convert the selected window to tensor
        window_tensor = torch.tensor(np.array(selected_window)).permute(
            3, 0, 1, 2)  # (C, T, H, W)

        if self.transform:
            window_tensor = self.transform(window_tensor)

        return window_tensor, label


def create_dataloaders(video_dir: str | Path,
                       batch_size: int = 32,
                       window_size: int = 128,
                       stride: int = 32
                       ) -> Tuple[
                           CustomVideoDataset, DataLoader[Tuple[torch.Tensor, int]],
                           CustomVideoDataset, DataLoader[Tuple[torch.Tensor, int]],
                           CustomVideoDataset, DataLoader[Tuple[torch.Tensor, int]], List[str]]:
    """Creates dataloaders for train, validate, and test datasets."""
    # Define transformations
    video_transform = VideoTransform(size=(112, 112))
    train_dataset = CustomVideoDataset(os.path.join(
        video_dir, 'train'), transform=video_transform, window_size=window_size, stride=stride)
    validate_dataset = CustomVideoDataset(os.path.join(
        video_dir, 'validate'), transform=video_transform, window_size=window_size, stride=stride)
    test_dataset = CustomVideoDataset(os.path.join(
        video_dir, 'test'), transform=video_transform, window_size=window_size, stride=stride)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    validate_dataloader = DataLoader(
        validate_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    return train_dataset, train_dataloader, validate_dataset, validate_dataloader, test_dataset, test_dataloader, train_dataset.classes

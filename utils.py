import os
from pathlib import Path

import torch


def save_model(model: torch.nn.Module,
               target_dir: Path,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="model.pth")
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)


def load_model(model: torch.nn.Module, model_path: str | Path) -> torch.nn.Module:
    """
    load_model: Loads the model params from the given path

    Args:
        model (torch.nn.Module): initialized model to copy saved parameters to
        model_path (str): path where model params are stored

    Returns:
        torch.nn.Module: updated model with saved params
    """
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    return model

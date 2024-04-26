import torch


def get_device():
    """
    Get available device for the calculations.
    """
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"

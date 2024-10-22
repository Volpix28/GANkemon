import os
import torch

import numpy as np


def set_seeds(seed=42):
    """
    Set the seed for the utilized frameworks and libraries for reproducability.
    :param seed:
    :return:
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

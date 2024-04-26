def count_parameters(model):
    """
    Count parameters of Pytorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

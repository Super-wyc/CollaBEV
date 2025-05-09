import torch


def to_device(obj, device):
    """递归地将所有张量移动到指定设备"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(to_device(x, device) for x in obj)
    else:
        return obj
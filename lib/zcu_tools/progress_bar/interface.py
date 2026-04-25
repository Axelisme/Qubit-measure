from .base import BaseProgressBar


def make_pbar(*args, **kwargs) -> BaseProgressBar:
    from .backend.tqdm import TQDMProgressBar

    return TQDMProgressBar(*args, **kwargs)

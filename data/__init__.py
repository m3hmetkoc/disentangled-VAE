"""
Data module for Moving MNIST dataset.
"""

from .moving_mnist import (
    MovingMNIST,
    MovingMNISTWithLabels,
    get_moving_mnist_dataloaders
)

__all__ = [
    'MovingMNIST',
    'MovingMNISTWithLabels',
    'get_moving_mnist_dataloaders'
]


"""
Disentangled Content-Motion VAE for Video Generation

A research project implementing and comparing three VAE architectures:
1. BaselineVAE - Simple VAE without temporal modeling
2. LSTMVAE - VAE with LSTM for temporal dependencies  
3. DisentangledVAE - VAE with explicit content/motion separation (main contribution)

This project demonstrates that explicit architectural separation of content
and motion provides better controllability for video generation.
"""

__version__ = '1.0.0'
__author__ = 'Research Project'


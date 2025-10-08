import yaml
import random
from collections import deque
import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt

def load_hyperparameters(file_path="hyperparameters.yaml"):
    with open(file_path, 'r') as file:
        hyperparameters = yaml.safe_load(file)
    return hyperparameters

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, policy, value):
        self.buffer.append((state, policy, value))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def board_to_tensor(board):
    """
    Convert a chess.Board object to a 12x8x8 tensor.
    6 piece types for each color -> 12 channels.
    """
    piece_map = board.piece_map()
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    # Map piece symbols to channels
    piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,   # white
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # black
    }

    for square, piece in piece_map.items():
        row = 7 - (square // 8)  # 0-indexed from top
        col = square % 8
        channel = piece_to_channel[piece.symbol()]
        tensor[channel, row, col] = 1.0

    return torch.tensor(tensor)

def save_stats(stats, file_path="stats/training_stats.json"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(stats, f)

def plot_stats(stats):
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(stats['avg_losses'], label='Average Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Average Total Loss')
    plt.legend()
    plt.savefig("plots/average_loss.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(stats['avg_policy_losses'], label='Average Policy Loss', color='orange')
    plt.xlabel('Iteration')
    plt.ylabel('Policy Loss')
    plt.title('Average Policy Loss')
    plt.legend()
    plt.savefig("plots/average_policy_loss.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(stats['avg_value_losses'], label='Average Value Loss', color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Value Loss')
    plt.title('Average Value Loss')
    plt.legend()
    plt.savefig("plots/average_value_loss.png")
    plt.close()


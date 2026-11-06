import math
import torch


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    # TODO: return a (seq_len, d_model) tensor of sinusoidal positional encodings
    positions = torch.arange(seq_len)
    i = torch.arange(0, d_model, 2)
    frequencies = torch.exp( - torch.log(torch.tensor(10000)) * 2 * i / d_model)
    angles = positions.reshape(-1, 1) * frequencies.reshape(1, -1)

    position_encodings = torch.zeros((seq_len, d_model))
    position_encodings[:, 0::2] = torch.sin(angles)
    position_encodings[:, 1::2] = torch.cos(angles[:, :d_model // 2])

    return position_encodings

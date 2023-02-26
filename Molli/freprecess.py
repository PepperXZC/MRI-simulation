import torch
import math
import matrix_rot
import utils

device = utils.device

# 这里 t 是一个数，不是矩阵


def res(t, T1, T2, df):
    phi = 2*math.pi*df*t/1000
    E1 = math.exp(-t/T1)
    E2 = math.exp(-t/T2)
    z = matrix_rot.zrot(phi)
    A = torch.Tensor([
        [E2, 0, 0],
        [0, E2, 0],
        [0, 0, E1]
    ]).to(device) @ z
    B = torch.Tensor([0, 0, 1-E1]).T.to(device)
    return A, B


def mo(m_last, m0, dt, T1):
    return m0 + (m_last - m0) * math.exp(-dt/T1)

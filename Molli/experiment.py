# 基于256*256*3的矩阵进行逐点弛豫变化操作
# 如果是每个时刻t的点，应该是很难算。。而且目前也不考虑 Gradient spoil
# 那么目前就只考虑做一个【具有广播机制的】矩阵
import torch
import matrix_rot
import torch
import freprecess
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# point: m*3，m为抽样的点的个数
def rot(point:torch.Tensor, phi = torch.pi):
    # 返回 m*3，每条m仍为每个样本不变
    return point @ matrix_rot.zrot(phi).T

def relax(x, time, point, result, A, B, dt):
    for _ in range(time):
        point = point @ A.T + B
            # result[N_dt * index + dp + 1] = point[2]
            # dp += 1
        result.append(point)
        x.append(x[-1] + dt)
    return x, point, result
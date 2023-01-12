import torch
import matrix_rot
import main
import math
import freprecess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# points: m*3，m为抽样的点的个数
def rot(points:torch.Tensor, phi = math.pi):
    # 返回 m*3，每条m仍为每个样本不变
    return points @ matrix_rot.zrot(phi).T

def molli(info:main.info, points:torch.Tensor):
    # m = m_ini # 默认是[0,0,1]
    for LL in info.TI: # 每个不同的TI是一次LL Experiment
        # 
        A, B = freprecess.res(LL, info.T1, info.T2)
        # 只取 Mz
        A = torch.Tensor([
            [0,0,0],
            [0,0,0],
            [0,0,1]
        ]) @ A
        
        
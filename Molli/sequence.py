import torch
import matrix_rot
import torch
import freprecess
import math

# 默认全部都是 180y, 10y
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# points: m*3，m为抽样的点的个数
def rot(points:torch.Tensor, phi = torch.pi):
    # 返回 m*3，每条m仍为每个样本不变
    return points @ matrix_rot.zrot(phi).T

def molli(info, points:torch.Tensor):
    # for LL in info.TI: # 每个不同的TI是一次LL Experiment
    # m = torch.Tensor([0,0,1]).to(device).T
    # 默认是 1*3 的 [0,0,1] 开始

    # E1 = math.exp(- info.TR / info.T1_generate)
    # A_TI_1, B_TI_1 = freprecess.res(info.TI[0], info.T1_generate, info.T2, info.df)
    # A_TI_2, B_TI_2 = freprecess.res(info.TI[1], info.T1_generate, info.T2, info.df)
    Rflip_180 = matrix_rot.yrot(torch.pi) # 180y
    Rflip_10 = matrix_rot.yrot( info.fa_10 ) # 180y
        # Ei = info.fa_slice[index]
        # 有多少个 excitation 就做几次
    num_excitation = 2

    # 5-3
    dt = 0.05
    N_per_ext = int(info.TI / dt)
    
    # N_TI_5, N_TI_3 = int(info.TR[0] / dt), int(info.TR[1] / dt)
    N_per_interval = int(info.TR / dt)
    N_5_rest = int(N_per_ext - N_per_interval * 4 - info.t_before[0])
    N_3_rest = int(N_per_ext - N_per_interval * 2 - info.t_before[1])

    N_dt = num_excitation * N_per_ext
    # result = torch.zeros(N_dt * 5 + 1)
    # result[0] = points[2]
    result = []
    result.append(points[2])

    for index in range(num_excitation):
        # dp = 0
        points = points @ Rflip_180.T
        # result[N_dt * index + dp + 1] = points[2]
        result.append(points[2])
        # dp += 1
        A, B = freprecess.res(dt, info.T1_generate, info.T2, info.df)
        for _ in range(info.t_before[0]):
            points = points @ A.T + B
            # result[N_dt * index + dp + 1] = points[2]
            # dp += 1
            result.append(points[2])
        for _ in range(4):
            points = points @ Rflip_10.T
            # result[N_dt * index + dp + 1] = points[2]
            # dp += 1 
            result.append(points[2])
            for _ in range (N_per_interval):
                points = points @ A.T + B
                # result[N_dt * index + dp + 1] = points[2]
                # dp += 1
                result.append(points[2])
        for _ in range(N_5_rest):
            points = points @ A.T + B
            # result[N_dt * index + dp + 1] = points[2]
            # dp += 1
            result.append(points[2])
    return result

        # points = points @ A_TI_1.T + B_TI_1 # 每个样本都会加上B
        # result[N_dt * index * dp] = points[2]
        # dp += 1
        # A, B = freprecess.res(dt, info.T1_generate, info.T2, info.df)
        # for i in range(N_dt - 2):
        #     points = points @ A.T + B
        #     result[N_dt * index * dp] = points[2]
        #     dp += 1


# def readoutplus(m, m0, p, E1, info, index):
#     k = E1 * torch.cos(info.fa_slice[index])
#     b = m0 *  (1 - E1) * (1 - torch.pow(k, ))
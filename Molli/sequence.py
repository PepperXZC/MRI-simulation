import torch
import matrix_rot
import torch
import freprecess
import math

# 默认全部都是 180y, 10y
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# point: m*3，m为抽样的点的个数
def rot(point:torch.Tensor, phi = torch.pi):
    # 返回 m*3，每条m仍为每个样本不变
    return point @ matrix_rot.zrot(phi).T

def relax(time, point, result, A, B):
    for _ in range(time):
        point = point @ A.T + B
            # result[N_dt * index + dp + 1] = point[2]
            # dp += 1
        result.append(point[2])
    return point, result

def molli(info, point:torch.Tensor):
    Rflip_180 = matrix_rot.yrot(torch.pi) # 180y
    Rflip_10 = matrix_rot.yrot( info.fa_10 ) # 180y
    num_excitation = 3

    dt = 1
    N_per_ext = int(info.TI / dt)
    
    # N_TI_5, N_TI_3 = int(info.TR[0] / dt), int(info.TR[1] / dt)
    N_per_interval = int(info.TR / dt)
    N_5_rest = int(N_per_ext - N_per_interval * 1 - info.t_before[0])
    N_3_rest = int(N_per_ext - N_per_interval * 2 - info.t_before[1])

    N_dt = num_excitation * N_per_ext

    result = []
    result.append(point[2])

    for index in range(num_excitation):
        point = point @ Rflip_180.T
        result.append(point[2])
        A, B = freprecess.res(dt, info.T1, info.T2, info.df)
        point, result = relax(info.t_before[0], point, result, A, B)
        for _ in range(4):
            point = point @ Rflip_10.T
            result.append(point[2])
            point, result = relax(N_per_interval, point, result, A, B)
        point, result = relax(N_5_rest, point, result, A, B)
    return result

        # point = point @ A_TI_1.T + B_TI_1 # 每个样本都会加上B
        # result[N_dt * index * dp] = point[2]
        # dp += 1
        # A, B = freprecess.res(dt, info.T1_generate, info.T2, info.df)
        # for i in range(N_dt - 2):
        #     point = point @ A.T + B
        #     result[N_dt * index * dp] = point[2]
        #     dp += 1

def Mz_relax(info, time:torch.Tensor, point, point_after_0):
    point_list = torch.zeros(len(time), 3)
    point_list[:,2] = point * (1 - torch.exp(- time / info.T1)) + point_after_0[2] * torch.exp(- time / info.T1)
    return point_list

def molli_relax(info, point:torch.Tensor):
    result = []
    Rflip_180 = matrix_rot.yrot(torch.pi) # 180y
    Rflip_10 = matrix_rot.yrot( info.fa_10 ) # 180y

    result.append(point[2])

    point = point @ Rflip_180.T
    result.append(point[2])

    dt = 0.01
    time_before = torch.arange(0, info.t_before[0], dt)
    point_before = Mz_relax(info, time_before, 1, point)
    result += point_before[:, 2].tolist()
    point = result[-1]

    t_rest_5 = info.TI - info.t_before[0] - info.TR * 4

    for i in range(4):
        # point = point @ Rflip_10.T
        point = torch.Tensor([0, 0, point]) @ Rflip_10.T
        result.append(point[2])
        # time_interval = torch.arange(info.t_before[0] + i * info.TR, info.t_before[0] + (i+1) * info.TR, dt)
        time_interval = torch.arange(0, info.TR, dt)
        point_interval = Mz_relax(info, time_interval, 1, point)
        result += point_interval[:, 2].tolist()
        point = result[-1]
    
    point = torch.Tensor([0, 0, point])
    time_rest = torch.arange(0, t_rest_5, dt)
    point_rest = Mz_relax(info, time_rest, 1, point)
    result += point_before[:, 2].tolist()
    point = result[-1]
    
    return result
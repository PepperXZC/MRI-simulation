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

def relax(x, time, point, result, A, B, dt):
    for _ in range(time):
        point = point @ A.T + B
            # result[N_dt * index + dp + 1] = point[2]
            # dp += 1
        result.append(point)
        x.append(x[-1] + dt)
    return x, point, result

class molli:
    def __init__(self, info) -> None:
        self.point = torch.Tensor([0,0,1]).to(device).T # 初始化
        self.info = info
        self.Rflip_10 = matrix_rot.yrot( info.fa_10 )
        self.Rflip_180 = matrix_rot.yrot(torch.pi)
    # 因为考虑到 10y 中有RF，需要同时制作时刻表 x_time
        self.dt = 0.1
        self.N_TI_5 = [int(num / self.dt) for num in info.TI_5]
        self.N_TI_3 = [int(num / self.dt) for num in info.TI_3]
        self.N_per_ext_5 = int(info.total_time[0] / self.dt)
        self.N_per_ext_3 = int(info.total_time[1] / self.dt)
        if len(self.N_TI_5) != 0:
            self.N_5_rest = int(self.N_per_ext_5 - self.N_TI_5[-1] - info.TR * info.rep_time / self.dt)
        else:
            self.N_5_rest = int(self.N_per_ext_5 - info.TR * info.rep_time / self.dt)
    
        if len(self.N_TI_3) != 0:
            self.N_3_rest = int(self.N_per_ext_3 - self.N_TI_3[-1] - info.TR * info.rep_time / self.dt)
        else:
            self.N_3_rest = int(self.N_per_ext_3 - info.TR * info.rep_time / self.dt)
    # num_excitation = 1
        self.result = []
        self.result.append(self.point)
        self.x_time = []
        self.x_time.append(0)
    
    def simulation(self):
        A, B = freprecess.res(self.dt, self.info.T1, self.info.T2, self.info.df)
        for index in range(self.info.num_excitation):
            self.point = self.point @ self.Rflip_180.T
            self.result.append(self.point)
            self.x_time.append(self.x_time[-1] + self.dt)
            # 假设 180y 是在同一个dt完成，也就是瞬间完成
            
            # point, result = relax(info.t_before[0], point, result, A, B)
            # for index in range(len(info.TI_5)):

            for i in range(len(self.info.TI_5)):
                self.point = self.point @ self.Rflip_10.T
                self.result.append(self.point)
                self.x_time.append(self.x_time[-1] + self.info.TR * self.info.rep_time)

                if i != len(self.info.TI_5) - 1:
                    n_interval = int((self.info.TI_5[i + 1] - self.info.TI_5[i] - self.info.TR * self.info.rep_time) / self.dt)
                else:
                    n_interval = self.N_5_rest

                self.x_time, self.point, self.result = relax(self.x_time, n_interval, self.point, self.result, A, B, self.dt)
            
            if len(self.info.TI_5) == 0:
                self.x_time, self.point, self.result = relax(self.x_time, self.N_5_rest, self.point, self.result, A, B, self.dt)

            self.point = self.point @ self.Rflip_180.T
            self.result.append(self.point)
            self.x_time.append(self.x_time[-1] + self.dt)

            for i in range(len(self.info.TI_3)):
                self.point = self.point @ self.Rflip_10.T
                self.result.append(self.point)
                self.x_time.append(self.x_time[-1] + self.info.TR * self.info.rep_time)

                if i != len(self.info.TI_3) - 1:
                    n_interval = int((self.info.TI_3[i + 1] - self.info.TI_3[i] - self.info.TR * self.info.rep_time) / self.dt)
                else:
                    n_interval = self.N_3_rest

                self.x_time, self.point, self.result = relax(self.x_time, n_interval, self.point, self.result, A, B, self.dt)

            # x_time, point, result = relax(x_time, N_5_rest, point, result, A, B, dt)
            if len(self.info.TI_3) == 0:
                self.x_time, self.point, self.result = relax(self.x_time, self.N_3_rest, self.point, self.result, A, B, self.dt)
    
    def catch(self,t):
        if type(self.dt) == int:
            t = int(round(t, 0))
        else:
            t = round(t, len(str(self.dt))-2)
        assert t in self.x_time
        index = self.x_time.index(t)
        return self.result[index]


    # print(point, result[0][2])
    # return x_time, point, result
    

    
    
    # N_TI_5, N_TI_3 = int(info.TR[0] / dt), int(info.TR[1] / dt)
    # N_per_interval = int(info.TR / dt)
    # N_5_rest = int(N_per_ext_5 - N_per_interval * 1 - info.t_before[0])
    
    # print(N_TI, N_5_rest)
    # N_3_rest = int(N_per_ext_3 - N_per_interval * 2 - info.t_before[1])

    # N_dt = num_excitation * N_per_ext

    

   

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
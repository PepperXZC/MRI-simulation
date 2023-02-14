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
        # result = [temp_list + [point] for temp_list in result]
        if dt < 1:
            x.append(round(x[-1] + dt, len(str(dt))-2))
        else:
            x.append(x[-1] + dt)
    return x, point, result

class molli:
    def __init__(self, info) -> None:
        # self.point = torch.Tensor([0,0,1]).to(device).T # 初始化
        self.point = info.m0
        self.m0 = info.m0
        self.info = info
        self.Rflip_10 = [matrix_rot.yrot( angle ) for angle in info.fa_10] # list
        self.Rflip_180 = matrix_rot.yrot(torch.pi)
    # 因为考虑到 10y 中有RF，需要同时制作时刻表 x_time
        self.dt = info.dt
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
        self.result = [[self.point] for _ in range(len(self.Rflip_10))]
        # self.result.append(self.point)
        # self.x_time = []
        # self.x_time.append(0)
        self.x_time = [[0] for _ in range(len(self.Rflip_10))]
        # 设置时间戳，记录 readout 所在的时间点 目前只设置 Rflip_10 只有一个角度的情况
        # 目前假设 TR * rep_time 之后得到一张图
        self.readout_time = []
    
    def simulation(self):
        A, B = freprecess.res(self.dt, self.info.T1, self.info.T2, self.info.df)
        for pulse_index in range(len(self.Rflip_10)):
            self.point = self.m0
            for _ in range(self.info.num_excitation):
                self.point = self.point @ self.Rflip_180.T
                self.result[pulse_index].append(self.point)
                # 因为 result 是多维的列表所以原本的 append 操作要按照下面这个来写
                # self.result = [temp_list + [self.point] for temp_list in self.result]
                self.x_time[pulse_index].append(self.x_time[pulse_index][-1] + self.dt)
                # self.x_time = [temp_list + [temp_list[-1] + self.dt] for temp_list in self.x_time]
                # 假设 180y 是在同一个dt完成，也就是瞬间完成
                assert len(self.x_time[0]) == len(self.result[0])
                # point, result = relax(info.t_before[0], point, result, A, B)
                # for index in range(len(info.TI_5)):
                
                for i in range(len(self.info.TI_5)):
                    # self.point = self.point @ self.Rflip_10.T
                    self.point = self.point @ self.Rflip_10[pulse_index].T
                    self.result[pulse_index].append(self.point)
                    # self.result = [temp_list + [self.point] for temp_list in self.result]
                    after_read_time = self.x_time[pulse_index][-1] + self.info.TR * self.info.rep_time
                    self.x_time[pulse_index].append(after_read_time)
                    self.readout_time.append(after_read_time)
                    # self.x_time = [temp_list + [temp_list[-1] + self.info.TR * self.info.rep_time] for temp_list in self.x_time]

                    assert len(self.x_time[0]) == len(self.result[0])
                    if i != len(self.info.TI_5) - 1:
                        n_interval = int((self.info.TI_5[i + 1] - self.info.TI_5[i] - self.info.TR * self.info.rep_time) / self.dt)
                    else:
                        n_interval = self.N_5_rest

                    self.x_time[pulse_index], self.point, self.result[pulse_index] = relax(self.x_time[pulse_index], n_interval, self.point, self.result[pulse_index], A, B, self.dt)

                    assert len(self.x_time[0]) == len(self.result[0])
                if len(self.info.TI_5) == 0:
                    self.x_time[pulse_index], self.point, self.result[pulse_index] = relax(self.x_time[pulse_index], self.N_5_rest, self.point, self.result[pulse_index], A, B, self.dt)

                self.point = self.point @ self.Rflip_180.T
                self.result[pulse_index].append(self.point)
                # self.result = [temp_list + [self.point] for temp_list in self.result]
                self.x_time[pulse_index].append(self.x_time[pulse_index][-1] + self.dt)
                # self.x_time = [temp_list + [temp_list[-1] + self.dt] for temp_list in self.x_time]
                assert len(self.x_time[0]) == len(self.result[0])
                for i in range(len(self.info.TI_3)):
                    # self.point = self.point @ self.Rflip_10.T
                    self.point = self.point @ self.Rflip_10[pulse_index].T
                    self.result[pulse_index].append(self.point)
                    # self.result = [temp_list + [self.point] for temp_list in self.result]
                    after_read_time = self.x_time[pulse_index][-1] + self.info.TR * self.info.rep_time
                    self.x_time[pulse_index].append(after_read_time)
                    self.readout_time.append(after_read_time)
                    # self.x_time = [temp_list + [temp_list[-1] + self.info.TR * self.info.rep_time] for temp_list in self.x_time]

                    if i != len(self.info.TI_3) - 1:
                        n_interval = int((self.info.TI_3[i + 1] - self.info.TI_3[i] - self.info.TR * self.info.rep_time) / self.dt)
                    else:
                        n_interval = self.N_3_rest

                    self.x_time[pulse_index], self.point, self.result[pulse_index] = relax(self.x_time[pulse_index], n_interval, self.point, self.result[pulse_index], A, B, self.dt)
                    assert len(self.x_time[0]) == len(self.result[0])
                # x_time, point, result = relax(x_time, N_5_rest, point, result, A, B, dt)
                if len(self.info.TI_3) == 0:
                    self.x_time[pulse_index], self.point, self.result[pulse_index] = relax(self.x_time[pulse_index], self.N_3_rest, self.point, self.result[pulse_index], A, B, self.dt)
    
    def catch(self,t):
        if type(self.dt) == int:
            t = int(round(t, 0))
        else:
            t = round(t, len(str(self.dt))-2)
        assert t in self.x_time
        index = self.x_time.index(t)
        # return [temp_list[index] for temp_list in self.result]
        return self.result[index, 2]

    
    def get_ro_time(self):
        torch_list = torch.Tensor(self.readout_time)
        return torch_list[self.info.readout_index]
    
    def readout533(self):
        time_list = self.get_ro_time()
        res = torch.zeros(len(time_list), 3)
        for i in range(len(time_list)):
            if self.info.dt < 1:
                temp_time = round(float(time_list[i]), len(str(self.info.dt)) - 2)
            else:
                temp_time = time_list[i]
            # index = torch.where(self.x_time == temp_time)[0]
            index = self.x_time[0].index(temp_time)
            res[i] = self.result[0][index]
        # 只输出Mz
        return res[:, 2]
        # return self.result[index]
        
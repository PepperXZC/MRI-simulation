import torch
import matrix_rot
import torch
import freprecess
import math
import sys
import copy

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
    
    def insert_points(self, pulse_index, now_point, now_time, new_point, new_time):
        n = int(self.info.TR * self.info.rep_time / self.dt)
        point_interval = (new_point - now_point) / n
        temp_point, temp_time = copy.deepcopy(now_point), copy.deepcopy(now_time)
        # temp_point_list, temp_time_list = [now_point], [now_time]
        for _ in range(n-1):
            temp_point += point_interval
            temp_time += self.dt
            if self.dt < 1:
                temp_time = round(float(temp_time), len(str(self.dt)) - 2)
            self.x_time[pulse_index].append(copy.deepcopy(temp_time))
            self.result[pulse_index].append(copy.deepcopy(temp_point))
        try:
            assert (now_point[0] + n * point_interval[0]) == new_point[0]
        except AssertionError:
            print(now_point[0], now_point[0] + n * point_interval[0], new_point[0])
        self.x_time[pulse_index].append(new_time)
        self.result[pulse_index].append(new_point)
    
    def ti_relax(self, pulse_index, ti_info, A, B):
        for i in range(len(ti_info)):
            if i == 0:
                rest = ti_info[0]
                self.x_time[pulse_index], self.point, self.result[pulse_index] = relax(self.x_time[pulse_index], rest, self.point, self.result[pulse_index], A, B, self.dt)

            now_point = self.point              
            self.point = self.point @ self.Rflip_10[pulse_index].T
            # self.result[pulse_index].append(self.point)
            # self.result = [temp_list + [self.point] for temp_list in self.result]
            now_time = self.x_time[pulse_index][-1]
            after_read_time = now_time + self.info.TR * self.info.rep_time
            # self.x_time[pulse_index].append(after_read_time)
            self.insert_points(pulse_index, now_point, now_time, self.point, after_read_time)
            self.readout_time.append(after_read_time)
            # self.x_time = [temp_list + [temp_list[-1] + self.info.TR * self.info.rep_time] for temp_list in self.x_time]

            assert len(self.x_time[0]) == len(self.result[0])
            if i != len(ti_info) - 1:
                n_interval = int((ti_info[i + 1] - ti_info[i] - self.info.TR * self.info.rep_time) / self.dt)
            else:
                n_interval = self.N_5_rest

            self.x_time[pulse_index], self.point, self.result[pulse_index] = relax(self.x_time[pulse_index], n_interval, self.point, self.result[pulse_index], A, B, self.dt)

            assert len(self.x_time[0]) == len(self.result[0])
        if len(ti_info) == 0:
            self.x_time[pulse_index], self.point, self.result[pulse_index] = relax(self.x_time[pulse_index], self.N_5_rest, self.point, self.result[pulse_index], A, B, self.dt)

    def inversion_relax(self, pulse_index):
        self.point = self.point @ self.Rflip_180.T
        self.result[pulse_index].append(self.point)
        # 因为 result 是多维的列表所以原本的 append 操作要按照下面这个来写
        # self.result = [temp_list + [self.point] for temp_list in self.result]
        self.x_time[pulse_index].append(self.x_time[pulse_index][-1] + self.dt)

    def simulation(self):
        A, B = freprecess.res(self.dt, self.info.T1, self.info.T2, self.info.df)
        for pulse_index in range(len(self.Rflip_10)):
            self.point = self.m0
            for _ in range(self.info.num_excitation):
                self.inversion_relax(pulse_index)
                # self.x_time = [temp_list + [temp_list[-1] + self.dt] for temp_list in self.x_time]
                # 假设 180y 是下一个dt完成，也就是瞬间完成
                assert len(self.x_time[0]) == len(self.result[0])                            
                self.ti_relax(pulse_index, self.info.TI_5, A, B)
                self.inversion_relax(pulse_index)
                assert len(self.x_time[0]) == len(self.result[0])

                self.ti_relax(pulse_index, self.info.TI_3, A, B)
                
    
    def catch(self,t):
        if type(self.dt) == int:
            t = int(round(t, 0))
        else:
            t = round(t, len(str(self.dt))-2)
        assert t in self.x_time
        index = self.x_time.index(t)
        # return [temp_list[index] for temp_list in self.result]
        return self.result[index, 2]

    
    def get_ro_time(self) -> list:
        # torch_list = torch.Tensor(self.readout_time)
        return self.readout_time[:8]
    
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
        
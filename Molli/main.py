import torch
import matrix_rot
import math
import sequence
import freprecess
import matplotlib.pyplot as plt
import experiment

import random
def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 默认在[0, 10000] 内寻找 T1
# 默认 TI 是 torch.tensor
class info:
    def __init__(
        self, 
        T1_generate = 100, 
        T2 = 45, 
        TR = 10,
        rep_time = 1, # 假设每个RF给出 rep_time 个读取次数 
        TI_5 = [50, 150, 250, 350, 450],
        # TI_5 = [],
        TI_3 = [100, 200, 300],
        readout_index = [0,5,1,6,2,7,3,4],
        # TI_3 = [],
        # 因为不知道第一个read开始之前有多少时间间隙
        FA_pulse = math.pi,
        # FA_small =  [10 * math.pi / 180],
        FA_small = torch.arange(5, 35, 5) * math.pi / 180,
        # fa_slice = 10,
        df = 30,
        t_interval = 30,
        total_time = [600, 500], # 5-3-3
        num_excitation = 1,
        fov = 10,
        bandwidth = 3,
        dt = 0.1,
        roll_rate = 5,
        c = 2
    ) -> None:
        '''
        在这里，整体的时间如下(按顺序)：
        total_time = (180y + TI_5[0] + TR * rep_time + TI_5[1] - ...)
        其中 TE 为每个 TR 中 10y时刻 到 readout 梯度的中点
        '''
        self.TI_5 = TI_5 # 每个 molli 中有两个TI
        self.TI_3 = TI_3 # 每个 molli 中有两个TI
        self.T1 = T1_generate # float64
        self.rep_time = rep_time
        self.fa_10 = FA_small
        self.TR = TR
        self.df = df
        self.total_time = total_time
        self.T2 = T2
        self.fa_180 = FA_pulse
        self.t_interval = t_interval
        self.num_excitation = num_excitation
        self.m0 = torch.Tensor([0,0,1]).to(device).T
        self.fov = fov
        self.bandwidth = bandwidth
        self.dt = dt
        # 这个量代表 每 1 秒钟，血流流动 roll_rate 厘米
        self.roll_rate = roll_rate
        # 这个量代表画框下每1cm对应实际心肌部位的 c:real_length厘米
        self.real_length = c
        self.readout_index = readout_index
        # self.fa_slice = fa_slice # 由 Slice profile 给出, 
        # 是个 5 维向量，因为有5个fa
        # 先不管它的 sub-slice 版本！只是一个数

        # self.theExp = torch.exp(-TI / T1_generate)
        
        # # 转化为 norm，我也不知道为什么非得这么做，先试试吧，可以删掉
        # self.theExp = self.theExp / torch.linalg.norm(self.theExp)

    def get_readout_time(self):
        return (self.TI_5 + self.TI_3)

def plot_5_graph(info_sample, molli_sample):
    for index in range(len(info_sample.fa_10)):
        angle = int(info_sample.fa_10[index] * 180 / math.pi)
        plt.plot(molli_sample.x_time[0], [key[2] for key in molli_sample.result[index]], color=randomcolor(), label=str(angle))
    plt.legend(loc=0)
    plt.show()

def readout_pool(info_sample, molli_sample):
    ro_time = molli_sample.get_ro_time()
    test_pool = experiment.pool(info_sample)
    pool_info = torch.zeros(len(ro_time), info_sample.fov, info_sample.fov)
    last_t = 0
    for i in range(len(ro_time)):
        t_interval = ro_time[i] - last_t
        test_pool.roll(t_interval)
        pool_info[i] = test_pool.pool
        last_t = ro_time[i]

def main():
    test_info = info()

    m0 = torch.Tensor([0,0,1]).to(device).T
    # result = sequence.molli_relax(test_info, m0)
    program = sequence.molli(test_info)
    # x = torch.arange(0, len(result), 1)
    program.simulation()
    # for t in program.x_time:
    #     program.catch(t)
    # print([key[2] for key in result])
    # print(program.readout_time)
    ro_time = torch.Tensor(program.readout_time)
    index = [0,5,1,6,2,7,3,4]
    
    
    
    plot_5_graph(test_info, program)
    # print(program.get_ro_time())
    # print(len(program.x_time[0]))
    
    # 下面这个过程检查 是否每两个时刻间隔都是 0.1
    # for i in range(1, len(program.x_time[0])):
    #     if (len(str(program.x_time[0][i])) - len(str(program.x_time[0][i-1]))) > 1:
    #         print('False', program.x_time[0][i], program.x_time[0][i - 1])
    # 9503 = 5 * TR * rep_time
    # [0, 0.1, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8]
    #  在 x_time 中存在数据缺省: 我们跳过了 TR * reptime 之间的间隔，选择直接得到结果了
    # 需要补全这中间的间隔
    # print(pool_info[0][0][0])
    # print(program.x_time[0].index(pool_info[0][0][0]))
    # print(program.readout533())

main()
# for index in range(len(test_info.fa_10)):
#     angle = int(test_info.fa_10[index] * 180 / math.pi)
#     plt.plot(program.x_time[0], [key[2] for key in program.result[index]], color=randomcolor(), label=str(angle))
#     plt.legend(loc=0)
# plt.show()
# plt.plot(program.x_time, [key[0] for key in program.result], color='r', label='Mx')
# plt.plot(program.x_time, [key[1] for key in program.result], color='g', label='My')
# plt.show()

# print(program.x_time[0])

# exp = experiment.pool(test_info)
# exp.roll(7)
# print(exp.pool)
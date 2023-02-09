import torch
import matrix_rot
import math
import sequence
import freprecess
import matplotlib.pyplot as plt

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
        # TI_3 = [],
        # 因为不知道第一个read开始之前有多少时间间隙
        FA_pulse = math.pi,
        FA_small =  10 * math.pi / 180,
        # fa_slice = 10,
        fa_readout = [[10, 10, 10, 10, 10], [10, 10, 10]],
        df = 30,
        t_interval = 30,
        total_time = [600, 500], # 5-3-3
        num_excitation = 1,
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
        # self.fa_slice = fa_slice # 由 Slice profile 给出, 
        # 是个 5 维向量，因为有5个fa
        # 先不管它的 sub-slice 版本！只是一个数

        # self.theExp = torch.exp(-TI / T1_generate)
        
        # # 转化为 norm，我也不知道为什么非得这么做，先试试吧，可以删掉
        # self.theExp = self.theExp / torch.linalg.norm(self.theExp)


test_info = info()

m0 = torch.Tensor([0,0,1]).to(device).T
# result = sequence.molli_relax(test_info, m0)
program = sequence.molli(test_info)
# x = torch.arange(0, len(result), 1)
program.simulation()
# for t in program.x_time:
#     program.catch(t)
# print([key[2] for key in result])
plt.plot(program.x_time, [key[2] for key in program.result], color='b', label='Mz')
plt.plot(program.x_time, [key[0] for key in program.result], color='r', label='Mx')
plt.plot(program.x_time, [key[1] for key in program.result], color='g', label='My')
plt.legend(loc=0)
plt.show()

# dT = 1
# T = 1000
# N = int(T / dT + 1)
# df = 10	
# T1 = 600	
# T2 = 100	

# A, B = freprecess.res(dT,T1,T2,df)


# M = torch.zeros(N,3)
# M[0]=torch.Tensor([1,0,0]).T

# for k in range(1,N):
# 	M[k] = A @ M[k-1] +B


# # time = [0:N-1]*dT;
# time = torch.arange(0, N, dT)
# plt.plot(time,M[:,0], 'b-',time,M[:,1],'r--',time,M[:,2],'g-.')
# # plt.legend('M_x','M_y','M_z')
# plt.xlabel('Time (ms)')
# plt.ylabel('Magnetization')
# # plt.axis([min(time) max(time) -1 1]);
# # grid on; 
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
        TR = 20, 
        TI = 120,
        t_before = [20,20],
        # 因为不知道第一个read开始之前有多少时间间隙
        FA_pulse = math.pi,
        FA_small =  30 * math.pi / 180,
        # fa_slice = 10,
        fa_readout = [[10, 10, 10, 10, 10], [10, 10, 10]],
        df = 30,
        t_interval = 30,
        p = [5, 3] # 5-3-3
    ) -> None:
        self.TI = TI # 每个 molli 中有两个TI
        self.T1 = T1_generate # float64
        self.fa_10 = FA_small
        self.TR = TR
        self.t_before = t_before
        self.df = df
        self.T2 = T2
        self.p = p
        self.fa_180 = FA_pulse
        self.t_interval = t_interval
        # self.fa_slice = fa_slice # 由 Slice profile 给出, 
        # 是个 5 维向量，因为有5个fa
        # 先不管它的 sub-slice 版本！只是一个数

        # self.theExp = torch.exp(-TI / T1_generate)
        
        # # 转化为 norm，我也不知道为什么非得这么做，先试试吧，可以删掉
        # self.theExp = self.theExp / torch.linalg.norm(self.theExp)


test_info = info()

m0 = torch.Tensor([0,0,1]).to(device).T
# result = sequence.molli_relax(test_info, m0)
result = sequence.molli(test_info, m0)
x = torch.arange(0, len(result), 1)
plt.plot(x, result)
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
# plt.show()

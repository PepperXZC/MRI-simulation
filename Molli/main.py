import torch
import matrix_rot
import math
import sequence
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 默认在[0, 10000] 内寻找 T1
# 默认 TI 是 torch.tensor
class info:
    def __init__(
        self, 
        T1_generate = 200, 
        T2 = 100, 
        TR = 100, 
        TI = 500,
        t_before = [50,50],
        # 因为不知道第一个read开始之前有多少时间间隙
        FA_pulse = math.pi,
        FA_small = 30 * math.pi / 180,
        # fa_slice = 10,
        fa_readout = [[10, 10, 10, 10, 10], [10, 10, 10]],
        df = 0,
        t_interval = 15,
        p = [5, 3] # 5-3-3
    ) -> None:
        self.TI = TI # 每个 molli 中有两个TI
        self.T1_generate = T1_generate # float64
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
result = sequence.molli(test_info, m0)
x = torch.arange(0, len(result), 1)
plt.plot(x, result)
plt.show()
import torch
import matrix_rot
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 默认在[0, 10000] 内寻找 T1
# 默认 TI 是 torch.tensor
class info:
    def __init__(
        self, 
        T1_generate = 0, 
        T2 = 0, 
        TR = 0, 
        TI = torch.zeros(4),
        FA_pulse = math.pi,
        FA_small = 10 * math.pi / 180
    ) -> None:
        self.TI = TI
        self.T1_generate = T1_generate # 只需要生成的T1就可以了
        self.fa_10 = FA_small
        self.TR = TR
        self.fa_180 = FA_pulse

        # self.theExp = torch.exp(-TI / T1_generate)
        
        # # 转化为 norm，我也不知道为什么非得这么做，先试试吧，可以删掉
        # self.theExp = self.theExp / torch.linalg.norm(self.theExp)




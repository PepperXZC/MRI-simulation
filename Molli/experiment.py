# 基于256*256*3的矩阵进行逐点弛豫变化操作
# 如果是每个时刻t的点，应该是很难算。。而且目前也不考虑 Gradient spoil
# 那么目前就只考虑做一个【具有广播机制的】矩阵
import torch
import matrix_rot
import torch
import freprecess
import random
import math
import sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class pool:
    def __init__(self, info) -> None:
        self.info = info
        self.pool = torch.zeros(self.info.fov, self.info.fov)
        self.vassel = torch.zeros(self.info.fov, self.info.bandwidth)
        # 设定随机阈值
        self.half = 9
    
    
    def roll(self, t):
        # 规定：每1秒钟，血流数组滚动1个单元
        each_time = self.info.real_length / self.info.roll_rate
        rest_time = t
        now_time = 0
        # 各管各的变化。先变化pool：
        self.pool += t
        # 然后只管 vassel：
        if rest_time - each_time < 0:
            self.vassel += rest_time
        while (rest_time - each_time >= 0):
            self.vassel += each_time
            rest_time -= each_time
            self.vassel = torch.roll(self.vassel, 1, 0)
            for i in range(len(self.vassel[0])):
                # if random.randint(1, 10) < self.half:
                self.vassel[0][i] = 0
        
        a, b = int(self.info.fov) // 2, int(self.info.bandwidth) // 2
        lower, upper = a - b, a + b
        self.pool[:, lower:upper+1] = self.vassel


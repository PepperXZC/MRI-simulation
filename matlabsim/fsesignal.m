function [Msig,Mss] = fsesignal(T1,T2,TE,TR,df,ETL)
R90 = yrot(pi/2);
[Ate2, Bte2] = freeprecess(TE/2,T1,T2,df);
[Atr, Btr] = freeprecess(TR-ETL*TE,T1,T2,df);
% 在同一个 TR 中，有多少个180x就是多少的ETL。
% TE/2-180-TE/2 是一个 echo
% Atr = [0 0 0;0 0 0;0 0 1]*Atr;

R180 = xrot(pi);
% 计算一整个TR经历的所有算子的叠加
A = eye(3);
B = [0;0;0];
for k=1:ETL
    A = Ate2 * R180 * Ate2 * A;
    B = Bte2 + Ate2 * R180 * (Ate2 * B + Bte2);
end
% 这里每步的Mn都是AnMn + Bn的形式，迭代的是An与Bn，
% Bn是不可或缺的，因为尽管第一步Bn=0，但是后面会随着迭代变大
% 因为不是直接代入M进行迭代

A = R90 * Atr * A;
B = R90*(Btr+Atr*B);
% steady state 的定义似乎是定义在 90y 之后那个瞬间的 M

Mss = inv(eye(3)-A)*B;
M = Mss;

for k=1:ETL
    M = Ate2 * R180 * Ate2 * M + Ate2 * R180 * Bte2 + Bte2;
    Mss(:,k)=M;
    Msig(k)=M(1)+1i*M(2);
end
% 经过1-8个TE之后的各M值，存储在Mss矩阵中

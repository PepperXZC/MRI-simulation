function [Mss,Msig]=gssignal(flip,T1,T2,TE,TR,dfreq,phi)
Rflip = yrot(flip);
[Atr,Btr] = freeprecess(TR-TE,T1,T2,dfreq);
[Ate,Bte] = freeprecess(TE,T1,T2,dfreq);

Atr = zrot(phi)*Atr; % TR末尾的dephase
% 下面的和sssignal一样的
Mss = inv(eye(3)-Ate*Rflip*Atr) * (Ate*Rflip*Btr+Bte);
Msig = Mss(1)+i*Mss(2);

% phi is the angle by which the magnetization is dephased at the end of the TR
% >>> Mss=gssignal(pi/3,600,100,2,10,0,pi/2)
% >>> Mss=[0.1248, 0.1129, 0.1965]'
% 如果加入了一个线性的Gz，那么跟随不同的z坐标，不同的M就会有不同的dephase
% 所以这里只是模拟了一下，假设给了一个dephase角度，该怎么算。
% 这里给的结论就是按 Rz(dephase)来算的。
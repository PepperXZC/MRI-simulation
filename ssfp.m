function [Msig,Mss]=ssfp(flip,T1,T2,TE,TR,dfreq,phi)
Rflip = yrot(flip);
[Atr,Btr] = freeprecess(TR-TE,T1,T2,dfreq);
[Ate,Bte] = freeprecess(TE,T1,T2,dfreq);

Atr = zrot(phi)*Atr; % TR末尾的dephase
% 下面的和sssignal一样的
Mss = inv(eye(3)-Ate*Rflip*Atr) * (Ate*Rflip*Btr+Bte);
Msig = Mss(1)+i*Mss(2);

% 与 gssignal.m 是完全相同的
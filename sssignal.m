function [Msig,Mss] = sssignal(flip,T1,T2,TE,TR,dfreq)

Rflip = yrot(flip);
[Atr,Btr] = freeprecess(TR-TE,T1,T2,dfreq);
[Ate,Bte] = freeprecess(TE,T1,T2,dfreq);
Mss = inv(eye(3)-Ate*Rflip*Atr) * (Ate*Rflip*Btr+Bte);
Msig = Mss(1)+1i*Mss(2);
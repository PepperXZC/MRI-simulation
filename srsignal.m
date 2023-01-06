function [Msig,Mss] = srsignal(flip,T1,T2,TE,TR,dfreq)

Rflip = yrot(flip);
[Atr,Btr] = freeprecess(TR-TE,T1,T2,dfreq);
[Ate,Bte] = freeprecess(TE,T1,T2,dfreq);

Atr = [0 0 0;0 0 0;0 0 1]*Atr;

Mss = inv(eye(3)-Ate*Rflip*Atr) * (Ate*Rflip*Btr+Bte);
Msig = Mss(1)+i*Mss(2);

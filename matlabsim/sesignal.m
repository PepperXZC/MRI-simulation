function [Msig,Mss] = sesignal(T1,T2,TE,TR,dfreq)

[Ate2, Bte2] = freeprecess(TE / 2,T1,T2,dfreq); % 0-TE, TE/2-TE 都可以用这个
[Atr, Btr] = freeprecess(TR - TE,T1,T2,dfreq);

Atr = [0 0 0; 0 0 0; 0 0 1] * Atr; % 只在这一段去取Mz？
% 从这里开始，就可以neglect所有的transverse magnetization，我不懂为什么

Rflip = yrot(pi/2);
Rrefoc = xrot(pi);

Mss = inv(eye(3)-Ate2*Rrefoc*Ate2*Rflip*Atr) * (Bte2+Ate2*Rrefoc*(Bte2+Ate2*Rflip*Btr));
Msig = Mss(1)+1i*Mss(2);


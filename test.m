df = 0;		% Hz off-resonance.
T1 = 600;	% ms.
T2 = 100;	% ms.
TE = 50;		% ms.
TR = 1000;	% ms.
flip = pi/2;	% radians.
ETL=8;

[Msig,Mss] = fsesignal(T1,T2,TE,TR,df,ETL)
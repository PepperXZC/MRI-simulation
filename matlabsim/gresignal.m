function [Mss,Msig]=gresignal(flip,T1,T2,TE,TR,dfreq)
N = 100;
M = zeros(3,N);
phi = ([1:N]/N - 0.5) * 4 * pi

for k=1:100
    M(:,k) = gssignal(flip,T1,T2,TE,TR,dfreq,phi(k));
end
Mss = mean(M, 2);
Msig = Mss(1)+i*Mss(2);
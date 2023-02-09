T1 = 600;	% ms.
T2 = 100;	% ms.
TE = 0:2.5:10;	% ms.
TR = 10;	% ms.
flip = pi/3;

df = [-100:100];

sig = zeros(length(df),length(TE));
for n=1:length(TE)
    for k=1:length(df)
        [Mss,Msig] = gresignal(flip,T1,T2,TE(n),TR,df(k));
        sig(k, n) = Msig;
    end
end
subplot(2,1,1);
plot(df,abs(sig));
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;
legend('TE=0', 'TE=2.5', 'TE=5.0', 'TE=7.5', 'TE=10');
title('Gradient-Spoiled Sequence Magnitude and Phase');
axis([-100 100 0 0.15]);

subplot(2,1,2);
plot(df,angle(sig));
xlabel('Frequency (Hz)');
ylabel('Phase (radians)');
axis([min(df) max(df) -pi pi]);
grid on;
legend('TE=0', 'TE=2.5', 'TE=5.0', 'TE=7.5', 'TE=10');


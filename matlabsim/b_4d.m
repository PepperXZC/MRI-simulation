T1 = 600;	% ms.
T2 = 100;	% ms.
TE = 2.5;	% ms.
TR = 5;	% ms.
flip = pi/3;
phi=[0 pi/2 pi 1.5*pi];

df = [-500:10:500]; 	%Hz

Sig = zeros(length(df),length(TE));

for n=1:length(phi)

	for k=1:length(df)
		[Msig, Mss] = ssfp(flip,T1,T2,TE,TR,df(k),phi(n));
		Sig(k,n)=Msig;
    end

end
	% ===== Plot the Results ======

subplot(2,1,1);
plot(df,abs(Sig));
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;
legend( '\phi=0', '\phi=\pi/2', '\phi=\pi', '\phi=3\pi/2');

subplot(2,1,2);
plot(df,angle(Sig));
xlabel('Frequency (Hz)');
ylabel('Phase (radians)');
axis([min(df) max(df) -pi pi]);
grid on;
legend( '\phi=0', '\phi=\pi/2', '\phi=\pi', '\phi=3\pi/2');
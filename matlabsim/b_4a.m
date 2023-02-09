T1 = 600;	% ms.
T2 = 100;	% ms.
TE = 0:2.5:10;	% ms.
TR = 10;	% ms.
flip = pi/3;

df = [-100:100]; 	%Hz

Sig = zeros(length(df),length(TE));

for n=1:length(TE)

	for k=1:length(df)
		[Msig,Mss] = sssignal(flip,T1,T2,TE(n),TR,df(k));
        % 没有Gz
		Sig(k,n)=Mss(1)+i*Mss(2);
	end;

end;
	% ===== Plot the Results ======

subplot(2,1,1);
plot(df,abs(Sig));
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;
legend('TE=0', 'TE=2.5', 'TE=5.0', 'TE=7.5', 'TE=10');

subplot(2,1,2);
plot(df,angle(Sig));
xlabel('Frequency (Hz)');
ylabel('Phase (radians)');
axis([min(df) max(df) -pi pi]);
grid on;
legend('TE=0', 'TE=2.5', 'TE=5.0', 'TE=7.5', 'TE=10');

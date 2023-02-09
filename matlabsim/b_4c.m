T1 = 600;	% ms.
T2 = 100;	% ms.
TR = [2,6,10];	% ms.
flip = pi/3;

df = [-500:500]; 	%Hz

Sig = zeros(length(df),length(TE));

for n=1:length(TR)

	for k=1:length(df)
		[Msig,Mss] = sssignal(flip,T1,T2,TR(n)/2,TR(n),df(k));
		Sig(k,n)=Mss(1)+1i*Mss(2);
    end

end
	% ===== Plot the Results ======

subplot(2,1,1);
plot(df,abs(Sig));
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;
legend( 'TR=2',  'TR=6', 'TR=10');

subplot(2,1,2);
plot(df,angle(Sig));
xlabel('Frequency (Hz)');
ylabel('Phase (radians)');
axis([min(df) max(df) -pi pi]);
grid on;
legend( 'TR=2',  'TR=6', 'TR=10');
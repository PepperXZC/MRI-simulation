dT = 1;		% 1ms delta-time.
T = 1000;	% total duration
df = 10;	% Hz off-resonance.
T1 = 600;	% ms.
T2 = 100;	% ms.
TE = 50;	% ms.
TR = 500;	% ms.

N1 = round(TE/2/dT);
N2 = round((TR-TE/2)/dT);

[A,B] = freeprecess(dT,T1,T2,df);

M = zeros(3,N1+N2);
M(:,1)=[0;0;1];

Rflip = yrot(pi/2); % 90y
Rrefoc = xrot(pi); % 180x

% 90 pulse
M(:,2)=A*Rflip*M(:,1)+B;
for k=3:(N1+1) % in TE/2
    M(:,k) = A*M(:,k-1)+B;
end
% 180 pulse
M(:,N1+2)=A*Rrefoc*M(:,N1+1)+B;

for k=2:N2-1 % in rest time
	M(:,k+N1+1) = A*M(:,k+N1)+B;
end;

time = [0:N1+N2-1]*dT;
plot(time,M(1,:),'b-',time,M(2,:),'r--',time,M(3,:),'g-.');
legend('M_x','M_y','M_z');
xlabel('Time (ms)');
ylabel('Magnetization');
axis([min(time) max(time) -1 1]);
grid on;
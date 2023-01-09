fplot('abs(sesignal(600,100,50,x,0))/sqrt(x)',[50,4000],'b-');
hold on;
fplot('abs(sesignal(1000,150,50,x,0)/sqrt(x))',[50,4000],'r--');
fplot('abs(sesignal(600,100,50,x,0)/sqrt(x))-abs(sesignal(1000,150,50,x,0)/sqrt(x))',[50,4000],'g-.');

grid on;
xlabel('TR (ms)');
ylabel('Signal efficiency');
title('Signal efficiency vs TR');
legend('Tissue A','Tissue B','A-B');
hold off;

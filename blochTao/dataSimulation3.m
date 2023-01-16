function dataset=dataSimulation3(T1, flip, M0, acq_Durations, inverf, TI, TR, ex_num)
%% 
dataset = zeros(1,9);  % 每次采集图像时的Mz
% acq_Durations = [782, 841, 782, 781, 3247, 781, 782, 782];

% TR = 2.82;	% ms.
% ex_num = 85; % 采集图像的脉冲数量
ATSum = (ex_num-1) * TR; % 采集一张图像的时间 ms.
% flip = 10/180*pi;

% inverf = 0.96;  % inversion factor 0.96
% M0 = 1;
Mz_0 = -M0*inverf; 
dataset(1,1) = Mz_0;

%% 1 2 3 4 5
TI1 = TI-100;
TI2 = TI;
% T1 = 1000;

% inversion recovery
Mz_1=M0*(1-exp(-TI1/T1))+Mz_0*exp(-TI1/T1);
dataset(1,2) = Mz_1;

for i = 2:5 
    Mz_0 = dataset(1,i);
    Mz_1 = FLASHsimulation(TR,T1,M0,flip,ex_num,Mz_0);
    Mz_r0 = Mz_1;
    TA_re = acq_Durations(i-1)-ATSum; 
    Mz_r1=M0*(1-exp(-TA_re/T1))+Mz_r0*exp(-TA_re/T1);
    dataset(1,i+1) = Mz_r1;
end

%% 3个recovery的心动周期

Re_3 = acq_Durations(5)-ATSum-TI2;
Mz_r0=FLASHsimulation(TR,T1,M0,flip,ex_num,dataset(1,6));
Mz_r1=M0*(1-exp(-Re_3/T1))+Mz_r0*exp(-Re_3/T1);

%% 6 7 8
Mz_02 = -Mz_r1*inverf; % inversion
% inversion recovery
Mz_12=M0*(1-exp(-TI2/T1))+Mz_02*exp(-TI2/T1);
dataset(1,7) = Mz_12;

for i = 1:2 

    Mz_0 = dataset(1,i+6);
    Mz_1 = FLASHsimulation(TR,T1,M0,flip,ex_num,Mz_0);

%     if i==3
%         break;
%     end
    Mz_r0 = Mz_1;
    TA_re = acq_Durations(5+i)-ATSum; 
    Mz_r1=M0*(1-exp(-TA_re/T1))+Mz_r0*exp(-TA_re/T1);
    dataset(1,i+7) = Mz_r1;

end

end
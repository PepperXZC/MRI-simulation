function [t1,M0fit,alphafit,inverfit,resnorm] = t1fit_onepx2(signal,acq_Durations,TI,TR,ex_num,m0_initial)

F  = @(p, acq_Durations) BLESSPC_lm2(p(1), p(2), p(3), p(4), TI, TR, ex_num,acq_Durations);
p0 = [1000, m0_initial, 10, 0.96];
options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt','Display','none',...
    'FunctionTolerance',1e-8,'StepTolerance',1e-12, 'MaxIterations',1000);
lb = [200, 0.1, 0.1, 0.96];
ub = [3000, inf, 20, 0.96];

coef = zeros(1,4);
% 非线性最小二乘求解器 lsqcurvefit
[coef(1,:),resnorm] = lsqcurvefit(F,p0,acq_Durations,signal,lb,ub,options);
t1 = coef(1,1);
M0fit = coef(1,2);
alphafit = coef(1,3);
inverfit = coef(1,4);

end



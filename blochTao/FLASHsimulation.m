function Mz_1 = FLASHsimulation(TR,T1,M0,flip,ex_num,Mz_0)

    E1 = exp(-TR/T1);
    K = E1*cos(flip);
    b = M0*(1-E1)*(1-power(K,ex_num-1))/(1-K);
    Mz_1 = (power(K,ex_num-1)*Mz_0+b)*cos(flip);
    
end
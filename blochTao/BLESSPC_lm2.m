function sam_sig=BLESSPC_lm2(T1, M0, degree, inverf, TI, TR, ex_num, acq_Durations)
    
    % p : [1000, m0_initial, 10, 0.96]
    flip = degree/180*pi;
    dataset=dataSimulation3(T1, flip, M0, acq_Durations, inverf, TI, TR, ex_num);
    ss = dataset .* sin(flip);
    sam_sig = ss(:,2:9);

end
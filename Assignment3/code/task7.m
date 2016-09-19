load('sp500.mat')
TRANS_HAT = [0 prior'; zeros(size(transition,1),1) transition];
EMIS_HAT = [zeros(1,size(emission,2)); emission];
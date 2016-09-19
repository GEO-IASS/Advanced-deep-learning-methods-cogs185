load('sp500.mat')
TRANS_HAT = [0 prior'; zeros(size(transition,1),1) transition];
EMIS_HAT = [zeros(1,size(emission,2)); emission];
[PSTATESS,~,~,~,~] = hmmdecode(price_change(1:100), TRANS_HAT, EMIS_HAT); 
viterbiStates = hmmviterbi(price_change(1:100), TRANS_HAT, EMIS_HAT);


a1 = plot(PSTATESS(2,:), 'LineWidth',3); 
M1 = 'bullish';
hold on;
a2 = plot(PSTATESS(3,:), 'LineWidth',3); 
M2 = 'bearish';
hold on;
a3 = plot(PSTATESS(4,:), 'LineWidth',3); 
M3 = 'stable ';
hold on;
legend([a1; a2; a3], [M1; M2; M3]);

figure;
plot(viterbiStates(1,:) - 1, 'LineWidth',3);
% task 2
transitionProb = [0.85 , 0.15; 0.10 , 0.90];
emissionProb = [1/3, 1/4, 5/12; 1/4, 1/4, 1/2;];
for i = 1: 50
    [seqs_train{i, 1}, states_train{i, 1}] = hmmgenerate(10, transitionProb, emissionProb);
end

save('task2.mat', 'seqs_train', 'states_train');
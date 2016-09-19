for i = 1: 2
    viterbiStates = hmmviterbi(seqs_train{i}, transitionProb, emissionProb)
    disp('train states = ')
    disp(states_train{i})
    disp('----')
end
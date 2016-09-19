for i = 1: 2
    viterbiStates = hmmviterbi(seqs_train{i}, transitionProb, emissionProb)
    disp('train states = ')
    disp(states_train{i})
    [PSTATESS,logpseq,FORWARD,BACKWARD,S] = hmmdecode(seqs_train{i}, transitionProb, emissionProb);
    disp('Posterior state probability: ');
    disp(PSTATESS);
    disp('Forward probability: ');
    disp(FORWARD);
    disp('Backward probability: ');
    disp(BACKWARD);
    disp('----')
end
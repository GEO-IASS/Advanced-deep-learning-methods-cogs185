for i = 1: 10
    [estimatedTrans, estimatedEmission] = hmmestimate(seqs_train{i}, states_train{i})
end
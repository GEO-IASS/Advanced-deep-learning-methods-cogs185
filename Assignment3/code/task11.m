load('story.mat')
voc_dict = unique(lower(training(:,1)));
tag_dict = unique(upper(training(:,2)));

% map voc and tag to numbers
for i = 1:size(training,1)
    training{i,3} = find(strcmp(voc_dict, lower(training{i,1})));
    training{i,4} = find(strcmp(tag_dict, upper(training{i,2})));
end

% generate HMM model
[estimatedTrans, estimatedEmission] = hmmestimate([training{:,3}], [training{:,4}]);

% generate sequence
seqs_train = {};
for i = 1: 10
    [seqs_train{i, 1}, states_train{i, 1}] = hmmgenerate(int32(rand(1)*120+20), estimatedTrans, estimatedEmission);
end

% map generated result back to string
seqs_train_res = {};
for i = 1: size(seqs_train,1)
    for j = 1: size(seqs_train{i},2)
        seqs_train_res{i}(j) = voc_dict(seqs_train{i}(j));
    end
end
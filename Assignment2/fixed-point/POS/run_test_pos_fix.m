%% here, parse and save the data 

data_splits = [500 1000 2000 4000 8000];

global ttime_global;        % time to evaluate the time consumption
global svm_param;           % svm parameter in training
global cst_label_vec;       % constant used in the label vector
global num_rep; 
global delta;               % standard deviation of the perturbation added 
cst_label_vec  = 0;         % 

% with perturbations
num_rep = 2;    delta = .25;          

% without perturbations
%num_rep = 1;   delta = 0;

M = 10;      % range of context
numIter = 5;

rand('seed', 1);
for xx = 1 : 5
    data_split = data_splits(xx);
    
    delete POS/local/posTrainData.mat
    delete POS/local/trainData.mat

    str = sprintf('POS/data/%i/posTrainData.mat', data_split);
    load(str)
    save('POS/local/posTrainData.mat', 'theData');
    cval = 1;   % trade-off parameter of SVM
    svm_param = ['-s 5 -c ' num2str(cval)];
    
    ttime_global = struct;
    ttime_global.train_svm = [];
    ttime_global.extend_context = [];
    ttime_global.predict_svm = [];
    
    ttime_global.test_time_in_train = [];
    
    % Fixed-Point Model
    [testErrRateMat, testErrRate2Mat, ttime, test_time] = test_pos_fix(M/2 ,numIter);
    str = sprintf('POS/result_fix/%i_%f.mat', data_split, delta);
    save(str, 'testErrRateMat', 'testErrRate2Mat', 'ttime', 'ttime_global');
end
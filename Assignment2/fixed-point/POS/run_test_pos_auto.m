%% here, parse and save the data 

%
%       POS data is of high-dimensional, each word has 446054-D features,
%       so it is not easy to fit into memory even using sparse matrix. So I
%       have optimized the code on POS dataset. The training time of
%       auto-context moel consists of the time: the time to extend the
%       context (excluding the time to process the data), the time to train
%       the SVM model, and the time to apply the auto-context model on the
%       training sentences (excluding the time to process the training data
%       again). Hope this is a fair comparision between the training time
%       of auto-context and fixed-point; 
%

data_splits = [500 1000 2000 4000 8000];

global ttime_global;        % time to evaluate the time consumption
global svm_param;           % svm parameter in training
global cst_label_vec;       % constant used in the label vector
cst_label_vec  = 0;         % 
M = 6;                      % range of context
numIter = 5;                % the number of iterations
for xx = 1 : 5
    data_split = data_splits(xx);
    
    delete POS\local\posTrainData.mat
    delete POS\local\trainData.mat
    str = sprintf('POS\\data\\%i\\posTrainData.mat', data_split);
    load(str)
    save('POS\local\posTrainData.mat', 'theData');
    cval = 1;   % trade-off parameter of SVM
    svm_param = ['-s 5 -c ' num2str(cval)];
    
    %   Auto-context Model
    ttime_global = struct;
    ttime_global.train_svm = [];
    ttime_global.extend_context = [];
    ttime_global.predict_svm = [];
    ttime_global.test_time_in_train = [];
    [testErrRateMat_auto, testErrRate2Mat_auto, ttime_auto, test_time_auto] = test_pos_auto_context(M/2 ,numIter);
    
    %                   time to extend the context              time to train the SVMs         time to predict the training setences 
    train_time = sum(ttime_global.extend_context(2:end)) + sum(ttime_global.test_time_in_train) + sum(ttime_global.predict_svm);
    
    str = sprintf('auto_%i.mat', data_split);
    save(str, 'ttime_global', 'testErrRateMat_auto', 'testErrRate2Mat_auto', 'train_time');
end
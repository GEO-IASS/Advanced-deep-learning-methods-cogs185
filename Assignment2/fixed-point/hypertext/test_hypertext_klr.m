function [acc, acc_train, m_time_train, m_time_test] = test_hypertext_klr(M, numIter)
% This function performs the hypertext classification task using the
%   fixed-point model 
% M             the order-of context
% numIter       the number of iterations in testing

load dataToUse

numData = size(data, 2); 
dim = size(data, 1); 

labels = theLabels; 
labels(theLabels == 2) = 1;
labels(theLabels == 3) = 2;
labels(theLabels == 4) = 3;
labels(theLabels == 6) = 4;
labels = labels + 1; 

schools = unique(theSchool); 

[otlk, inlk] = extend_link(numData, M, in_link2, out_link2); 

data = data_norm; 

acc_mat = []; 
num_mat = [];
acc_mat_train = [];
num_mat_train = []; 
time_test_mat = []; 
time_train_mat = []; 
for i = 1 : length(schools)
    allIdx = 1 : numData;
    idx = find(theSchool == schools(i));
    testIdx = allIdx(idx);
    allIdx(idx) = [];
    
    label_1 = labels;
    label_2 = labels;
    label_1(testIdx) = -1;
    label_2(allIdx) = -1;
 
    tic, 
    featMat1 = extend_context_hyper(label_1, 5, allIdx, inlk, otlk, M); 
    
    G = zeros(length(allIdx), 5); 
    for i = 1 : length(allIdx)
        G(i, labels(allIdx(i))) = 1; 
    end
    XTrain  = [data(:, allIdx)', featMat1']; 
    model = klrtrain(XTrain, G); 
    
%    model = llsvmtrain(labels(allIdx), [data(:, allIdx)', featMat1'], ['-s 4 -c ' num2str(cval)]);
    time_train = toc; 
    
    time_train_mat = [time_train_mat, time_train]; 
    

    % predict the testing data 
    tic
    [accuracy_test] = iterative_prediction(numIter, model, data(:, testIdx), testIdx, labels, XTrain, inlk, otlk, M);
    time_test = toc; 
    
    time_test_mat = [time_test_mat, time_test]; 
    
    % predict the training data 
    tic
    [accuracy_train] = iterative_prediction(numIter, model, data(:, allIdx), allIdx, labels, XTrain, inlk, otlk, M);
    time_verify_train = toc; 

    acc_mat = [acc_mat, accuracy_test];
    num_mat = [num_mat, length(testIdx)];
    acc_mat_train = [acc_mat_train, accuracy_train]; 
    num_mat_train = [num_mat_train, length(allIdx)]; 
end

disp('testing error')
acc = sum(acc_mat .* num_mat) / sum(num_mat); 
disp(acc)
disp('training error')
acc_train = mean(sum(acc_mat_train .* num_mat_train) / sum(num_mat_train)); 
disp(acc_train)
disp('training time')
m_time_train = mean(time_train_mat); 
disp(m_time_train)
disp('testing time')
m_time_test = mean(time_test_mat); 
disp(m_time_test)

function [acc, acc_mat, pred_mat] = iterative_prediction(numIter, model, data, dataIdx, labels, XTrain, inlk, otlk, M)

acc_mat = []; 
pred_mat = []; 

new_labels = labels; 
new_labels(:) = -1;

rbfScale = 10;
for iter = 1 : numIter
    featMat2 = extend_context_hyper(new_labels, 5, dataIdx, inlk, otlk, M);
    % [predict_test, accuracy_test] = llsvmpredict(labels(dataIdx), [data',
    % featMat2'], model);
    Krbf = kernelRBF([data', featMat2'], XTrain, rbfScale);
    Y = Krbf * model;    
    [junk labelY] = max(Y,[],2);
    accuracy_test = sum(labelY == labels(dataIdx)) / length(dataIdx); 
    
    
    new_labels(dataIdx) = labelY(:, 1); 
    
    acc_mat = [acc_mat, accuracy_test];
end
acc = accuracy_test;


function model = klrtrain(X, G)
    C = size(G, 2);
    nInstances = size(X, 1);    
    fprintf(1, 'computing rbf kernel... \n');
    rbfScale = 10;
    Krbf = kernelRBF(X, X, rbfScale);
    % Krbf = rbf(X, 1);
    % load 'krbf.mat';
    options = [];    
    [drop, y] = max(G, [], 2);
    funObj = @(u)SoftmaxLoss2(u, Krbf, y, C);
    fprintf('Training kernel(rbf) multinomial logistic regression model...\n');
    lambda = 1e-5;
    %randn(nInstances*(C - 1), 1)
    uRBF = minFunc(@penalizedKernelL2_matrix, zeros(nInstances * (C - 1), 1), options, Krbf, C - 1, funObj, lambda);
    uRBF = reshape(uRBF,[nInstances C-1]);
    model = [uRBF zeros(nInstances,1)];
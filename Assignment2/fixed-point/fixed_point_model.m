function [models, errRate, errRate2] = fixed_point_model(trainData, testData, trainY, testY, ...
    M, C, numIter, svm_param, cst_label_vec)
%% a function using svm as the classifier
% Input:
% 	-trainData:     the training data
%   -testData:      the testing data
%   -trainY:        the label for the training data
%   -testY:         the label for the testing data
%   -M:             the range of context
%   -C:             the number of classes
%   -numIter:       the number of iterations in prediction
%   -svm_param:     the parameter for SVM
%   -cst_label_vec: the constant for labeling vector
% Output:
%   -model:         the model trained
%   -errRate:       average error per character
%   -errRate2:      average error per sequence
%
% author:	Quannan Li (quannan.li@gmail.com)

if exist('cst_label_vec', 'var') == 0
    cst_label_vec = 0; 
end

numNodes = 0;
for i = 1 : length(trainData)
    numNodes = numNodes + size(trainData{i}, 2);
end

% parse the training data 
D = size(trainData{1}, 1); 
W = zeros(2 * M * C + D, 1);
W = sparse(W);
G = zeros(C, numNodes);
c = 0;
curLine = 1;
for i = 1 : length(trainData)
    Ln = size(trainData{i}, 2); 
    if i/length(trainData) >= c/100
        fprintf('%i\t', c);
        c = c + 1;
    end
    
    W(1:D, curLine : curLine + Ln - 1) = trainData{i};
    for j = 1 : size(trainData{i}, 2)
        y = cst_label_vec*ones(C, 1);
        y(trainY(curLine)) = 1;        
        G(:, curLine) = y;
        curLine = curLine + 1;
    end
    
    W(D+1:end, curLine - Ln : curLine-1) = extend_context(G(:, curLine - Ln : curLine-1), M, C);
end

disp('start to train the linear SVM model')
% llsvmtrain is the train program of liblinear svm
[model] = llsvmtrain(trainY, W, svm_param, 'col');
%[a b c] = llsvmpredict(trainY, W, model); 
for i = 1 : numIter
    models.clfs{i} = model; 
end
models.cst_label_vec = cst_label_vec;

[errRate, errRate2] = iterative_test(testData, testY, models, D, C, M, numIter);
end
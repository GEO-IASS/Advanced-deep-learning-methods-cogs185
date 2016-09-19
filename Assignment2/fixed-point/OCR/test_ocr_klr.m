function [errRate, errRate2, ttime, test_time] = test_ocr_klr(letterData, foldNum, M, numIter)
%Input:     
%   -letterData:    the input OCR data
%   -foldNum:       the number of validation, from 0 to 9
%   -M:             the range of context, from 0 : 2 : 14
%Output:
%   -errRate:       the average error per character
%   -errRate2:      the average error per sequence  (used in the paper)
%   -ttime:          training time
%   -test_time:      testing time

%% Constants
    global cst_label_vec;
    global ratio_rep; 
    global delta; 
    
    FOLD = 6;
    BM = 7;
    NEXT_ID = 3;
   
    LABEL = 2;
    C = 26;
    D = 129;
    %% Data preparation
    trainLetterNumber = size(letterData(letterData(:, FOLD) == foldNum, :), 1);
    trainData = [];
    testData = [];    
    data.z = [];
    for i = 1 :  size(letterData, 1)
        y = cst_label_vec*ones(1, C);
        y(letterData(i, LABEL) + 1) = 1;
        data.z = [data.z; [1 letterData(i, BM : end) y]];  
        disp(size(data.z))
        if letterData(i, NEXT_ID) == -1
            
            if letterData(i, FOLD) == foldNum
                %for training
                trainData = [trainData; data];
            else
                %for testing
                testData = [testData; data];                
            end            
            data.z = [];
        end
    end
    
    %% extend training data with context
    W = zeros(trainLetterNumber, 2 * M * C + D);
    G = zeros(trainLetterNumber, C);
    curLine = 1;
    for n = 1 : size(trainData, 1)
        z = trainData(n).z;
        
        % if add perturbation
        delta2 = 0; 
        num_rep = 1;
        if rand < ratio_rep
            delta2 = delta; 
            num_rep = 2;
        end
             
        for nn = 1 : num_rep
            Ln = size(z, 1);
            W(curLine : curLine + Ln - 1, 1:D) = z(:, 1 : D);
            labeling = z(:, D + 1 : D + C); 
            if num_rep > 1 & nn > 1
                labeling = labeling + rand(size(labeling))*delta2;
            end
            W(curLine : curLine + Ln - 1, D+1:end) = extend_context_ocr(labeling, M, C);
            for i = 1 : Ln           
                G(curLine, :) = z(i, D + 1 : D + C);
                curLine = curLine + 1;
            end
        end
    end
    %% training
    tic, 
    model = my_train_kernel_logistic(W, G);
    ttime = toc; 
    
    %% Testinbbg    
    tic
    [errRate, errRate2] = run_test(testData, model, D, C, M, W, numIter);
    test_time = toc; 
end


function model = my_train_kernel_logistic(X, G)
    C = size(G, 2);
    D = size(X, 2);
    nInstances = size(X, 1);    
    fprintf(1, 'computing rbf kernel... \n');
    rbfScale = 1;
    Krbf = kernelRBF(X, X, rbfScale);
    options = [];    
    [drop, y] = max(G, [], 2);
    funObj = @(u)SoftmaxLoss2(u, Krbf, y, C);
    fprintf('Training kernel(rbf) multinomial logistic regression model...\n');
    lambda = 1e-2;
    %randn(nInstances*(C - 1), 1)
    uRBF = minFunc(@penalizedKernelL2_matrix, zeros(nInstances * (C - 1), 1), options, Krbf, C - 1, funObj, lambda);
    uRBF = reshape(uRBF,[nInstances C-1]);
    model = [uRBF zeros(nInstances,1)];
end

function [errRate, errRate2] = run_test(testData, model, D, C, M, XTrain, numIter)
    global cst_label_vec;
    numTestData = size(testData, 1);
    
    errCount = zeros(1, numIter);
    totalCount = 0;
    errCount2 = zeros(1, numIter);
    totalCount2 = 0;
    
    for n = 1 : numTestData
        z = testData(n).z;
        L = size(z, 1);
        g = z(:, D + 1 : D + C);
        z(:, D + 1 : D + C) = zeros(L, C);
        tzs = infer_context_klr(z, g, model, D, C, M, numIter, XTrain);
        [drop, tg] = max(testData(n).z(:, D + 1 : D + C)');
        
        totalCount = totalCount + L;
        totalCount2 = totalCount2 + 1; 
        for it = 1 : numIter
            [drop, ty] = max(tzs{it}(:, D + 1 : D + C)');
            errCount(it) = errCount(it) + sum(ty ~= tg);
            errCount2(it) = errCount2(it) + sum(ty ~= tg) / L; 
        end
        fprintf(1, '********** tested sample:%d   error rate: %f\n', totalCount, errCount(end) / totalCount);
    end
    errRate = errCount / totalCount;
    errRate2 = errCount2 / totalCount2; 
end
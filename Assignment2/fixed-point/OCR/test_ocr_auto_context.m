function [errRate, errRate2, ttime, test_time] = test_ocr_auto_context(letterData, foldNum, M, numIter)
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
    try 
        load model_auto_ocr_
    catch
%        numIter = 5;
        tic, 
        [models, trainErrRateMat, trainErrRate2Mat] = my_train_kernel_logistic_auto_context(trainData, trainLetterNumber, M, C, D, numIter); 
        ttime = toc; 
        save model_auto_ocr models
    end
    
    %% Testing     
    tic
    [errRate, errRate2] = run_test_auto_context(testData, models, D, C, M);
    test_time = toc;
end

function [errRateMat, errRate2Mat] = run_test_auto_context(testData, models, D, C, M)

numIter = length(models);
numTestData = size(testData, 1);
numTestInst = 0; 
for i = 1 : numTestData
    numTestInst = numTestInst + size(testData(i).z, 1);
end

Gb = zeros(numTestInst, C); 
errRateMat = []; 
errRate2Mat = []; 
for iter = 1 : numIter
    [errRate, errRate2, Gb] = run_test(testData, Gb, models{iter}.model, D, C, M, models{iter}.W);
    errRateMat = [errRateMat, errRate];
    errRate2Mat = [errRate2Mat, errRate2];
end

end

function [models, errRateMat, errRate2Mat] = my_train_kernel_logistic_auto_context(trainData, trainLetterNumber, M, C, D, numIter)

%   1. create data 
%   2. train a model
%   3. prediction

models = cell(1, numIter);
errRateMat = []; 
errRate2Mat = []; 

Gb = zeros(trainLetterNumber, C);
for iter = 1 : numIter
    
    % Create Suitable Data 
    W = zeros(trainLetterNumber, 2 * M * C + D);
    curLine = 1;
    for n = 1 : size(trainData, 1)        
        z = trainData(n).z;
        Ln = size(z, 1);
        z(:, D + 1 : D + C) = Gb(curLine : curLine + Ln - 1, :); 
        W(curLine : curLine + Ln - 1, 1:D) = z(:, 1:D);
        W(curLine : curLine + Ln - 1, D+1:end) = extend_context_ocr(Gb(curLine : curLine + Ln - 1, :), M, C);
        for i = 1 : Ln           
            G(curLine, :) = trainData(n).z(i, D + 1 : D + C);
            curLine = curLine + 1;
        end
    end
    
    % Train the model
    model = my_train_kernel_logistic(W, G);
    models{iter}.model = model;
    models{iter}.W = W;
    
    % perform prediction 
    if iter < numIter
        [errRate, errRate2, Gb] = run_test(trainData, Gb, model, D, C, M, W);
        errRateMat = [errRateMat, errRate];
        errRate2Mat = [errRate2Mat, errRate2];
    end
end
end

function [errRate, errRate2, newGb] = run_test(testData, Gb, model, D, C, M, XTrain)
    numTestData = size(testData, 1);
    errCount = 0;
    totalCount = 0;
    errCount2 = 0;
    totalCount2 = 0;
    newGb = zeros(size(Gb)); 
    curLine = 1; 
    for n = 1 : numTestData
        z = testData(n).z;
        L = size(z, 1);
        g = z(:, D + 1 : D + C);
        z(:, D + 1 : D + C) = Gb(curLine : curLine + L - 1, :);
        tz = infer_context_klr(z, g, model, D, C, M, 1, XTrain);
        [drop, ty] = max(tz{end}(:,  D+(1 :  C))');
        [drop, tg] = max(testData(n).z(:, D + 1 : D + C)'); 
    
        newGb(curLine : curLine + L - 1, : ) = tz{end}(:,  D+(1 :  C)); 
        curLine = curLine + L; 
        
        errCount = errCount + sum(ty ~= tg);
        totalCount = totalCount + L;
        errCount2 = errCount2 + sum(ty ~= tg) / L; 
        totalCount2 = totalCount2 + 1;  
    end
    errRate = errCount / totalCount;
    errRate2 = errCount2 / totalCount2;
end
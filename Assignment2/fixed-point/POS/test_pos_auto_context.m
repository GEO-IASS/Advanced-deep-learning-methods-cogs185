function [testErrRateMat, testErrRate2Mat, ttime, test_time] = test_pos_auto_context(M ,numIter)
%Input:
%   -M:                 the range of the context
%   -numIter:           the number of iterations 
%Output:  
%   -testErrRateMat:    average error per character
%   -testErrRate2Mat:   average error per sequence
%   -ttime:             training time
%   -test_time:         testing time

    %% Constants
    
    DIM = 446054;
    BM = 7;
    NEXT_ID = 3;
    LABEL = 2;
    C = 41;
    D = DIM;            %%%446054 + 1;
  
    %% training
    t_start = clock; 
    [models, trainErrRateMat, trainErrRate2Mat] = train_svm_model_auto_context('POS\local\posTrainData.mat', D, C, M, numIter, LABEL, BM, NEXT_ID);
    t_end = clock; 
    ttime = etime(t_end, t_start);
    
    tic, 
    [testErrRateMat, testErrRate2Mat] = run_test_less_memory_batch('POS\local\posTestData.mat', models, D, C, M, LABEL, BM, NEXT_ID);
    test_time = toc
end

% train the auto-context model with SVM as the classifier
function [models, errRateMat, errRate2Mat] = train_svm_model_auto_context(trainDataFile, D, C, M, numIter, LABEL, BM, NEXT_ID)

global svm_param;
global ttime_global;
global cst_label_vec;

models = cell(1, numIter); 
errRateMat = []; 
errRate2Mat = []; 

load(trainDataFile);
theData = theData'; 
numTrainInst = size(theData, 2);
Gb = zeros(C, numTrainInst);
num_Seq = 0; 
for iter = 1 : numIter
    
    t_st = clock();
    if iter == 1
        W = zeros(2 * M * C + D, 1);
        W = sparse(W);
        G = zeros(C, numTrainInst);
    end
    curLine = 1;
    ys = []; 
    c = 1; 
    for i = 1 : numTrainInst 
        if i*100 / numTrainInst > c
            fprintf('%d\t', c); 
            c = c + 1; 
        end
        y = cst_label_vec*ones(C, 1);
        y(theData(LABEL, i) + 1) = 1;
        ys = [ys y];
        
        if iter == 1
            W(1 : D, i) = theData(BM : end, i);
        end
            
        if theData(NEXT_ID, i) == -1
            Ln = size(ys, 2);
            G(:, curLine : curLine + Ln - 1) = ys;
            W(D+1 : end, curLine : curLine + Ln - 1) = extend_context(Gb(:, curLine : curLine + Ln - 1), M, C);
            curLine = curLine + Ln;
            ys = [];
            num_Seq = num_Seq + 1;
        end
    end
    fprintf('\n');
    t_end = clock;
    
    % The time to extend the context
    ttime_global.extend_context = [ttime_global.extend_context; etime(t_end, t_st)];
                    
    [drop, GG] = max(G, [], 1);
    GG = GG';
    disp(['training:   train svm for the ' int2str(iter) '-th iteration']);
    
    t_st = clock;
    [model] = llsvmtrain(GG, W, svm_param, 'col');
    t_end = clock; 
    ttime_global.train_svm = [ttime_global.train_svm; etime(t_end, t_st)];
    
    models{iter}.model = model; 
    
    if iter < numIter
        disp(['training:   predict using svm for the ' int2str(iter) '-th iteration']);
        t_st = clock;
        [errRate, errRate2, Gb] = run_test_less_memory_auto(trainDataFile, model, Gb, D, C, M, LABEL, BM, NEXT_ID);
        t_end = clock;
        
        % the time to predict the training sentences in auto-context;
        ttime_global.test_time_in_train = [ttime_global.test_time_in_train; etime(t_end, t_st)];

        errRateMat = [errRateMat, errRate];
        errRate2Mat = [errRate2Mat, errRate2];
    end
end
end


% infer the context based on the SVM model
function [errs] = infer_context_batch(z, g, models, D, C, M) 

    MAX_ITER = length(models);
    
    [drop, gg] = max(g, [], 1);
    gg = gg'; 
    y_pre = zeros(size(g));
    W = z(1 : D, :);
    errs = zeros(1, MAX_ITER);
    iterCount = 1; 
    while iterCount <= MAX_ITER
        W(D + 1 : D + 2 * M * C, :) = extend_context(y_pre, M, C);
        y = predict_svm_model(W, gg, C, models{iterCount}.model);
        
        [drop, pp] = max(y, [], 1); 
        err = sum(pp ~= gg');
        errs(iterCount) = err;
        
        iterCount = iterCount + 1;
        y_pre = y;
    end
end

% the function to predict the training sentences in auto-context; 
% specifically optimizied for POS dataset for better memory usage.  
function [errRate, errRate2, G] = run_test_less_memory_auto(testData, model, Gb, D, C, M, LABEL, BM, NEXT_ID)
    
    global cst_label_vec;
    
    errCount = 0;
    totalCount = 0;
    errCount2 = 0;
    totalCount2 = 0;
    
    load(testData);
    theData = theData'; 
    
    z = [];
    tys = [];
    curLine = 1;
    G = zeros(size(Gb)); 
    c = 0; 
    fprintf('testing: \t'); 
    for i = 1 :  size(theData, 2)
        if i*100 / size(theData, 2) > c
            fprintf('%i\t', c);
            c = c + 1;
        end
        y = cst_label_vec*ones(C, 1);
        y(theData(LABEL, i) + 1) = 1;
        z = [z, [theData(BM : end, i); y]];
        if theData(NEXT_ID, i) == -1
            L = size(z, 2);
            g = z(D + 1 : D + C, :);
            z(D + 1 : D + C, :) = Gb(:, curLine : curLine + L -1);
            tz = infer_context(z, g, model, D, C, M, 1);
            G(:, curLine : curLine + L - 1 ) = tz(D + 1 : D + C, :);
            
            [drop, ty] = max(tz(D + 1 : D + C, :));
            tys = [tys, ty];
            [drop, tg] = max(g);
            
            curLine = curLine + L;

            errCount = errCount + sum(ty ~= tg);
            totalCount = totalCount + L;
            errCount2 = errCount2 + sum(ty ~= tg)/L;
            totalCount2 = totalCount2 + 1;
            z = [];
        end
    end
    fprintf('n'); 
    
    errRate = errCount / totalCount;
    errRate2 = errCount2 / totalCount2;

%    save results tys errRate errRate2
end

% function to test the auto-context model on the testing split; 
% specifically optimized for POS data set.
function [errRateMat, errRate2Mat] = run_test_less_memory_batch(testData, models, D, C, M, LABEL, BM, NEXT_ID)
  
    global cst_label_vec; 
    
    numIter = length(models); 
    errRateMat = zeros(1, numIter); 
    errRate2Mat = zeros(1, numIter); 
    
    totalCount = 0;
    totalCount2 = 0;
    
    load(testData);
    theData = theData'; 
    
    z = [];
    tys = [];
    curLine = 1;
    c = 0; 
    fprintf('testing: \t'); 
    for i = 1 :  size(theData, 2)
        if i*100 / size(theData, 2) > c
            fprintf('%i\t', c);
            c = c + 1;
        end
        y = cst_label_vec*ones(C, 1);
        y(theData(LABEL, i) + 1) = 1;
        z = [z, [theData(BM : end, i); y]];
        if theData(NEXT_ID, i) == -1
            g = z(D + 1 : D + C, :);
            L = size(z, 2);
            [errs] = infer_context_batch(z, g, models, D, C, M); 
            errRateMat = errRateMat + errs; 
            errRate2Mat = errRate2Mat + errs/L; 
            
            curLine = curLine + L; 

            totalCount = totalCount + L;
            totalCount2 = totalCount2 + 1; 
            z = [];
        end
    end
    fprintf('n'); 
    
    errRateMat = errRateMat / totalCount;
    errRate2Mat = errRate2Mat / totalCount2;
%    save results tys errRateMat errRate2Mat
end


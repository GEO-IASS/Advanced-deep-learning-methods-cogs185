function [errRate, errRate2, ttime, test_time] = test_pos_fix(M ,numIter)
%Input
%   -M:          the range of the context
%   -numIter:    the number of iterations 
%Output
%   -errRate:   the error per character
%   -errRate2:  the error per sequence   (used in the paper)
%   -ttime:     the training time 
%   -test_time: the testing time 

    %% Constants
    global cst_label_vec; 
    global ttime_global; 
    global num_rep; 
    global delta;
    global g_model;
    
    DIM = 446054;
    BM = 7;
    NEXT_ID = 3;
    LABEL = 2;
    C = 41;
    D = DIM;            %%%446054 + 1;
    %% Data preparation
    load POS/local/posTrainData.mat
    % take 10% data as training data
    theData = theData(1:int32(size(theData,1)/10),:);
    theData = theData';
    trainLetterNumber = size(theData, 2);

    tic,
    disp('parse the data')
    z = [];
    c = 0;
    W = zeros(2 * M * C + D, 1);
    W = sparse(W);
    G = zeros(C, trainLetterNumber);
    curLine = 1;
    
    for i = 1 :  size(theData, 2)
        if i/trainLetterNumber >= c/100
            fprintf('%i\t', c);
            c = c + 1;
        end
        y = cst_label_vec*ones(C, 1);
        y(theData(LABEL, i) + 1) = 1;
        z = [z, [theData(BM : end, i); y]];
        if theData(NEXT_ID, i) == -1
            for ii = 1 : num_rep
                Ln = size(z, 2);
                W(1:D, curLine : curLine + Ln - 1) = z(1:D, :);
                
                labeling = z(D+1 : end, :);
                if num_rep > 1 & ii > 1
                    labeling = labeling + rand(size(labeling))*delta;
                end
                W(D+1:end, curLine : curLine + Ln - 1) = extend_context(labeling, M, C);
                for j = 1 : Ln
                    G(:, curLine) = z(D + 1 : D + C, j);
                    curLine = curLine + 1;
                end
            end
            z = [];
        end
    end
    fprintf('\n')
   
    %% training
    disp('start to train the linear SVM model')
    [drop, GG] = max(G, [], 1);
    GG = GG';
    tic
    
    g_model = train_svm_model(GG, W);
    ttime = toc;
    ttime_global.train_svm = ttime;
    
    tic, 
    [errRate, errRate2] = run_test_less_memory('POS/local/posTestData.mat', g_model, D, C, M, numIter, LABEL, BM, NEXT_ID);
	test_time = toc;
end

function [model] = train_svm_model(labels, data)

global svm_param;
[model] = llsvmtrain(labels, data, svm_param, 'col');

end

function [errRate, errRate2] = run_test_less_memory(testData, model, D, C, M, numIter, LABEL, BM, NEXT_ID)
    
    global cst_label_vec;
    
    errCount = 0;
    totalCount = 0;
    errCount2 = 0;
    totalCount2 = 0;
    
    load(testData);
    % crop first 10% rows as test data
    theData = theData(1:int32(size(theData,1)/10),:);
    theData = theData';    
    z = [];
    
    tys = [];
    for i = 1 :  size(theData, 2)
        y = cst_label_vec*ones(C, 1);
        y(theData(LABEL, i) + 1) = 1;
        z = [z [theData(BM : end, i); y]];
        if theData(NEXT_ID, i) == -1
            L = size(z, 2);
            g = z(D + 1 : D + C, :);

            z(D + 1 : D + C, :) = -zeros(C, L);
            tz = infer_context(z, g, model, D, C, M, numIter);
            [drop, ty] = max(tz(D + 1 : D + C, :));
            tys = [tys, ty];
            
            [drop, tg] = max(g);
            errCount = errCount + sum(ty ~= tg);
            totalCount = totalCount + L;

            errCount2 = errCount2 + sum(ty ~= tg)/L;
            totalCount2 = totalCount2 + 1;
            fprintf(1, '********** tested sample:%d   error rate: %f \t error rate2: %f \n', totalCount, errCount / totalCount, errCount2 / totalCount2);
            z = [];
        end
    end
    
    errRate = errCount / totalCount;
    errRate2 = errCount2 / totalCount2;
end

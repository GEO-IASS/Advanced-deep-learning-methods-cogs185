function [errRate, errRate2] = iterative_test(testData, testY, model, D, C, M, numIter)
%
    
    cst_label_vec = model.cst_label_vec;
    
    errCount = 0;
    totalCount = 0;
    errCount2 = 0;
    totalCount2 = 0;
    
    curLine =  1;
    tys = []; 
    for i = 1 : length(testData)
        g = [];
        L = size(testData{i}, 2);
        for j = 1 : size(testData{i}, 2)
            y = cst_label_vec*ones(C, 1);
            y(testY(curLine)) = 1;
            curLine = curLine + 1;
            g = [g y];
        end
        z = [testData{i} ; g];
        
        z(D + 1 : D + C, :) = -zeros(C, L);
        tz = iterative_infer_context(z, g, model, D, C, M, numIter, cst_label_vec);
        [drop, ty] = max(tz(D + 1 : D + C, :));
        tys = [tys, ty];
        [drop, tg] = max(g);
        errCount = errCount + sum(ty ~= tg);
        totalCount = totalCount + L;
        errCount2 = errCount2 + sum(ty ~= tg)/L;
        totalCount2 = totalCount2 + 1;
        fprintf(1, '********** tested sample:%d   error rate: %f \t error rate2: %f \n', totalCount, errCount / totalCount, errCount2 / totalCount2);
    end
    
    errRate = errCount / totalCount;
    errRate2 = errCount2 / totalCount2;
end
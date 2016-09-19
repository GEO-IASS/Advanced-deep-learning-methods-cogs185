function Y = my_predict_kernel_logistic(X, Yold, model, XTrain) 
    global cst_label_vec; 
    global sigma; 
    
    rbfScale = 1;
    Krbf = kernelRBF(X, XTrain,rbfScale);
    Y = Krbf * model;    
    [junk labelY] = max(Y,[],2);
    [drop, labelOldY] = max(Yold, [], 2);
    errCount = sum(labelY ~= labelOldY);
    totalCount = size(labelY, 1);
    fprintf(1, 'Accuracy = %f (%d / %d) \n', 1 - errCount / totalCount, totalCount - errCount, totalCount);
    
    if sigma < 0
        Y = cst_label_vec * ones(size(Y));
        for i = 1 : size(Y, 1)
            Y(i, labelY(i)) = 1;
        end
    else
        
        expY = exp(sigma*Y);
        for i = 1 : size(Y, 1)
            Y(i,:) = expY(i,:) / sum(expY(i,:));
        end
    end
end
function zs = infer_context_klr(z, g, model, D, C, M, MAX_ITER, XTrain)
    % infer context with kernel_logisitc regression
    
    iterCount = 0; 
    
    zs = cell(1, MAX_ITER); 
    
    % the iteration process 
    W = z(:, 1:D); 
    while iterCount < MAX_ITER
        W(:, D+1:D+2*M*C) = extend_context_ocr(z(:, D+1:end), M, C);
        y = my_predict_kernel_logistic(W, g, model, XTrain);
        iterDiff = norm(z(:, D + 1 : D + C) - y);
        z(:, D + 1 : D + C) = y;
        zs{iterCount+1} = z;
        iterCount = iterCount + 1;
        
        if iterDiff < 1e-3           
            break;
        end       
    end
    
    if iterCount < MAX_ITER
        for j = MAX_ITER : -1 : iterCount + 1
            zs{j} = zs{iterCount};
        end
    end
end
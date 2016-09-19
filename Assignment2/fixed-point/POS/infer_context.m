function z = infer_context(z, g, model, D, C, M, MAX_ITER)
    
    global ttime_global;
    
    iterCount = 0;
    [drop, gg] = max(g, [], 1);
    gg = gg';
    y_pre = z(D + 1 : D + C, :);
    W = z(1:D, :);
    while iterCount < MAX_ITER
        
        t_st = clock;
        W(D + 1 : D + 2 * M * C, :) = extend_context(y_pre, M, C);
        y = predict_svm_model(W, gg, C, model);
        t_end = clock;
        ttime_global.predict_svm = [ttime_global.predict_svm; etime(t_end, t_st)];
        
        iterDiff = norm(z(D + 1 : D + C, :) - y);
        z(D + 1 : D + C, :) = 1*y;
        if iterDiff < 1e-3 | all(all(y == y_pre))
            break;
        end       
        iterCount = iterCount + 1;
        y_pre = y; 
    end
end
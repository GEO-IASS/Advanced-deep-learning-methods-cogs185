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
    uRBF = minFunc(@penalizedKernelL2_matrix, zeros(nInstances * (C - 1), 1), options, Krbf, C - 1, funObj, lambda);
    uRBF = reshape(uRBF,[nInstances C-1]);
    model = [uRBF zeros(nInstances,1)];
end
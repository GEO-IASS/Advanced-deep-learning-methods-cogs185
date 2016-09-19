% train_svm_struct using option3
% double(y(1) ~= y(2)); double(y(2) ~= y(3))
% optmized using gibbs sampling
function train_svm_struct
    % ------------------------------------------------------------------
    %                                                      Generate data
    % ------------------------------------------------------------------
    load letter1.data
    global x_train x_test y_train y_test parm model C;
    
    % only load first 5000 samples
    letter1 = letter1(1:5000,:);
    FOLD = 6;
    LABEL = 2;
    NEXT_ID = 3;
    CURR_ID = 5;
    C = 26;
    foldNum = 1;
    chop = 3;
    BM = 7;
    D = 128;
    
    x_train = {};
    y_train = {};
    x_test = {};
    y_test = {};
    temp_x = {};
    temp_y = [];
    
    for i = 1 :  size(letter1, 1)
        temp_x = [temp_x letter1(i, BM:end)'];
        temp_y = [temp_y letter1(i, LABEL) + 1];
        
        if(letter1(i, NEXT_ID) == -1)
            if letter1(i, FOLD) == foldNum
                %for training
                x_train = [x_train;{temp_x(:, 1: chop)}];
                y_train = [y_train;temp_y(:, 1: chop)];
            else
                %for testing
                x_test = [x_test;{temp_x(:, 1: chop)}];
                y_test = [y_test;temp_y(:, 1: chop)];               
            end
            temp_x = {};
            temp_y = [];
        end
    end
    
    rand('seed', 1);
    perm = randperm(length(x_train));
    training_samples = 20;
    
    x_train = x_train(perm);
    x_train = x_train(1:training_samples,1:end);
    y_train = y_train(perm);
    y_train = y_train(1:training_samples,1:end);
    x_test = x_test(perm);
    y_test = y_test(perm);
    testing_samples = size(y_test, 1);
    % ------------------------------------------------------------------
    %                                                    Run SVM struct
    % ------------------------------------------------------------------
    parm.patterns = x_train ;
    parm.labels = y_train ;
    parm.lossFn = @lossCB ;
    parm.constraintFn  = @constraintCB ;
    parm.featureFn = @featureCB ;
    parm.dimension = C*D*chop+2;
    parm.verbose = 1 ;
    tic
    model = svm_struct_learn(' -c 1.0 -o 1 -v 1  ', parm) ;
    training_time = toc;
    w = model.w ;
    
    % predict
    tic
    correct = 0;
    for i = 1: length(x_test)
        if sum(predict(parm, model, x_test{i}) ~= y_test{i}) == 0
            correct = correct + 1;
        end
    end
    testing_time = toc;
    
    disp('acc: ')
    acc = correct/length(x_test)
    save('train_svm_struct_2_gibbs_test.mat', 'acc', 'training_time', 'testing_time', 'training_samples', 'testing_samples');
end
% ------------------------------------------------------------------
%                                               SVM struct callbacks
% ------------------------------------------------------------------

function ypredict = predict(param, model, x) 
  max = -1;
  C = 26;
  iter = 3;
  % use gibbs sampling
  for i = 1: iter
      for j = 1: 3
          for k = 1: C
              yh = [1 1 1];
              yh(j) = k;
              val = dot(featureCB(param, x, yh), model.w);
              if val > max
                  max = val;
                  ypredict = yh;
              end
          end
      end
  end
  
end

function delta = lossCB(param, y, ybar)
  delta = double(sum(y ~= ybar)) ;
end

function psi = featureCB(param, x, y)
  D = 128;
  window_size = 3;
  res = [];
  for i=1:window_size
        res = [res; zeros( D*(y(i) - 1), 1); x{i} ;zeros( D*(26-y(i)), 1)];
  end
  psi = sparse([res ;double(y(1) ~= y(2)); double(y(2) ~= y(3))]);
end

function yhat = constraintCB(param, model, x, y)
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>
  max = -1;
  C = 26;
  iter = 3;
  % use gibbs sampling
  for i = 1: iter
      for j = 1: 3
          for k = 1: C
              yh = [1 1 1];
              yh(j) = k;
              val = dot(featureCB(param, x, yh), model.w) + sum(lossCB(param, y, yh))/length(yh);
              if val > max
                  max = val;
                  yhat = yh;
              end
          end
      end
  end
end
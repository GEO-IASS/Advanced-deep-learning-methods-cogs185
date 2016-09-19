% 
%   This script performs hypertext classification using the Conditional
%   Random Fields. The toolbox of UGM by Mark Schmidt is called.  
%   
%   Both CRFs with/without regularization are used, and the CRF with L2 regularization has better generization ability. 
%

clear all; 
close all; 
clc; 

FLG_REGULARIZATION = 2;             %   0:      No regularization
                                    %   2:      L2-Regularized
                                  
lambda_val = 1; 
                                        
load dataToUse.mat
numData = size(data, 2);
dim = size(data, 1); 

labels = theLabels; 
labels(theLabels == 2) = 1;
labels(theLabels == 3) = 2;
labels(theLabels == 4) = 3;
labels(theLabels == 6) = 4;

ys = int32(1+labels);
schools = unique(theSchool); 

% prepare the adjacency matrix % 
adjs = zeros(numData); 
for i = 1 : numData
    idx = in_link2{i}; 
    adjs(i, idx) = 1; 
    idx = out_link2{i}; 
    adjs(i, idx) = 1; 
end
adjs = adjs + adjs'; 
adjs(adjs >= 1) = 1; 
for i = 1 : size(adjs, 1)
    adjs(i, i) = 0; 
end

data = data_norm; 
maxStates = max(ys);
nInstances = 1; 

trainAccs = []; 
testAccs = [];
ws = cell(1, length(schools)); 

t1 = 0; 
t2 = 0; 

for i = 1 : length(schools)
    %%      data spliting     
    allIdx = 1 : numData;
    idx = find(theSchool == schools(i));
    testIdx = allIdx(idx);
    allIdx(idx) = []; 
    
    %%      training  
    % Prepare the y %
    nNodes = length(allIdx);
    y = ys(allIdx); 
    y = reshape(y,[1 1 nNodes]);
    
    % prepare the adj %
    adj = adjs(allIdx, allIdx); 
    nStates = repmat(maxStates, 1, nNodes); 
    edgeStruct = UGM_makeEdgeStruct(adj,nStates);

    % Prepare the Xnode %
    X = data_norm(:, allIdx);
    Xnode = ones(1,dim+1,nNodes);
    Xnode(1, 2 : end, :) = X;
    nNodeFeatures = size(Xnode, 2); 
    
    % Prepare Xedge %
    Xedge = ones(1, 1, edgeStruct.nEdges);
    nEdgeFeatures = size(Xedge, 2); 
    
    % Prepare nodeMap %
    nodeMap = zeros(nNodes, maxStates, nNodeFeatures, 'int32'); 
    ic = 1; 
    idx_bias = []; 
    for j = 2 : maxStates
        for f = 1 : nNodeFeatures
            nodeMap(:, j, f) = ic;
            if f == 1
                idx_bias = [idx_bias, ic]; 
            end
            ic = ic + 1; 
        end
    end
    
    % prepare edgeMap % 
    edgeMap = zeros(maxStates, maxStates, edgeStruct.nEdges, nEdgeFeatures, 'int32'); 
    for j = 1 : maxStates
        for k = 1 : maxStates
            if k == j & k == 1
                continue; 
            end
            for f = 1 : nEdgeFeatures
                edgeMap(j, k, :, f) = ic; 
                idx_bias = [idx_bias, ic]; 
                ic = ic + 1; 
            end
        end
    end
    
    % train the parameter % 
    tic
    if isempty(ws{i})
        nParams = max([nodeMap(:);edgeMap(:)]);
        w = zeros(nParams,1);
        if FLG_REGULARIZATION == 0
            w = minFunc(@UGM_CRF_NLL,randn(size(w)),[],Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_LBP); 
        elseif FLG_REGULARIZATION == 2
            % Set up regularization parameters
            lambda = lambda_val*ones(size(w));
            lambda(idx_bias) = 0;          % Don't penalize node bias variable
            regFunObj = @(w)penalizedL2(w,@UGM_CRF_NLL,lambda,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_LBP);

            % Optimize
            w = zeros(nParams,1);
            w = minFunc(regFunObj,w);  
        end
        ws{i} = w; 
    else
        w = ws{i}; 
    end
    t1 = t1 + toc; 
    
    [nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,1);
    LBPDecoding = UGM_Decode_LBP(nodePot,edgePot,edgeStruct); 
    train_acc = sum(LBPDecoding == y(:)) / length(LBPDecoding);
    trainAccs = [trainAccs, train_acc];
    disp(train_acc)
    
    %%      testing   
    % Prepare the y %
    nNodes = length(testIdx);
    y = ys(testIdx); 
    y = reshape(y,[1 1 nNodes]);
    
    % prepare the adj %
    adj = adjs(testIdx, testIdx); 
    nStates = repmat(maxStates, 1, nNodes); 
    edgeStruct = UGM_makeEdgeStruct(adj,nStates);

    % Prepare the Xnode %
    X = data_norm(:, testIdx);
    Xnode = ones(1,dim+1,nNodes);
    Xnode(1, 2 : end, :) = X;
    nNodeFeatures = size(Xnode, 2); 
    
    % Prepare Xedge %
    Xedge = ones(1, 1, edgeStruct.nEdges);
    nEdgeFeatures = size(Xedge, 2); 
    
    % Prepare nodeMap %
    nodeMap = zeros(nNodes, maxStates, nNodeFeatures, 'int32'); 
    ic = 1; 
    idx_bias = []; 
    for j = 2 : maxStates
        for f = 1 : nNodeFeatures
            nodeMap(:, j, f) = ic;
            if f == 1
                idx_bias = [idx_bias, ic]; 
            end
            ic = ic + 1; 
        end
    end
    
    % prepare edgeMap % 
    edgeMap = zeros(maxStates, maxStates, edgeStruct.nEdges, nEdgeFeatures, 'int32'); 
    for j = 1 : maxStates
        for k = 1 : maxStates
            if k == j & k == 1
                continue; 
            end
            for f = 1 : nEdgeFeatures
                edgeMap(j, k, :, f) = ic; 
                idx_bias = [idx_bias, ic]; 
                ic = ic + 1; 
            end
        end
    end
    
    tic
    [nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct,1);
    LBPDecoding = UGM_Decode_LBP(nodePot,edgePot,edgeStruct); 
    t2 = t2 + toc; 
    
    test_acc = sum(LBPDecoding == y(:)) / length(LBPDecoding);
    testAccs = [testAccs, test_acc];
    disp(test_acc)
end

time_train  = t1 / 4; 
time_test = t2 / 4; 

str = sprintf('hypertext\\result\\crf_L2_%i.mat', lambda_val); 
save(str, 'testAccs', 'trainAccs', 'ws', 'time_train', 'time_test');
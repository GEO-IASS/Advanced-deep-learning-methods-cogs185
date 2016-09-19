% add the paths 

% add the library needed 
addpath library/additionalCode
addpath library/liblinear
addpath library/minFunc
addpath(genpath('library/UGM/UGM_2011'));

addpath hypertext
addpath OCR
addpath POS

load trainData; 
load trainY;
tic
[model, errRate, errRate2] = fixed_point_model(trainData, trainData, trainY, trainY, 2, 41, 2, '-s 5 -c 1', 0);
toc
%[model, errRate, errRate2] = auto_context_model(trainData, trainData, trainY, trainY, 3, 41, 2, '-s 5 -c 1', 0);

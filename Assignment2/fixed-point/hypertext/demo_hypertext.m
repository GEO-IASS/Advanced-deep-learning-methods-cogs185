% This script shows the experiment settings in the hypertexts 
% It produces the result reported in the paper
% For this data set, no pertubation is added as we use a histogram of the neighboring labelings

global cval;        % cval is: the trade-off parameter of SVM used 

%% Run SVM only 
cval = 0.01; 
[acc_svm, acc_train, time_train_svm] = test_hypertext(0, 1); 


%% Run Fixed-Point model 
cval = 0.01; 
[acc_fix, acc_train, time_train_fix] = test_hypertext(6, 6);    

%% Run Auto-context model 
cval = 0.01; 
[acc_auto, acc_train, time_train_auto] = test_hypertext_auto_context(6, 6); 

%% Run CRF on the hypertext 
hypertext_crf;
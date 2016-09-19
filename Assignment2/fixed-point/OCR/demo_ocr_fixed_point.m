% OCR dataset
load original_data/letter1.data

letter1 = letter1(1:5000,:);

global delta; 
global ratio_rep; 
global cst_label_vec;       % constant used in the labeling vector
global sigma;               % sigma to convert output of KLR to probabiity
                            % if sigma < 0, set the entry corresponding to
                            % the class with maximum output to be 1
delta = .1;
ratio_rep = 0;             % for ratio_rep of the samples, produce 2 replicas
                           %  if ratio_rep is 0, no perturbation
cst_label_vec  = -1;       % cst_label_vec  = -1 performs better than cst_label_vec  = 0
sigma = -1;                % sigma = -1 performs much better than sigma > 0
numIter = 5;                % number of iterations in testing

%% For the fixed-point model
err_cell = cell(1);
err_seq_cell = cell(1);
ttime_cell = cell(1); 
test_time_cell = cell(1); 
for M = 14 : -2 : 0     % range of context
    err_mat = [];
    err_seq_mat = [];
    ttime_mat = []; 
    test_time_mat = [];
    for i = 0 : 9       % the index of cross-validation 
        [err, err_seq, ttime, test_time] = test_ocr_klr(letter1, i, M/2, numIter);
        err_mat = [err_mat; err];
        err_seq_mat = [err_seq_mat; err_seq];
        ttime_mat = [ttime_mat, ttime];
        test_time_mat = [test_time_mat, test_time];
        
%        save ocr_fix err_mat err_seq_mat ttime_mat test_time_mat
    end
    err_cell{M/2 + 1} = err_mat;
    err_seq_cell{M/2 + 1} = err_seq_mat;
    ttime_cell{M/2 + 1} = ttime_mat;
    test_time_cell{M/2 + 1} = test_time_mat;
        
    str = ['OCR/result_ocr/ocr_res_fixed_point_' num2str(ratio_rep) '.mat'];
    save(str, 'err_cell', 'err_seq_cell', 'ttime_cell', 'test_time_cell')
end
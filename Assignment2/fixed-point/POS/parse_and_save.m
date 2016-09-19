%% parse and save the data
train_split = 2000; 

data_folder = 'original_data/pos';
cates_path = 'original_data/pos/cates'; 

try
    load posTrainData
catch
    disp('load the training data')
    data_train_path = sprintf('%s//train%i_train.shin', data_folder, train_split); 
    theData = parse_pos_data(data_train_path, cates_path); 
    a = size(theData, 2);
    theData(:, a+1:DIM+6) = 0; 
    save posTrainData theData
end

try
    load posVerifyData
catch
    disp('load the verifying data')
    data_verify_path = sprintf('%s//train%i_verify.shin', data_folder, train_split); 
    theData = parse_pos_data(data_verify_path, cates_path); 
    a = size(theData, 2);
    theData(:, a+1:DIM+6) = 0; 
    save posVerifyData theData
end

try
    load posTestData
catch
    disp('load the testing data')
    data_test_path = sprintf('%s//test.shin', data_folder); 
    theData = parse_pos_data(data_test_path, cates_path); 
    a = size(theData, 2);
    theData(:, a+1:DIM+6) = 0;  
    save posTestData theData 
end
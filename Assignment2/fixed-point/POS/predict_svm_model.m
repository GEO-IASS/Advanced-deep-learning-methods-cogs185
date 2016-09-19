function [Y] = predict_svm_model(X, Yold, C, model)
global cst_label_vec; 

[c, accuracy, decision] = llsvmpredict(Yold, sparse(X'), model);
labelY = c;

%errCount = sum(labelY ~= Yold);
%totalCount = size(labelY, 1);

Y = cst_label_vec * ones(C, size(Yold, 1));
for i = 1 : size(Y, 2)
	Y(labelY(i), i) = 1;
end

end
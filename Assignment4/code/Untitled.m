facemat = [80        85          60       55   50; 90        85          70       45 50; 95        80          40       50 60];
Z = facemat - (ones(3,1)*mean(facemat));
%C = Z*Z'/size(facemat,2); 
%[V, D] = eig(C);
%[sv si] = sort(diag(D),'descend');
%Vs = V(:,si);

[A_hat, E_hat, iter] = inexact_alm_rpca(Z);
[coeff,score,latent,tsquared,explained,mu]=pca(A_hat, 'Algorithm', 'svd');

%[coeff,score,latent,tsquared,explained,mu]=pca(facemat, 'Algorithm', 'svd');
%meData = facemat-(ones(3, 1) * mean(facemat));
%meData*coeff;


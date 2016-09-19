load('Face_40by40_500.mat')
[coeff, E_hat, iter] = inexact_alm_rpca(facemat, 2);
%[COEFF, SCORE, latent, tsquared, explained, mu1] = pca(facemat', 'Algorithm', 'eig');
res = COEFF(:, 1:100) * SCORE(:, 1:100)';
%Z = facemat - (ones(1600,1)*mean(facemat));
%C = Z*Z'/size(facemat,2); 
%[V, D] = eig(C);
%[sv si] = sort(diag(D),'descend');
%Vs = V(:,si);
[pic, maxgray] = getpgmraw('/Users/yunfanyang/Downloads/CroppedYale/yaleB09/yaleB09_P00A-020E-10.pgm');
I = pic;
                    
X = reshape(I,size(I,1)*size(I,2)/3,3);
[COEFF, SCORE, latent, tsquared, explained, mu1] = pca(X);
Itransformed = X*coeff;
Ipc1 = reshape(Itransformed(:,1),size(I,1),size(I,2)/3);
Ipc2 = reshape(Itransformed(:,2),size(I,1),size(I,2)/3);
Ipc3 = reshape(Itransformed(:,3),size(I,1),size(I,2)/3);
figure, imshow(Ipc1,[]);
figure, imshow(Ipc2,[]);
figure, imshow(Ipc3,[]);


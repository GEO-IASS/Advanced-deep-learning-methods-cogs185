function [XX] = kernelRBF(X1,X2,sigma)

disp('start to compute the kernel matrix')
%tic

XX = distance( X1, X2 ); 
% if ~exist('sigma', 'var')
%     sigma = prctile(prctile(XX, ratio), ratio); 
%     sigma = sigma^.5; 
% end

Z = 1/sqrt(2*pi*sigma^2);
XX = Z*exp(-XX/(2*sigma^2));
XX = full(XX); 
%toc

fprintf('\n'); 

function [ sdistance ] = distance( inst1, inst2 )
%DISTANCE Summary of this function goes here
%   Detailed explanation goes here

len1 = size(inst1, 1); 
len2 = size(inst2, 1); 

D1 = sum(inst1.^2,2); 
D2 = sum(inst2.^2,2); 

sdistance = repmat(D1,1, len2)+repmat(D2',len1, 1)-2*inst1*inst2';
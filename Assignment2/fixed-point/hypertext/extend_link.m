function [otlk, inlk] = extend_link(numData, M, in_link2, out_link2)
% This function performs the extension of higher-order links
%
inlk = cell(numData, M); 
otlk = cell(numData, M);
intotal = cell(numData, 1); 
ottotal = cell(numData, 1); 

k = 1; 
for i = 1 : numData
    inlk{i, k} = in_link2{i}; 
    intotal{i} = inlk{i, k}; 
    otlk{i, k} = out_link2{i};
    ottotal{i} = otlk{i, k}; 
end
k = k + 1; 
while k <= M
    for i = 1 : numData
        for j = 1 : length(inlk{i, k-1})
            inlk{i, k} = [inlk{i, k}, in_link2{inlk{i, k-1}(j)}];
        end
        new_ele = setdiff(inlk{i, k}, intotal{i});
        inlk{i, k} = new_ele; 
        %inlk{i, k} = unique(inlk{i, k}); 
        %inlk{i, k} = unique(new_ele);  
        intotal{i} = [intotal{i}, new_ele]; 
    end
    k = k + 1; 
end
k = 2;
while k <= M
    for i = 1 : numData
        for j = 1 : length(otlk{i, k-1})
            otlk{i, k} = [otlk{i, k}, out_link2{otlk{i, k-1}(j)}];
        end
        new_ele = setdiff(otlk{i, k}, ottotal{i}); 
        otlk{i, k} = new_ele;  
        %otlk{i, k} = unique(otlk{i, k}); 
        %otlk{i, k} = unique(new_ele); 
        ottotal{i} = [ottotal{i}, new_ele]; 
    end
    k = k + 1; 
end
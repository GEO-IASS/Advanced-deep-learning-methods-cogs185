function featMat = extend_context_hyper(labels, C, dataIdx, inlk, otlk, M)
% This function performs the extension of higher-order linking context
%

if M <= 0
    featMat = []; 
    return; 
end

USE_GLOBAL = 1; 

if USE_GLOBAL
	dim = 2*C*M+2*C;
else
    dim = 2*C*M;
end
num = length(dataIdx);
featMat = zeros(dim, num);

ini = zeros(C, 1);
ini_total = zeros(C, 1); 
feat = zeros(dim, 1);
for i = 1 : num
    c = 0;
    for j = 1 : M
        lbs = labels(inlk{dataIdx(i), j});
        ini(:) = 0;
        for k = 0 : C-1
            ini(k+1) = sum(lbs == k); 
        end        
        
        ini_total = ini_total + ini; 
        
        ini = ini / (sum(ini) + 1e-20); 
        feat(c+(1:C)) = ini; 
        c = c + C; 
    end
    if USE_GLOBAL   
        ini_total = ini_total / sum(ini_total + 1e-20); 
        feat(c+(1:C)) = ini_total; 
        c = c + C;
    end
    
	ini_total(:) = 0; 
    for j = 1 : M
        lbs = labels(otlk{dataIdx(i), j});
        ini(:) = 0; 
        for k = 0 : C-1
            ini(k+1) = sum(lbs == k); 
        end
                
        ini_total = ini_total + ini;
        
        ini = ini / (sum(ini) + 1e-20); 
        feat(c+(1:C)) = ini; 
        c = c + C; 
    end
    
    if USE_GLOBAL   
        ini_total = ini_total / sum(ini_total + 1e-20); 
        feat(c+(1:C)) = ini_total; 
        c = c + C;
    end
    
    featMat(:, i) = feat; 
end
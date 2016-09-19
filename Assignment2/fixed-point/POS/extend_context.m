function W = extend_context(z, M, C)
% the function to extend the contexts.. % column wise, to speed up.  
    Ln = size(z, 2);
    W = zeros(2 * M * C, Ln);
    for i = 1 : Ln
%        W(i, 1 : D) = z(i, 1 : D);
        for k = -M : M
            if (i + k >= 1) && (i + k <= Ln)
                if (k < 0)
                    W((M + k) * C + 1 : (M + k) * C + C, i) = 1*z(1: C, i + k);
                elseif (k > 0)
                    W((M + k - 1) * C + 1 : (M + k - 1) * C + C, i) = 1*z(1 : C, i + k);
                end
            end
        end
    end
end
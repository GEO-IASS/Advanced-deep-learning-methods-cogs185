function W = extend_context_ocr(z, M, C)
% extend_context for OCR task
    Ln = size(z, 1);
    W = zeros(Ln, 2 * M * C);
    for i = 1 : Ln
        for k = -M : M
            if (i + k >= 1) && (i + k <= Ln)
                if (k < 0)
                    W(i,   (M + k) * C + 1 :   (M + k) * C + C) = z(i + k,   1:   C);
                elseif (k > 0)
                    W(i,   (M + k - 1) * C + 1 :   (M + k - 1) * C + C) = z(i + k,   1 :   C);
                end
            end
        end
    end
end
function T_new = transform_TFT(T_old, M1, M2, M3, inverse)

    if nargin < 5
        inverse = 0;
    end

    if inverse == 0
        M1i = inv(M1); T_new = zeros(3, 3, 3);
        T_new(:, :, 1) = M2 * (M1i(1, 1) * T_old(:, :, 1) + M1i(2, 1) * T_old(:, :, 2) + M1i(3, 1) * T_old(:, :, 3)) * M3.';
        T_new(:, :, 2) = M2 * (M1i(1, 2) * T_old(:, :, 1) + M1i(2, 2) * T_old(:, :, 2) + M1i(3, 2) * T_old(:, :, 3)) * M3.';
        T_new(:, :, 3) = M2 * (M1i(1, 3) * T_old(:, :, 1) + M1i(2, 3) * T_old(:, :, 2) + M1i(3, 3) * T_old(:, :, 3)) * M3.';

    elseif inverse == 1
        M2i = inv(M2); M3i = inv(M3); T_new = zeros(3, 3, 3);
        T_new(:, :, 1) = M2i * (M1(1, 1) * T_old(:, :, 1) + M1(2, 1) * T_old(:, :, 2) + M1(3, 1) * T_old(:, :, 3)) * M3i.';
        T_new(:, :, 2) = M2i * (M1(1, 2) * T_old(:, :, 1) + M1(2, 2) * T_old(:, :, 2) + M1(3, 2) * T_old(:, :, 3)) * M3i.';
        T_new(:, :, 3) = M2i * (M1(1, 3) * T_old(:, :, 1) + M1(2, 3) * T_old(:, :, 2) + M1(3, 3) * T_old(:, :, 3)) * M3i.';
    end

    T_new = T_new / norm(T_new(:));

end

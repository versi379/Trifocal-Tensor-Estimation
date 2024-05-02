% This function computes a 3x3x3 tensor based on input matrices.
% It iterates over the third dimension of the tensor, and for each index,
% computes a 3x3 matrix using matrix multiplications and additions.
% The computed matrices are stored in the corresponding slices of the tensor.

function T = ComputeTensorfromMatrices(T0, U, V, W)
    T = zeros(3, 3, 3);

    for i = 1:3
        T(:, :, i) = V' * (U(1, i) * T0(:, :, 1) +U(2, i) * T0(:, :, 2) +U(3, i) * T0(:, :, 3)) * W;
    end

end
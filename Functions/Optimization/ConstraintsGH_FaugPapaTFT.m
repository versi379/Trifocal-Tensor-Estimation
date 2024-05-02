% This function implements constraints and parameters for the optimization
% of the Faugeras-Papadopopoulo's parameterization with Gauss-Helmert.

function [f, g, A, B, C, D] = ConstraintsGH_FaugPapaTFT(obs, x, ~)

    T = reshape(x, 3, 3, 3); % tensor
    obs = reshape(obs, 6, []); % observations
    N = size(obs, 2);

    f = zeros(4 * N, 1); % constraints for tensor and observations (trilinearities)
    A = zeros(4 * N, 27); % jacobian of f w.r.t. the tensor T
    B = zeros(4 * N, 6 * N); % jacobian of f w.r.t. the observations

    for i = 1:N
        % points in the three images for correspondance i
        x1 = obs(1:2, i); x2 = obs(3:4, i); x3 = obs(5:6, i);

        % 4 trilinearities
        ind2 = 4 * (i - 1);
        S2 = [0 -1; -1 0; x2(2) x2(1)];
        S3 = [0 -1; -1 0; x3(2) x3(1)];
        f(ind2 + 1:ind2 + 4) = reshape(S2.' * (x1(1) * T(:, :, 1) + x1(2) * T(:, :, 2) + T(:, :, 3)) * S3, 4, 1);

        % Jacobians for the trilinearities
        A(ind2 + 1:ind2 + 4, :) = kron(S3, S2).' * kron([x1; 1].', eye(9));
        B(ind2 + 1:ind2 + 4, 6 * (i - 1) + 1) = reshape(S2.' * T(:, :, 1) * S3, 4, 1);
        B(ind2 + 1:ind2 + 4, 6 * (i - 1) + 2) = reshape(S2.' * T(:, :, 2) * S3, 4, 1);
        B(ind2 + 1:ind2 + 4, 6 * (i - 1) + (3:4)) = kron(S3.' * reshape(T(3, :, :), 3, 3) * [x1; 1], [0, 1; 1, 0]);
        B(ind2 + 1:ind2 + 4, 6 * (i - 1) + (5:6)) = kron([0, 1; 1, 0], S2.' * reshape(T(:, 3, :), 3, 3) * [x1; 1]);
    end

    g = zeros(12, 1); % constraints on the parameters of T
    C = zeros(12, 27); % jacobian of g w.r.t. the parameters of T
    D = zeros(12, 0);

    for i = 1:3
        g(i, :) = det(T(:, :, i));

        for j = 1:3

            for k = 1:3
                C(i, j + 3 * (k - 1) + 9 * (i - 1)) = minor(T(:, :, i), j, k);
            end

        end

    end

    i = 0;

    for k2 = 1:2

        for k3 = 1:2

            for l2 = k2 + 1:3

                for l3 = k3 + 1:3
                    i = i + 1;
                    A1 = reshape([T(k2, k3, :), T(k2, l3, :), T(l2, l3, :)], 3, 3);
                    A2 = reshape([T(k2, k3, :), T(l2, k3, :), T(l2, l3, :)], 3, 3);
                    A3 = reshape([T(l2, k3, :), T(k2, l3, :), T(l2, l3, :)], 3, 3);
                    A4 = reshape([T(k2, k3, :), T(l2, k3, :), T(k2, l3, :)], 3, 3);
                    g(3 +i, 1) = det(A1) * det(A2) - det(A3) * det(A4);

                    for i1 = 1:3
                        C(3 + i, k2 + 3 * (k3 - 1) + 9 * (i1 - 1)) = minor(A1, i1, 1) * det(A2) + ...
                            det(A1) * minor(A2, i1, 1) - det(A3) * minor(A4, i1, 1);
                        C(3 + i, k2 + 3 * (l3 - 1) + 9 * (i1 - 1)) = minor(A1, i1, 2) * det(A2) - ...
                            minor(A3, i1, 2) * det(A4) - det(A3) * minor(A4, i1, 3);
                        C(3 + i, l2 + 3 * (l3 - 1) + 9 * (i1 - 1)) = minor(A1, i1, 3) * det(A2) + ...
                            det(A1) * minor(A2, i1, 3) - minor(A3, i1, 3) * det(A4);
                        C(3 + i, l2 + 3 * (k3 - 1) + 9 * (i1 - 1)) = det(A1) * minor(A2, i1, 2) - ...
                            minor(A3, i1, 1) * det(A4) - det(A3) * minor(A4, i1, 2);
                    end

                end

            end

        end

    end

end

%%% Computes a minor of a matrix A given the row and column indexes
function m = minor(A, i, j)
    [h, w] = size(A);
    m = det(A([1:i - 1, i + 1:h], [1:j - 1, j + 1:w])) * (-1) ^ (i + j);
end
function [R_t_2, R_t_3, Reconst, T, iter] = FaugPapaTFTPoseEst(Corresp, CalM)

    % Normalization of the data
    [x1, Normal1] = Normalize2DPoints(Corresp(1:2, :));
    [x2, Normal2] = Normalize2DPoints(Corresp(3:4, :));
    [x3, Normal3] = Normalize2DPoints(Corresp(5:6, :));

    % Model to estimate T: linear equations
    [T, P1, P2, P3] = LinearTFT(x1, x2, x3);

    % compute 3d estimated points to have initial estimated reprojected image
    % points
    points3D = Triangulate3DPoints({P1, P2, P3}, [x1; x2; x3]);
    p1_est = P1 * points3D; p1_est = p1_est(1:2, :) ./ repmat(p1_est(3, :), 2, 1);
    p2_est = P2 * points3D; p2_est = p2_est(1:2, :) ./ repmat(p2_est(3, :), 2, 1);
    p3_est = P3 * points3D; p3_est = p3_est(1:2, :) ./ repmat(p3_est(3, :), 2, 1);

    % minimize reprojection error with Gauss-Helmert
    N = size(x1, 2);
    param0 = T(:);
    obs = reshape([x1(1:2, :); x2(1:2, :); x3(1:2, :)], 6 * N, 1);
    obs_est = reshape([p1_est; p2_est; p3_est], 6 * N, 1);
    y = zeros(0, 1);
    [~, param, ~, iter] = GaussHelmert(@constrGH, obs_est, param0, y, obs, eye(6 * N));
    T = reshape(param, 3, 3, 3);

    % denormalization
    T = TransformTFT(T, Normal1, Normal2, Normal3, 1);

    % Find orientation using calibration and TFT
    [R_t_2, R_t_3] = PoseEstfromTFT(T, CalM, Corresp);

    % Find 3D points by triangulation
    Reconst = Triangulate3DPoints({CalM(1:3, :) * eye(3, 4), CalM(4:6, :) * R_t_2, CalM(7:9, :) * R_t_3}, Corresp);
    Reconst = Reconst(1:3, :) ./ repmat(Reconst(4, :), 3, 1);

end

%%% function with GH constraints and parameters for Faugeras-Papadopopoulo's
%%% parameterization
function [f, g, A, B, C, D] = constrGH(obs, x, ~)

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

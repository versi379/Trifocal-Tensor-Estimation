% Description:
% This function computes the FM from corresponding points in two images
% using Gauss-Helmert optimization, initialized by the linear solution.
%
% Input:
% p1: 3xN (homogeneous) or 2xN (cartesian) matrix
%        of N image points in image 1
% p2: 3xN (homogeneous) or 2xN (cartesian) matrix
%        of N image points in image 2
%
% Output:
% F: 3x3 Fundamental Matrix (FM)
% iter: number of iterations needed in GH algorithm to reach minimum

function [F, iter] = OptimalFM(p1, p2)

    % Number of correspondence points is computed as
    % number of columns in either matrix p1 or p2
    N = size(p1, 2);

    % Same number of image points check
    if N ~= size(p2, 2)
        error('Number of points in image 1 and image 2 must be equal.');
    end

    % Minimum number of correspondences check
    if N < 8
        error('At least 8 correspondence points are necessary.');
    end

    % Homogeneous to cartesian coordinates
    if size(p1, 1) == 3
        p1 = p1(1:2, :) ./ repmat(p1(3, :), 2, 1);
        p2 = p2(1:2, :) ./ repmat(p2(3, :), 2, 1);
    end

    % Normalize image points
    [x1, Normal1] = Normalize2DPoints(p1(1:2, :));
    [x2, Normal2] = Normalize2DPoints(p2(1:2, :));

    % --- OPTIMIZED GAUSS-HELMERT ALGORITHM ---

    % Initial FM (linear) estimate
    F = LinearFM(x1, x2); F = F / sqrt(sum(F(1:9) .^ 2));

    % Projection matrices from FM and 3D points
    [U, ~, ~] = svd(F); epi21 = U(:, 3);
    P1 = eye(3, 4);
    P2 = [CrossProdMatrix(epi21) * F epi21];
    points3D = Triangulate3DPoints({P1, P2}, [x1; x2]);

    % Refinement using GH on FM parameters
    p1_est = P1 * points3D; p1_est = p1_est(1:2, :) ./ repmat(p1_est(3, :), 2, 1);
    p2_est = P2 * points3D; p2_est = p2_est(1:2, :) ./ repmat(p2_est(3, :), 2, 1);
    p = reshape(F, 9, 1);
    x = reshape([x1(1:2, :); x2(1:2, :)], 4 * N, 1);
    x_est = reshape([p1_est; p2_est], 4 * N, 1);
    y = zeros(0, 1);
    P = eye(4 * N);
    [~, p_opt, ~, iter] = GaussHelmert(@constraintsGH_F, x_est, p, y, x, P);

    % Recover parameters
    F = reshape(p_opt, 3, 3);

    % Denormalization: transform FM back to original space
    F = Normal2.' * F * Normal1;

    % Constraint enforcement: singularity constraint
    [U, D, V] = svd(F); D(3, 3) = 0;
    F = U * D * V.';

end

% This function implements constraints and parameters for the optimization
% of the FM with Gauss-Helmert.
function [f, g, A, B, C, D] = constraintsGH_F(x, p, ~)

    N = size(x, 1) / 4;
    x = reshape(x, 4, N);

    F = reshape(p, 3, 3);

    g = [det(F); sum(F(1:9) .^ 2) - 1];

    C = [F(5) * F(9) - F(6) * F(8), F(6) * F(7) - F(4) * F(9), F(4) * F(8) - F(5) * F(7), ...
             F(3) * F(8) - F(2) * F(9), F(1) * F(9) - F(3) * F(7), F(2) * F(7) - F(1) * F(8), ...
             F(2) * F(6) - F(3) * F(5), F(3) * F(4) - F(1) * F(6), F(1) * F(5) - F(2) * F(4);
         2 * reshape(F, 9, 1).'];

    f = zeros(N, 1);
    A = zeros(N, 9);
    B = zeros(N, 4 * N);

    for i = 1:N
        x1 = [x(1:2, i); 1]; x2 = [x(3:4, i); 1];
        f(i, :) = x2.' * F * x1;
        A(i, :) = [x1(1) * x2(1), x1(1) * x2(2), x1(1), x1(2) * x2(1), x1(2) * x2(2), x1(2), x2(1), x2(2), 1];
        B(i, 4 * (i - 1) + 1:4 * (i - 1) + 4) = [F(3) + F(1) * x2(1) + F(2) * x2(2), F(6) + F(4) * x2(1) + F(5) * x2(2), ...
                                                     F(7) + F(1) * x1(1) + F(4) * x1(2), F(8) + F(2) * x1(1) + F(5) * x1(2)];
    end

    D = zeros(2, 0);

end

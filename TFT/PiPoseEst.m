function [R_t_2, R_t_3, Reconst, T, iter] = PiPoseEst(Corresp, CalM)

    % Number of correspondences
    N = size(Corresp, 2);

    % Normalization of the data
    [x1, Normal1] = Normalize2DPoints(Corresp(1:2, :));
    [x2, Normal2] = Normalize2DPoints(Corresp(3:4, :));
    [x3, Normal3] = Normalize2DPoints(Corresp(5:6, :));

    % First approximation of T: linear equations
    [~, P1, P2, P3] = LinearTFT(x1, x2, x3);

    % find homography H sending camera centers to fundamental points
    M = [null(P1), null(P2), null(P3)];
    M = [M, null(M.')];
    P1 = P1 * M; P2 = P2 * M; P3 = P3 * M;

    % find Pi matrices
    Pi1 = inv(P1(:, 2:4)); Pi2 = inv(P2(:, [1 3 4])); Pi3 = inv(P3(:, [1 2 4]));
    Pi1 = [0 0 0; Pi1];
    Pi2 = [Pi2(1, :); 0 0 0; Pi2(2:3, :)];
    Pi3 = [Pi3(1:2, :); 0 0 0; Pi3(3, :)];

    % minimal parameterization
    Pi1 = Pi1 ./ (norm(Pi1(4, :))); Pi2 = Pi2 ./ (norm(Pi2(4, :))); Pi3 = Pi3 ./ (norm(Pi3(4, :)));
    Q = eye(4);
    Q(1, 1) = 1 ./ norm(Pi3(1, :) - dot(Pi3(1, :), Pi3(4, :)) * Pi3(4, :)); Q(1, 4) = -Q(1, 1) * dot(Pi3(1, :), Pi3(4, :));
    Q(2, 2) = 1 ./ norm(Pi1(2, :) - dot(Pi1(2, :), Pi1(4, :)) * Pi1(4, :)); Q(2, 4) = -Q(2, 2) * dot(Pi1(2, :), Pi1(4, :));
    Q(3, 3) = 1 ./ norm(Pi2(3, :) - dot(Pi2(3, :), Pi2(4, :)) * Pi2(4, :)); Q(3, 4) = -Q(3, 3) * dot(Pi2(3, :), Pi2(4, :));
    Pi1 = Q * Pi1; Pi2 = Q * Pi2; Pi3 = Q * Pi3;

    P1 = P1 * inv(Q); P2 = P2 * inv(Q); P3 = P3 * inv(Q);
    points3D = Triangulate3DPoints({P1, P2, P3}, [x1; x2; x3]);
    p1_est = P1 * points3D; p1_est = p1_est(1:2, :) ./ repmat(p1_est(3, :), 2, 1);
    p2_est = P2 * points3D; p2_est = p2_est(1:2, :) ./ repmat(p2_est(3, :), 2, 1);
    p3_est = P3 * points3D; p3_est = p3_est(1:2, :) ./ repmat(p3_est(3, :), 2, 1);

    % minimize error using Gauss-Helmert
    pi = [reshape(Pi1(2:4, :).', 9, 1); reshape(Pi2([1 3 4], :).', 9, 1); reshape(Pi3([1 2 4], :).', 9, 1)];
    x = reshape([x1; x2; x3], 6 * N, 1);
    x_est = reshape([p1_est; p2_est; p3_est], 6 * N, 1);
    y = zeros(0, 1);
    P = eye(6 * N);
    [~, pi_opt, ~, iter] = GaussHelmert(@ConstraintsGH_PiTFT, x_est, pi, y, x, P);

    % retrieve geometry from optimized parameters
    Pi1 = (reshape(pi_opt(1:9), 3, 3)).';
    Pi2 = (reshape(pi_opt(10:18), 3, 3)).';
    Pi3 = (reshape(pi_opt(19:27), 3, 3)).';
    P1 = zeros(3, 4); P2 = zeros(3, 4); P3 = zeros(3, 4);
    P1(:, 2:4) = inv(Pi1);
    P2(:, [1 3 4]) = inv(Pi2);
    P3(:, [1 2 4]) = inv(Pi3);
    T = TFTfromProj(P1, P2, P3);

    % denormalization
    T = TransformTFT(T, Normal1, Normal2, Normal3, 1);

    % Find orientation using calibration and TFT
    [R_t_2, R_t_3] = PoseEstfromTFT(T, CalM, Corresp);

    % Find 3D points by triangulation
    Reconst = Triangulate3DPoints({CalM(1:3, :) * eye(3, 4), CalM(4:6, :) * R_t_2, CalM(7:9, :) * R_t_3}, Corresp);
    Reconst = Reconst(1:3, :) ./ repmat(Reconst(4, :), 3, 1);

end

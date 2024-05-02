% Description:
% This function estimates the pose of three views based on corresponding
% triplets of points, using the Ponce-Hebert's parameterization of the TFT.
% An initial trifocal tensor is computed linearly from the trilinearities
% using the triplets of correspondences. Then a minimal parameterization is
% computed using the constraints presented in Section X (Report).
% After the optimization the essential matrices are 
% computed from the tensor and the orientations are extracted by SVD.
% 
% Input:
% matchingPoints: 6xN matrix, containing in each column the 3 projections of
%                 the same space point onto the 3 images
% calMatrices: 9x3 matrix containing the 3x3 calibration matrices for
%              each camera concatenated
%
% Output:
% R_t_2: 3x4 matrix containing the rotation matrix and translation
%        vector [R2,t2] for the second camera
% R_t_3: 3x4 matrix containing the rotation matrix and translation
%        vector [R3,t3] for the third camera
% Rec: 3xN matrix containing the 3D Recruction of the
%      correspsondences
% iter: number of iterations needed in GH algorithm to reach minimum

function [R_t_2, R_t_3, Rec, T, iter] = PiPoseEst(matchingPoints, calMatrices)

    % Number of matchingPointsondences
    N = size(matchingPoints, 2);

    % Normalize image points
    [x1, Normal1] = Normalize2DPoints(matchingPoints(1:2, :));
    [x2, Normal2] = Normalize2DPoints(matchingPoints(3:4, :));
    [x3, Normal3] = Normalize2DPoints(matchingPoints(5:6, :));

    % Compute TFT (linear estimation)
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

    % Minimize reprojection error with Gauss-Helmert
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

    % Denormalization: transform TFT back to original space
    T = TransformTFT(T, Normal1, Normal2, Normal3, 1);

    % Find orientation using calibration matrices and TFT
    [R_t_2, R_t_3] = PoseEstfromTFT(T, calMatrices, matchingPoints);

    % Find 3D points by triangulation
    Rec = Triangulate3DPoints({calMatrices(1:3, :) * eye(3, 4), calMatrices(4:6, :) * R_t_2, calMatrices(7:9, :) * R_t_3}, matchingPoints);
    Rec = Rec(1:3, :) ./ repmat(Rec(4, :), 3, 1);

end

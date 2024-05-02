% Description:
% This function estimates the pose of three views based on corresponding
% triplets of points, using linear fundamental matrix estimation.
% The fundamental matrices are derived through algebraic minimization of the
% epipolar equations for two out of the three potential pairs of views.
% Essential matrices are computed using calibration matrices, while orientations are extracted via SVD.
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
% Rec: 3xN matrix containing the 3D reconstruction of the
%      correspondences

function [R_t_2, R_t_3, Rec, T, iter] = LinearFMPoseEst(matchingPoints, calMatrices)

    % Number of correspondence points is computed as
    % number of columns in matrix matchingPoints
    N = size(matchingPoints, 2);

    % Extract calibration matrix for each camera
    K1 = calMatrices(1:3, :); K2 = calMatrices(4:6, :); K3 = calMatrices(7:9, :);

    % Normalize image points
    [x1, Normal1] = Normalize2DPoints(matchingPoints(1:2, :));
    [x2, Normal2] = Normalize2DPoints(matchingPoints(3:4, :));
    [x3, Normal3] = Normalize2DPoints(matchingPoints(5:6, :));

    % Compute FMs
    F21 = LinearFM(x1, x2);
    F31 = LinearFM(x1, x3);

    % Denormalization: transform FMs back to original space
    F21 = Normal2.' * F21 * Normal1;
    F31 = Normal3.' * F31 * Normal1;

    % Find orientation using calibration matrices and FMs
    [R2, t2] = ExtractRTfromFM(F21, K1, K2, matchingPoints(1:2, :), matchingPoints(3:4, :));
    [R3, t3] = ExtractRTfromFM(F31, K1, K3, matchingPoints(1:2, :), matchingPoints(5:6, :));

    % Find the norm of t31 using the image points and reconstruction from images 1 and 2
    u3 = K3 * t3;
    X = Triangulate3DPoints({K1 * eye(3, 4), K2 * [R2, t2]}, matchingPoints(1:4, :));
    X = X(1:3, :) ./ repmat(X(4, :), 3, 1);
    X3 = K3 * R3 * X;
    lam = -sum(dot(cross([matchingPoints(5:6, :); ones(1, N)], X3, 1), cross([matchingPoints(5:6, :); ones(1, N)], repmat(u3, 1, N)), 1)) / ...
        sum(sum(cross([matchingPoints(5:6, :); ones(1, N)], repmat(u3, 1, N)) .^ 2));
    t3 = lam * t3;

    % Matrices containing the rotation matrix and translation
    % vector [Ri,ti] for the second and the third camera
    R_t_2 = [R2, t2]; R_t_3 = [R3, t3];

    % Find 3D points by triangulation
    Rec = Triangulate3DPoints({K1 * eye(3, 4), K2 * R_t_2, K3 * R_t_3}, matchingPoints);
    Rec = Rec(1:3, :) ./ repmat(Rec(4, :), 3, 1);
    iter = 0;
    T = TFTfromProj(K1 * eye(3, 4), K2 * R_t_2, K3 * R_t_3);

end

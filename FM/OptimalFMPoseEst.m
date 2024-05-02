
% Description:
% This function estimates the pose of three views based on corresponding
% triplets of points, using optimized fundamental matrix estimation.
% The fundamental matrices are initialized by the linear solution and
% refined by a Gauss-Helmert minimization of the standard error for
% two of the three possible pairs of views. The essential matrices are 
% computed using the calibration matrices. The orientations are extracted by SVD.
%
% Input:
% matchingPoints: 6xN matrix containing in each column, the 3 projections of
%             the same space point onto the 3 images.
% calMatrices: 9x3 matrix containing the M calibration 3x3 matrices for 
%             each camera concatenated.
%
% Output: 
% R_t_2: 3x4 matrix containing the rotation matrix and translation 
%             vector [R2,t2] for the second camera.
% R_t_3: 3x4 matrix containing the rotation matrix and translation 
%             vector [R3,t3] for the third camera.
% Rec: 3xN matrix containing the 3D Recruction of the
%             matchingPointsondences.
% iter: number of iterations needed in GH algorithm to reach minimum

function [R_t_2, R_t_3, Rec, T, iter] = OptimalFMPoseEst(matchingPoints, calMatrices)

    N = size(matchingPoints, 2);
    K1 = calMatrices(1:3, :); K2 = calMatrices(4:6, :); K3 = calMatrices(7:9, :);

    % the 2 fundamental matrices
    [F21, it1] = OptimalFM(matchingPoints(1:2, :), matchingPoints(3:4, :));
    [F31, it2] = OptimalFM(matchingPoints(1:2, :), matchingPoints(5:6, :));
    iter = it1 + it2;

    % Find orientation using calibration and F matrices
    [R2, t2] = ExtractRTfromFM(F21, K1, K2, matchingPoints(1:2, :), matchingPoints(3:4, :));
    [R3, t3] = ExtractRTfromFM(F31, K1, K3, matchingPoints(1:2, :), matchingPoints(5:6, :));

    % Find the norm of t31 using the image points and Recruction from
    % images 1 and 2
    u3 = K3 * t3;
    X = Triangulate3DPoints({K1 * eye(3, 4), K2 * [R2, t2]}, matchingPoints(1:4, :));
    X = X(1:3, :) ./ repmat(X(4, :), 3, 1);
    X3 = K3 * R3 * X;
    lam = -sum(dot(cross([matchingPoints(5:6, :); ones(1, N)], X3, 1), cross([matchingPoints(5:6, :); ones(1, N)], repmat(u3, 1, N)), 1)) / ...
        sum(sum(cross([matchingPoints(5:6, :); ones(1, N)], repmat(u3, 1, N)) .^ 2));
    t3 = lam * t3;

    R_t_2 = [R2, t2]; R_t_3 = [R3, t3];

    % Find 3D points by triangulation
    Rec = Triangulate3DPoints({K1 * eye(3, 4), K2 * R_t_2, K3 * R_t_3}, matchingPoints);
    Rec = Rec(1:3, :) ./ repmat(Rec(4, :), 3, 1);
    T = TFTfromProj(K1 * eye(3, 4), K2 * R_t_2, K3 * R_t_3);

end

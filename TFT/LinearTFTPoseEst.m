% Description:
% This function estimates the pose of three views based on corresponding
% triplets of points, using algebraic minimization of the linear contraints 
% given by the incidence relationships.
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

function [R_t_2, R_t_3, Rec, T, iter] = LinearTFTPoseEst(matchingPoints, calMatrices)

    % Normalize image points
    [x1, Normal1] = Normalize2DPoints(matchingPoints(1:2, :));
    [x2, Normal2] = Normalize2DPoints(matchingPoints(3:4, :));
    [x3, Normal3] = Normalize2DPoints(matchingPoints(5:6, :));

    % Compute TFT (linear estimation)
    T = LinearTFT(x1, x2, x3);

    % Denormalization: transform TFT back to original space
    T = TransformTFT(T, Normal1, Normal2, Normal3, 1);

    % Find orientation using calibration matrices and TFT
    [R_t_2, R_t_3] = PoseEstfromTFT(T, calMatrices, matchingPoints);

    % Find 3D points by triangulation
    Rec = Triangulate3DPoints({calMatrices(1:3, :) * eye(3, 4), calMatrices(4:6, :) * R_t_2, calMatrices(7:9, :) * R_t_3}, matchingPoints);
    Rec = Rec(1:3, :) ./ repmat(Rec(4, :), 3, 1);

    iter = 0;

end

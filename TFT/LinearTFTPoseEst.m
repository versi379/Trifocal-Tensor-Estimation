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
% iter: number of iterations needed in GH algorithm to reach minimum

function [R_t_2, R_t_3, Reconst, T, iter] = LinearTFTPoseEst(Corresp, CalM)

    % Normalization of the data
    [x1, Normal1] = Normalize2DPoints(Corresp(1:2, :));
    [x2, Normal2] = Normalize2DPoints(Corresp(3:4, :));
    [x3, Normal3] = Normalize2DPoints(Corresp(5:6, :));

    % Model to estimate T: linear equations
    T = LinearTFT(x1, x2, x3);

    % tensor denormalization
    T = TransformTFT(T, Normal1, Normal2, Normal3, 1);

    % Find orientation using calibration and TFT
    [R_t_2, R_t_3] = PoseEstfromTFT(T, CalM, Corresp);

    % Find 3D points by triangulation
    Reconst = Triangulate3DPoints({CalM(1:3, :) * eye(3, 4), CalM(4:6, :) * R_t_2, CalM(7:9, :) * R_t_3}, Corresp);
    Reconst = Reconst(1:3, :) ./ repmat(Reconst(4, :), 3, 1);

    iter = 0;

end

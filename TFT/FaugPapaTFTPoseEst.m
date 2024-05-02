%FAUGPAPATFTPOSEESTIMATION Pose estimation of 3 views from corresponding
% triplets of points using the TriFocal Tensor with minimal constraints
% by T. Papadopoulo and O. Faugeras.
%
%  An initial trifocal tensor is computed linearly from the trilinearities
%  using the triplets of correspondences. Then a minimal parameterization is
%  computed using the constraints presented in "A non linear method for
%  estimatin the projective geometry of three views" by O. Faugeras and
%  T. Papadopoulo. After the optimization the essential matrices are 
%  computed from the tensor and the orientations are extracted by SVD.
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
% iter: number of iterations needed in GH algorithm to reach minimum

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
    [~, param, ~, iter] = GaussHelmert(@ConstraintsGH_FaugPapaTFT, obs_est, param0, y, obs, eye(6 * N));
    T = reshape(param, 3, 3, 3);

    % denormalization
    T = TransformTFT(T, Normal1, Normal2, Normal3, 1);

    % Find orientation using calibration and TFT
    [R_t_2, R_t_3] = PoseEstfromTFT(T, CalM, Corresp);

    % Find 3D points by triangulation
    Reconst = Triangulate3DPoints({CalM(1:3, :) * eye(3, 4), CalM(4:6, :) * R_t_2, CalM(7:9, :) * R_t_3}, Corresp);
    Reconst = Reconst(1:3, :) ./ repmat(Reconst(4, :), 3, 1);

end


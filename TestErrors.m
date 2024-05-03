clear;

%% Add functions to working dir

addpath(genpath(pwd));

%% Generate random data for a triplet of images
N = 100; % Number of 3D points
noise = 1; % Sigma for the added Gaussian noise in pixels
seed = 1; % Seed for random generation
f = 50; % Focal length in mm
angle = 0; % Default angle (no collinearity of camera centers)
[calMatrices, R_t0, matchingPoints, points3D] = GenerateSyntheticScene(N, noise, seed, f, angle);

%% Select method to test
method = { ...
              @LinearTFTPoseEst, ... % 1) TFT - Linear Estimation
              @ResslTFTPoseEst, ... % 2) TFT - Ressl Estimation
              @NordbergTFTPoseEst, ... % 3) TFT - Nordberg Estimation
              @FaugPapaTFTPoseEst, ... % 4) TFT - Faugeras-Papadopoulo Estimation
              @PiPoseEst, ... % 5) TFT - Ponce-Hebert Estimation
              @PiColPoseEst, ... % 6) TFT - Ponce-Hebert (collinear cameras) Estimation
              @LinearFMPoseEstimation, ... % 7) FM - Linear Estimation
              @OptimalFMPoseEst}; % 8) FM - Optimized Estimation

[R_t_2, R_t_3, Rec] = method{1}(matchingPoints, calMatrices);

%% Compute errors
% Reprojection error
repr_err = ReprError({calMatrices(1:3, :) * eye(3, 4), calMatrices(4:6, :) * R_t_2, calMatrices(7:9, :) * R_t_3}, ...
    matchingPoints, Rec);
fprintf('Reprojection error is %f .\n', repr_err);

% Angular errors
[rot2_err, t2_err] = AngErrors(R_t0{1}, R_t_2);
[rot3_err, t3_err] = AngErrors(R_t0{2}, R_t_3);

fprintf('Angular errors in rotations are %f and %f, and in translations are %f and %f .\n', ...
    rot2_err, rot3_err, t2_err, t3_err);

%% Apply Bundle Adjustment
[R_t_ref, Rec_ref, iter, repr_err] = BundleAdjustment(calMatrices, [eye(3, 4); R_t_2; R_t_3], matchingPoints, Rec);
fprintf('Reprojection error is %f after Bundle Adjustment.\n', repr_err);

% Optimized angular errors
[rot2_err, t2_err] = AngErrors(R_t0{1}, R_t_ref(4:6, :));
[rot3_err, t3_err] = AngErrors(R_t0{2}, R_t_ref(7:9, :));
fprintf('Angular errors in rotations after BA are %f and %f, and in translations are %f and %f .\n', ...
    rot2_err, rot3_err, t2_err, t3_err);

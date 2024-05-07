clear; close all;

%% Add functions to working dir
addpath(genpath(pwd));

%% Dataset
% dataset = 'fountain-P11';
% dataset = 'Herz-Jesu-P8';
% dataset = 'entry-P10'; % .mat file to be added

%% Some parameters
path_to_data = strcat('Dataset/', dataset, '/cameras/');

switch dataset
    case 'fountain-P11'
        triplets_to_test = 1:70;
    case 'Herz-Jesu-P8'
        triplets_to_test = 1:50;
    case ''
end

initial_sample_size = 100;
bundle_adj_size = 50;
repr_err_th = 1;

%% Recover correspondances
corresp_file = matfile(strcat(path_to_data, 'Corresp_triplets', '.mat'));
indexes_sorted = corresp_file.indexes_sorted;
corresp_by_triplet = corresp_file.Corresp;
im_names = corresp_file.im_names;
clear corresp_file;

%% Method to test
methods = { ...
               @LinearTFTPoseEst, ... % 1) TFT - Linear Estimation
               @ResslTFTPoseEst, ... % 2) TFT - Ressl Estimation
               @NordbergTFTPoseEst, ... % 3) TFT - Nordberg Estimation
               @FaugPapaTFTPoseEst, ... % 4) TFT - Faugeras-Papadopoulo Estimation
               @PiPoseEst, ... % 5) TFT - Ponce-Hebert Estimation
               @PiColPoseEst, ... % 6) TFT - Ponce-Hebert (collinear cameras) Estimation
               @LinearFMPoseEst, ... % 7) FM - Linear Estimation
               @OptimalFMPoseEst}; % 8) FM - Optimized Estimation

methods_to_test = [1:5, 7:8]; % Method 6 is not tested

%% Vectors to be measured
repr_err = zeros(length(triplets_to_test), length(methods), 2);
rot_err = zeros(length(triplets_to_test), length(methods), 2);
t_err = zeros(length(triplets_to_test), length(methods), 2);
iter = zeros(length(triplets_to_test), length(methods), 2);
time = zeros(length(triplets_to_test), length(methods), 2);

%% Iterate through tiplets of images to test
for it = 1:length(triplets_to_test)

    % Extract information about images and correspondences in the triplet
    triplet = indexes_sorted(triplets_to_test(it), 1:3);
    im1 = triplet(1);
    im2 = triplet(2);
    im3 = triplet(3);
    Corresp = corresp_by_triplet{im1, im2, im3}';
    N = size(Corresp, 2);
    fprintf('Triplet %d/%d (%d,%d,%d) with %d matching points.\n', ...
        it, length(triplets_to_test), im1, im2, im3, N);

    % Extract ground truth camera poses and calibration matrices for each image in the triplet
    [K1, R1_true, t1_true, im_size] = ExtractCalibOrient(path_to_data, im_names{im1});
    [K2, R2_true, t2_true] = ExtractCalibOrient(path_to_data, im_names{im2});
    [K3, R3_true, t3_true] = ExtractCalibOrient(path_to_data, im_names{im3});
    calMatrices = [K1; K2; K3];
    R_t0 = {[R2_true * R1_true.', t2_true - R2_true * R1_true.' * t1_true], ...
                [R3_true * R1_true.', t3_true - R3_true * R1_true.' * t1_true]};

    % Remove noisy correspondences (reprojection error > 1 pixel)
    Rec0 = Triangulate3DPoints({K1 * eye(3, 4), K2 * R_t0{1}, K3 * R_t0{2}}, Corresp);
    Rec0 = bsxfun(@rdivide, Rec0(1:3, :), Rec0(4, :));
    Corresp_new = Project3DPoints(Rec0, {K1 * eye(3, 4), K2 * R_t0{1}, K3 * R_t0{2}});
    residuals = Corresp_new - Corresp; % Reprojection error
    Corresp_inliers = Corresp(:, sum(abs(residuals) > repr_err_th, 1) == 0);
    N = size(Corresp_inliers, 2);
    REr = ReprError({K1 * eye(3, 4), K2 * R_t0{1}, K3 * R_t0{2}}, Corresp_inliers);
    fprintf('%d valid correspondances with reprojection error %f.\n', N, REr);

    % Random samples drawn from remaining correspondences
    rng(it);
    init_sample = randsample(1:N, min(initial_sample_size, N));
    rng(it);
    ref_sample = randsample(init_sample, min(bundle_adj_size, length(init_sample)));
    Corresp_init = Corresp_inliers(:, init_sample);
    Corresp_ref = Corresp_inliers(:, ref_sample);

    % Iterate to reproduce different estimation methods implemented
    for m = methods_to_test

        fprintf('Method %d ', m);

        % Check minimum number of correspondences
        if (m > 6 && N < 8) || N < 7
            repr_err(i, m, :) = inf;
            rot_err(i, m, :) = inf;
            t_err(i, m, :) = inf;
            iter(i, m, :) = inf;
            time(i, m, :) = inf;
            continue;
        end

        % Perform pose estimation with method m
        t0 = cputime;
        [R_t_2, R_t_3, ~, ~, nit] = methods{m}(Corresp_init, calMatrices);
        t = cputime - t0;

        % Compute reprojection error
        repr_err(it, m, 1) = ReprError({calMatrices(1:3, :) * eye(3, 4), ...
                                            calMatrices(4:6, :) * R_t_2, calMatrices(7:9, :) * R_t_3}, Corresp_inliers);

        % Compute angular errors (rotation and translation)
        [rot2_err, t2_err] = AngErrors(R_t0{1}, R_t_2);
        [rot3_err, t3_err] = AngErrors(R_t0{2}, R_t_3);
        rot_err(it, m, 1) = (rot2_err + rot3_err) / 2;
        t_err(it, m, 1) = (t2_err + t3_err) / 2;

        % Compute number of iterations and time
        iter(it, m, 1) = nit; time(it, m, 1) = t;

        % Apply Bundle Adjustment
        fprintf('(ref)... ');
        t0 = cputime;
        [R_t_ref, ~, nit, repr_errBA] = BundleAdjustment(calMatrices, ...
            [eye(3, 4); R_t_2; R_t_3], Corresp_ref);
        t = cputime - t0;

        % Compute reprojection error
        repr_err(it, m, 2) = ReprError({calMatrices(1:3, :) * R_t_ref(1:3, :), ...
                                            calMatrices(4:6, :) * R_t_ref(4:6, :), ...
                                            calMatrices(7:9, :) * R_t_ref(7:9, :)}, Corresp_inliers);

        % Compute angular errors (rotation and translation)
        [rot2_err, t2_err] = AngErrors(R_t0{1}, R_t_ref(4:6, :));
        [rot3_err, t3_err] = AngErrors(R_t0{2}, R_t_ref(7:9, :));
        rot_err(it, m, 2) = (rot2_err + rot3_err) / 2;
        t_err(it, m, 2) = (t2_err + t3_err) / 2;

        % Compute number of iterations and time
        iter(it, m, 2) = nit; time(it, m, 2) = t;

    end

    fprintf('\n');

end

%% Means
% Means of metrics (reprojection error, angular errors, number of iterations, time)
% across all triplets are calculated for each method
means_all = zeros(8, 5, 2);

for m = methods_to_test
    % Initial means
    means_all(m, :, 1) = [mean(repr_err(:, m, 1)), mean(rot_err(:, m, 1)), ...
                              mean(t_err(:, m, 1)), mean(iter(:, m, 1)), mean(time(:, m, 1))];
    % BA means
    means_all(m, :, 2) = [mean(repr_err(:, m, 2)), mean(rot_err(:, m, 2)), ...
                              mean(t_err(:, m, 2)), mean(iter(:, m, 2)), mean(time(:, m, 2))];
end

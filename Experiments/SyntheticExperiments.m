clear; close all;

%% Add functions to working dir
addpath(genpath(pwd));

%% Variable to test
% option = 'noise'; % To vary noise
% option = 'focal';  % To vary focal length
% option = 'points'; % To vary number of points
% option = 'angle';  % To vary angle

%% Initial parameters
N = 10; % Number of 3D points
noise = 1; % Sigma for the added Gaussian noise in pixels
f = 50; % Focal length in mm
angle = 0; % Angle among three camera centers (default: no collinearity)
n_sim = 30; % Number of simulations to be performed

%% Option varying intervals
switch option
    case 'noise'
        interval = 0:0.25:3;
    case 'focal'
        interval = 20:20:300;
    case 'points'
        interval = [7:9, 10:5:25];
    case 'angle'
        interval = [166:2:174, 175:179, 179.5, 180];
end

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

if strcmp(option, 'angle')
    methods_to_test = 1:8;
else
    methods_to_test = [1:5, 7:8]; % Method 6 is not tested
end

%% Vectors to be measured
repr_err = zeros(length(interval), length(methods), 2); % Reprojection error
rot_err = zeros(length(interval), length(methods), 2); % Rotation error
t_err = zeros(length(interval), length(methods), 2); % Translation error
iter = zeros(length(interval), length(methods), 2); % Number of iterations
time = zeros(length(interval), length(methods), 2); % Time

%% Iterate to reproduce different "option" values in relative interval
for i = 1:length(interval)

    switch option
        case 'noise'
            noise = interval(i);
            fprintf('Noise = %fpix\n', noise);
        case 'focal'
            f = interval(i);
            fprintf('Focal length = %dmm\n', f);
        case 'points'
            N = interval(i);
            fprintf('Number of points = %d\n', N);
        case 'angle'
            angle = interval(i);
            fprintf('Angle among camera centers = %f\n', angle);
    end

    % Fixed a certain "option" value,
    % iterate to reproduce a given number of simulations
    for it = 1:n_sim

        % Generate random data for a triplet of images
        [calMatrices, R_t0, matchingPoints] = GenerateSyntheticScene(N + 100, noise, it, f, angle);
        rng(it);
        matchingPoints = matchingPoints(:, randsample(N + 100, N));

        % Iterate to reproduce different estimation methods implemented
        for m = methods_to_test

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
            [R_t_2, R_t_3, Rec, ~, nit] = methods{m}(matchingPoints, calMatrices);
            t = cputime - t0;

            % Compute reprojection error
            repr_err(i, m, 1) = repr_err(i, m, 1) + ...
                ReprError({calMatrices(1:3, :) * eye(3, 4), ...
                           calMatrices(4:6, :) * R_t_2, calMatrices(7:9, :) * R_t_3}, matchingPoints, Rec) / n_sim;

            % Compute angular errors (rotation and translation)
            [rot2_err, t2_err] = AngErrors(R_t0{1}, R_t_2);
            [rot3_err, t3_err] = AngErrors(R_t0{2}, R_t_3);
            rot_err(i, m, 1) = rot_err(i, m, 1) + (rot2_err + rot3_err) / (2 * n_sim);
            t_err(i, m, 1) = t_err(i, m, 1) + (t2_err + t3_err) / (2 * n_sim);

            % Compute number of iterations and time
            iter(i, m, 1) = iter(i, m, 1) + nit / n_sim;
            time(i, m, 1) = time(i, m, 1) + t / n_sim;

            % Apply Bundle Adjustment
            t0 = cputime;
            [R_t_ref, ~, nit, repr_errBA] = BundleAdjustment(calMatrices, ...
                [eye(3, 4); R_t_2; R_t_3], matchingPoints, Rec);
            t = cputime - t0;

            % Compute reprojection error
            repr_err(i, m, 2) = repr_err(i, m, 2) + repr_errBA / n_sim;

            % Compute angular errors (rotation and translation)
            [rot2_err, t2_err] = AngErrors(R_t0{1}, R_t_ref(4:6, :));
            [rot3_err, t3_err] = AngErrors(R_t0{2}, R_t_ref(7:9, :));
            rot_err(i, m, 2) = rot_err(i, m, 2) + (rot2_err + rot3_err) / (2 * n_sim);
            t_err(i, m, 2) = t_err(i, m, 2) + (t2_err + t3_err) / (2 * n_sim);

            % Compute number of iterations and time
            iter(i, m, 2) = iter(i, m, 2) + nit / n_sim;
            time(i, m, 2) = time(i, m, 2) + t / n_sim;

        end

    end

end

%% Plot results
methods_to_plot = methods_to_test;
method_names = {'Linear TFT', 'Ressl TFT', 'Nordberg TFT', 'Faugeras-Papadopoulo TFT', 'Ponce-Hebert TFT', ...
                    'Ponce-Hebert (collinear cameras) TFT', 'Linear FM', 'Optimized FM', 'Bundle Adjustment'};

%% Initial plots
figure('Units', 'inches', ...
       'Position', [0, 0, 6.875, 8.875], 'Name', 'Initial Results')

tiledlayout(3, 2);

% Reprojection error plot
nexttile
plot(interval, repr_err(:, methods_to_plot, 1))
xlabel(option)
ylabel('error (pixels)')
title('Initial Reprojection Error')

% Rotation error plot
nexttile
plot(interval, rot_err(:, methods_to_plot, 1))
xlabel(option)
ylabel('error (°)')
title('Initial Rotation Error')

% Translation error plot
nexttile
plot(interval, t_err(:, methods_to_plot, 1))
xlabel(option)
ylabel('error (°)')
title('Initial Translation Error')

% Number of iterations plot
nexttile
plot(interval, iter(:, methods_to_plot, 1))
xlabel(option)
ylabel('# iterations')
title('Initial Number of Iterations')

% Time plot
nexttile
plot(interval, time(:, methods_to_plot, 1))
xlabel(option)
ylabel('time (s)')
title('Initial Time')

% Legend
leg = legend(method_names(methods_to_plot), 'Location', 'Best');
leg.Layout.Tile = 6;

saveas(gcf, strcat('Experiments/Synthetic/', option, '/INIT', option, 'Plots.png'), 'png');

%% Bundle Adjustment plots
figure('Units', 'inches', ...
       'Position', [0, 0, 6.875, 8.875], 'Name', 'Results after Bundle Adjustment')

tiledlayout(3, 2);

% Reprojection error plot
nexttile
plot(interval, repr_err(:, methods_to_plot, 2))
xlabel(option)
ylabel('error (pixels)')
title('BA Reprojection Error')

% Rotation error plot
nexttile
plot(interval, rot_err(:, methods_to_plot, 2))
xlabel(option)
ylabel('error (°)')
title('BA Rotation Error')

% Translation error plot
nexttile
plot(interval, t_err(:, methods_to_plot, 2))
xlabel(option)
ylabel('error (°)')
title('BA Translation Error')

% Number of iterations plot
nexttile
plot(interval, iter(:, methods_to_plot, 2))
xlabel(option)
ylabel('# iterations')
title('BA Number of Iterations')

% Time plot
nexttile
plot(interval, time(:, methods_to_plot, 2))
xlabel(option)
ylabel('time (s)')
title('BA Time')

% Legend
leg = legend(method_names(methods_to_plot), 'Location', 'Best');
leg.Layout.Tile = 6;

saveas(gcf, strcat('Experiments/Synthetic/', option, '/BA', option, 'Plots.png'), 'png');

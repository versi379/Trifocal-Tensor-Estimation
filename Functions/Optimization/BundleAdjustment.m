% This function performs the Bundle Adjustment for the pose estimation of M cameras and N 3D points.
% The reprojection error of the N points to the M cameras is minimized over
% the possible positions of the space points and orientations of the
% cameras. The points do not need to be seen in all cameras but at least in
% two. An initial guess for the orientations of the cameras is needed while
% an initial triangulation of the space points is optional. The
% oprimization is carried out by the Levenberg-Marquardt algorithm. The
% rotations are parametrized by three angles each.

function [R_t, Rec, iter, repr_err] = BundleAdjustment(calMatrices, R_t_0, matchingPoints, Rec0)

    M = size(matchingPoints, 1) / 2; % Number of images/cameras
    N = size(matchingPoints, 2); % Number of 3D points to recover

    % Normalize image points
    for j = 1:M
        [new_Corr, Normal] = Normalize2DPoints(matchingPoints(2 * j - 1:2 * j, :));
        matchingPoints(2 * j - 1:2 * j, :) = new_Corr;
        calMatrices(3 * j - 2:3 * j, :) = Normal * calMatrices(3 * j - 2:3 * j, :);
    end

    % Compute triangulation of 3D points
    if nargin < 4
        Rec0 = zeros(3, N);

        for i = 1:N
            points = [];
            cameras = {};
            aux = 0;

            for j = 1:M

                if isnan(matchingPoints(2 * j - 1, i))
                    continue;
                end

                aux = aux + 1;
                points = [points; matchingPoints(2 * j - 1:2 * j, i)];
                cameras = {cameras{1:(aux - 1)}, ...
                                       calMatrices(3 * j - 2:3 * j, :) * R_t_0(3 * j - 2:3 * j, :)};
            end

            X = Triangulate3DPoints(cameras, points);
            Rec0(:, i) = X(1:3) / X(4);
        end

    end

    % Change coordinates so that the first pose is [ I 0 ]
    change_coord = R_t_0(1:3, :);
    R_t_0(1:3, :) = eye(3, 4);

    for j = 2:M
        R_t_0(3 * j - 2:3 * j, 4) = R_t_0(3 * j - 2:3 * j, 4) -R_t_0(3 * j - 2:3 * j, 1:3) * change_coord(:, 1:3).' * change_coord(:, 4);
        R_t_0(3 * j - 2:3 * j, 1:3) = R_t_0(3 * j - 2:3 * j, 1:3) * change_coord(:, 1:3).';
    end

    Rec0 = change_coord(:, 1:3) * Rec0 + repmat(change_coord(:, 4), 1, N);

    % Compute rotations parametrization by angles and translations
    angles0 = zeros(3, M);
    translations0 = zeros(3, M);

    for j = 1:M
        angles0(1, j) = -atan2(R_t_0(3 * j - 1, 3), R_t_0(3 * j, 3));
        angles0(2, j) = -atan2(-R_t_0(3 * j - 2, 3), norm(R_t_0(3 * j - 1:3 * j, 3)));
        angles0(3, j) = -atan2(R_t_0(3 * j - 2, 2), R_t_0(3 * j - 2, 1));
        translations0(:, j) = R_t_0(3 * j - 2:3 * j, 4);
    end

    % Optimization using Levenberg-Marquardt algorithm
    func = @(x)LeastSquareMinLM(x, matchingPoints, calMatrices);
    variables0 = reshape([angles0(:, 2:M), translations0(:, 2:M), Rec0], [], 1);
    options = optimoptions(@lsqnonlin, 'Algorithm', 'levenberg-marquardt', 'Jacobian', 'on', 'Display', 'off');
    [variables, ~, ~, ~, output] = lsqnonlin(func, variables0, [], [], options);

    % Final reprojection error
    repr_err = norm(func(variables));

    % Number of iterations
    iter = output.iterations;

    % Recover final reconstruction and orientations, and fix scale
    variables = reshape(variables, 3, N + 2 * (M - 1));
    angles = variables(:, 1:(M - 1));
    translations = variables(:, M:2 * (M - 1)); scale = 1 / norm(translations(:, 1));
    R_t = zeros(3 * M, 4); R_t(1:3, :) = eye(3, 4);

    for j = 1:(M - 1)
        Rx = [1 0 0; 0 cos(angles(1, j)) -sin(angles(1, j)); 0 sin(angles(1, j)) cos(angles(1, j))];
        Ry = [cos(angles(2, j)) 0 sin(angles(2, j)); 0 1 0; -sin(angles(2, j)) 0 cos(angles(2, j))];
        Rz = [cos(angles(3, j)) -sin(angles(3, j)) 0; sin(angles(3, j)) cos(angles(3, j)) 0; 0 0 1];
        R_t(3 * j + (1:3), :) = [Rx * Ry * Rz, scale * translations(:, j)];
    end

    Rec = scale * variables(:, 2 * (M - 1) + 1:2 * (M - 1) + N);

end

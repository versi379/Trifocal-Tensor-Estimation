% This function implements the Least Squares Minimization with Levenberg-Marquardt.

function [f, J] = LeastSquareMinLM(variables, matchingPoints, calMatrices)

    M = size(matchingPoints, 1) / 2; % Number of images/cameras
    N = size(matchingPoints, 2); % Number of 3D points to recover

    % Extract rotations and translations from variables
    variables = reshape(variables, 3, N + 2 * (M - 1));
    angles = variables(:, 1:(M - 1));
    translations = variables(:, M:2 * (M - 1));
    space_points = variables(:, 2 * (M - 1) + 1:2 * (M - 1) + N);

    Rot = repmat(eye(3 * M, 3), 1, 3); dRot = zeros(3 * M, 9);
    Pcam = zeros(3 * M, 4);
    Pcam(1:3, :) = calMatrices(1:3, :) * eye(3, 4);

    for j = 1:(M - 1)
        % Camera number j+1
        Rx = [1 0 0; 0 cos(angles(1, j)) -sin(angles(1, j)); 0 sin(angles(1, j)) cos(angles(1, j))];
        Ry = [cos(angles(2, j)) 0 sin(angles(2, j)); 0 1 0; -sin(angles(2, j)) 0 cos(angles(2, j))];
        Rz = [cos(angles(3, j)) -sin(angles(3, j)) 0; sin(angles(3, j)) cos(angles(3, j)) 0; 0 0 1];
        Rot(3 * j + (1:3), :) = [Rx, Ry, Rz];

        Dx = zeros(3); Dx(2:3, 2:3) = [-sin(angles(1, j)) -cos(angles(1, j)); cos(angles(1, j)) -sin(angles(1, j))];
        Dy = zeros(3); Dy([1 3], [1 3]) = [-sin(angles(2, j)) cos(angles(2, j)); -cos(angles(2, j)) -sin(angles(2, j))];
        Dz = zeros(3); Dz(1:2, 1:2) = [-sin(angles(3, j)) -cos(angles(3, j)); cos(angles(3, j)) -sin(angles(3, j))];
        dRot(3 * j + (1:3), :) = [Dx, Dy, Dz];

        Pcam(3 * j + (1:3), :) = calMatrices(3 * j + (1:3), :) * [Rx * Ry * Rz, translations(:, j)];
    end

    f = zeros(N * M * 2, 1);
    J = zeros(N * M * 2, 3 * (2 * (M - 1) + N));

    for i = 1:N
        % 3D point associated with correspondence i
        Point = space_points(:, i);

        for j = 1:M

            if isnan(matchingPoints(2 * j - 1, i))
                continue;
            end

            % Camera j
            ind_cam = 3 * j - 2:3 * j;
            P = Pcam(ind_cam, :); K = calMatrices(ind_cam, :);
            dR_x = dRot(ind_cam, 1:3); R_x = Rot(ind_cam, 1:3);
            dR_y = dRot(ind_cam, 4:6); R_y = Rot(ind_cam, 4:6);
            dR_z = dRot(ind_cam, 7:9); R_z = Rot(ind_cam, 7:9);

            % Point in image j for correspondence i
            point = matchingPoints(2 * j - 1:2 * j, i);

            ind = 2 * M * (i - 1) + 2 * (j - 1);

            % f: distance from p to projection P_j*P
            [aux, dgamma] = Gamma(P * [Point; 1]);
            [aux, ~, dydist] = Dist(point, aux);
            f(ind + 1:ind + 2) = aux;

            % Jacobians for f
            Jac = zeros(3, 3 * (2 * (M - 1) + N));

            % Respect 3D point
            Jac(:, 6 * (M - 1) + 3 * (i - 1) + (1:3)) = P(:, 1:3);

            if j > 1
                % Respect translation of camera j
                Jac(:, 3 * (M - 1) + 3 * (j - 2) + (1:3)) = K;

                % Respect rotation (angles)
                Jac(:, 3 * (j - 2) + (1:3)) = [K * (dR_x * R_y * R_z) * Point, ...
                                          K * (R_x * dR_y * R_z) * Point, K * (R_x * R_y * dR_z) * Point];
            end

            J(ind + 1:ind + 2, :) = dydist * dgamma * Jac;
        end

    end

end

function [f, dfx, dfy] = Dist(x, y)
    f = x - y;
    dfx = eye(2);
    dfy = -eye(2);
end

function [f, df] = Gamma(v)
    f = v(1:2) / v(3);
    df = [(1 / v(3)) * eye(2), [-v(1) / (v(3) ^ 2); -v(2) / (v(3) ^ 2)]];
end

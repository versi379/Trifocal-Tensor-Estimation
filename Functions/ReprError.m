% This function computes the reprojection error of N points
% for M perspective cameras.

function error = ReprError(ProjM, matchingPoints, Points3D)

    N = size(matchingPoints, 2); % Number of 3D points to reproject
    M = size(ProjM, 2); % Number of images/cameras

    % Compute triangulation of 3D points
    if nargin ~= 3
        Points3D_est = Triangulate3DPoints(ProjM, matchingPoints);
    elseif size(Points3D, 1) == 3
        Points3D_est = [Points3D; ones(1, N)];
    elseif size(Points3D, 1) == 4
        Points3D_est = Points3D;
    end

    % Convert to affine coordinates and adapt matchingPoints matrix
    if size(matchingPoints, 1) == 3 * M
        matchingPoints = reshape(matchingPoints, 3, N * M);
        matchingPoints = matchingPoints(1:2, :) ./ repmat(matchingPoints(3, :), 2, 1);
    else
        matchingPoints = reshape(matchingPoints, 2, N * M);
    end

    % Reproject points
    P = cell2mat(ProjM.');
    matchingPointsEst = reshape(P * Points3D_est, 3, M * N);
    matchingPointsEst = matchingPointsEst(1:2, :) ./ repmat(matchingPointsEst(3, :), 2, 1);

    % Compute RMS of distances
    error = sqrt(mean(sum((matchingPointsEst - matchingPoints) .^ 2, 1)));

end

% --- COMPLETED ---

% This function takes a set of points in 2D space and returns
% a normalization matrix that, applied to the points (in homogeneous coordinates),
% transforms them to have a mean at the origin (0,0) and an average distance to the origin equal to sqrt(2)
%
% Input:
%  - points: 2xN matrix of N 2-dimensional points (cartesian)
%
% Output:
%  - normMatrix: 3x3 isometric normalization matrix
%  - normPoints: 3xN matrix of N 2-dimensional points (homogeneous)

function [normPoints, normMatrix] = Normalize2DPoints(points)

    % Number of points is computed as
    % number of columns in either matrix points
    N = size(points, 2);

    % 2x1 array with mean value of each row in points
    point0 = mean(points, 2);

    % Mean value of 1xN array with Euclidean distances
    % from each point to mean point (point0)
    norm0 = mean(sqrt(sum((points - repmat(point0, 1, N)) .^ 2, 1)));

    % Scaling component of normalization matrix:
    % first two diagonal elements set to sqrt(2)/norm0 and third element set to 1
    N_matrix = diag([sqrt(2) / norm0; sqrt(2) / norm0; 1]);
    % Translation component of normalization matrix:
    % mean point of transformed points becomes origin
    N_matrix(1:2, 3) = -sqrt(2) * point0 / norm0;

    % Apply transformation on input points
    new_points = N_matrix(1:2, :) * [points; ones(1, N)];

end

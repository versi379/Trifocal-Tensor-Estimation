% --- DF ---

% This function computes the Fundamental Matrix (FM)
% from corresponding points in two images using
% linear equations derived from epipolar constraints
%
% Input:
% p1: 3xN (homogeneous) or 2xN (cartesian) matrix
%        of N image points in image 1
% p2: 3xN (homogeneous) or 2xN (cartesian) matrix
%        of N image points in image 2
%
% Output:
% F: 3x3 Fundamental Matrix (FM)

function F = LinearFMM(p1, p2)

    % Number of correspondence points is computed as
    % number of columns in either matrix p1 or p2
    N = size(p1, 2);

    % Same number of image points check
    if N ~= size(p2, 2)
        error('Number of points in image 1 and image 2 must be equal.');
    end

    % Minimum number of correspondences check
    if N < 8
        error('At least 8 correspondence points are necessary.');
    end

    % Homogeneous to cartesian coordinates
    if size(p1, 1) == 3
        p1 = p1(1:2, :) ./ repmat(p1(3, :), 2, 1);
        p2 = p2(1:2, :) ./ repmat(p2(3, :), 2, 1);
    end

    % Normalize image points
    [p1, Normal1] = Normalize2DPoints(p1(1:2, :));
    [p2, Normal2] = Normalize2DPoints(p2(1:2, :));

    % --- NORMALIZED 8 POINT ALGORITHM ---

    % Each row of matrix A corresponds to a pair of corresponding points,
    % each column corresponds to one of the coefficients of the FM
    A = zeros(N, 9);

    for i = 1:N
        x1 = p1(1:2, i); x2 = p2(1:2, i);
        A(i, :) = [x1(1) * x2(1), x1(1) * x2(2), x1(1), x1(2) * x2(1), ...
                       x1(2) * x2(2), x1(2), x2(1), x2(2), 1];
    end

    [~, ~, V] = svd(A);

    % Initial estimate of the FM
    F = reshape(V(:, size(V, 2)), 3, 3);

    % Undo normalization
    F = Normal2.' * F * Normal1;

    % Singularity constraint
    [U, D, V] = svd(F); D(3, 3) = 0;
    F = U * D * V.';

end

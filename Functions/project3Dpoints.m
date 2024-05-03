% This function projects a set of N 3D points onto M images.

function matchingPoints = Project3DPoints(Points3D, Pcam)

    N = size(Points3D, 2); % Number of 3D points to project
    M = size(Pcam, 2); % Number of images/cameras

    matchingPoints = zeros(2 * M, N);

    for m = 1:M
        x = Pcam{m} * [Points3D; ones(1, N)];
        matchingPoints(2 * (m - 1) + (1:2), :) = bsxfun(@rdivide, x(1:2, :), x(3, :));
    end

end

% This function performs triangulation of N 3D points from their image projections
% in M>1 images. The triangulation is initially computed by the
% Direct Linear Transformation (DLT) algorithm.

function space_points = Triangulate3DPoints(Pcam, image_points)

    N = size(image_points, 2); % Number of 3D points to triangulate
    M = size(Pcam, 2); % Number of images/cameras
    if M < 2
        return
    end

    switch size(image_points, 1)
        case 2 * M
            % Euclidean coordinates
        case 3 * M
            % Homogeneous coordinates
            aux = reshape(image_points, 3, N * M);
            aux = aux(1:2, :) ./ repmat(aux(3, :), 2, 1);
            image_points = reshape(aux, 2 * M, N);
        otherwise
            return
    end

    space_points = zeros(4, N);

    for n = 1:N
        corresp_n = image_points(:, n);

        % DLT solution
        ls_matrix = zeros(2 * M, 4); % Linear system matrix

        for i = 1:M
            point = corresp_n(2 * (i - 1) + 1:2 * (i - 1) + 2, :);
            ls_matrix(2 * (i - 1) + 1:2 * (i - 1) + 2, :) = ...
                [0 -1 point(2); 1 0 -point(1)] * Pcam{i};
        end

        [~, ~, V] = svd(ls_matrix);
        PointX = V(:, 4);
        space_points(:, n) = PointX;
    end

end

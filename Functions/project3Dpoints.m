function Corresp = Project3DPoints(Points3D, Pcam)

    M = size(Pcam, 2); % number of cameras
    N = size(Points3D, 2); %number of points to project

    Corresp = zeros(2 * M, N);

    for m = 1:M
        x = Pcam{m} * [Points3D; ones(1, N)];
        Corresp(2 * (m - 1) + (1:2), :) = bsxfun(@rdivide, x(1:2, :), x(3, :));
    end

end

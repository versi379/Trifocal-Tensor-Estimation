function [R_t_2, R_t_3, Reconst, T, iter] = OptimFPoseEstimation(Corresp, CalM)

    N = size(Corresp, 2);
    K1 = CalM(1:3, :); K2 = CalM(4:6, :); K3 = CalM(7:9, :);

    % the 2 fundamental matrices
    [F21, it1] = optimF(Corresp(1:2, :), Corresp(3:4, :));
    [F31, it2] = optimF(Corresp(1:2, :), Corresp(5:6, :));
    iter = it1 + it2;

    % Find orientation using calibration and F matrices
    [R2, t2] = recover_R_t(K1, K2, F21, Corresp(1:2, :), Corresp(3:4, :));
    [R3, t3] = recover_R_t(K1, K3, F31, Corresp(1:2, :), Corresp(5:6, :));

    % Find the norm of t31 using the image points and reconstruction from
    % images 1 and 2
    u3 = K3 * t3;
    X = triangulation3D({K1 * eye(3, 4), K2 * [R2, t2]}, Corresp(1:4, :));
    X = X(1:3, :) ./ repmat(X(4, :), 3, 1);
    X3 = K3 * R3 * X;
    lam = -sum(dot(cross([Corresp(5:6, :); ones(1, N)], X3, 1), cross([Corresp(5:6, :); ones(1, N)], repmat(u3, 1, N)), 1)) / ...
        sum(sum(cross([Corresp(5:6, :); ones(1, N)], repmat(u3, 1, N)) .^ 2));
    t3 = lam * t3;

    R_t_2 = [R2, t2]; R_t_3 = [R3, t3];

    % Find 3D points by triangulation
    Reconst = triangulation3D({K1 * eye(3, 4), K2 * R_t_2, K3 * R_t_3}, Corresp);
    Reconst = Reconst(1:3, :) ./ repmat(Reconst(4, :), 3, 1);
    T = TFT_from_P(K1 * eye(3, 4), K2 * R_t_2, K3 * R_t_3);

end

function [R_f, t_f] = recover_R_t(K1, K2, F21, x1, x2)

    E21 = K2.' * F21 * K1;
    W = [0 -1 0; 1 0 0; 0 0 1];
    [U, ~, V] = svd(E21);
    R = U * W * V.'; Rp = U * W.' * V.';
    R = R * sign(det(R)); Rp = Rp * sign(det(Rp));
    t = U(:, 3);

    %from the 4 possible solutions find the correct one using the image points
    num_points_seen = 0;

    for k = 1:4

        if k == 2 || k == 4
            t = -t;
        elseif k == 3
            R = Rp;
        end

        X1 = triangulation3D({[K1 [0; 0; 0]], K2 * [R, t]}, [x1; x2]); X1 = X1 ./ repmat(X1(4, :), 4, 1);
        X2 = [R t] * X1;

        if sum(sign(X1(3, :)) + sign(X2(3, :))) >= num_points_seen
            R_f = R; t_f = t;
            num_points_seen = sum(sign(X1(3, :)) + sign(X2(3, :)));
        end

    end

end

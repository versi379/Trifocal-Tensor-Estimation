function [R_t_2, R_t_3, Reconst, T, iter] = NordbergTFTPoseEst(Corresp, CalM)

    % Normalization of the data
    [x1, Normal1] = Normalize2DPoints(Corresp(1:2, :));
    [x2, Normal2] = Normalize2DPoints(Corresp(3:4, :));
    [x3, Normal3] = Normalize2DPoints(Corresp(5:6, :));

    % Model to estimate T: linear equations
    [T, P1, P2, P3] = LinearTFT(x1, x2, x3);

    % apply projective transformation so matrix B ( P3=[B|b] ) has full rank
    H = eye(4);

    if rank(P3(:, 1:3)) < 3
        H(4, 1:3) = null(P3(:, 1:3))';
    elseif rank(P2(:, 1:3)) < 3
        H(4, 1:3) = null(P2(:, 1:3))';
    end

    P1 = P1 * H; P2 = P2 * H; P3 = P3 * H;

    % Compute K. Nordberg param
    A = P2(:, 1:3); a = P2(:, 4); r = A \ a;
    B = P3(:, 1:3); b = P3(:, 4); s = B \ b;

    U = [r, CrossProdMatrix(r) ^ 2 * s, CrossProdMatrix(r) * s]; U = U * (U' * U) ^ (-1/2); U = sign(det(U)) * U;
    V = [a, CrossProdMatrix(a) * A * s, CrossProdMatrix(a) ^ 2 * A * s]; V = V * (V' * V) ^ (-1/2); V = sign(det(V)) * V;
    W = [b, CrossProdMatrix(b) * B * r, CrossProdMatrix(b) ^ 2 * B * r]; W = W * (W' * W) ^ (-1/2); W = sign(det(W)) * W;

    % good representation of U V W
    [~, ~, v] = svd(U - eye(3)); vec_u = v(:, 3);
    o_u = atan2(vec_u' * [U(3, 2) - U(2, 3); U(1, 3) - U(3, 1); U(2, 1) - U(1, 2)] / 2, (trace(U) - 1) / 2);
    [~, ~, v] = svd(V - eye(3)); vec_v = v(:, 3);
    o_v = atan2(vec_v' * [V(3, 2) - V(2, 3); V(1, 3) - V(3, 1); V(2, 1) - V(1, 2)] / 2, (trace(V) - 1) / 2);
    [~, ~, v] = svd(W - eye(3)); vec_w = v(:, 3);
    o_w = atan2(vec_w' * [W(3, 2) - W(2, 3); W(1, 3) - W(3, 1); W(2, 1) - W(1, 2)] / 2, (trace(W) - 1) / 2);

    % sparce tensor
    Ts = ComputeTensorfromMatrices(T, U, V, W);
    paramT = Ts([1, 7, 10, 12, 16, 19:22, 25])';
    paramT = paramT / norm(paramT);

    % compute 3D estimated points to have initial estimated reprojected image
    % points
    points3D = Triangulate3DPoints({P1, P2, P3}, [x1; x2; x3]);
    p1_est = P1 * points3D; p1_est = p1_est(1:2, :) ./ repmat(p1_est(3, :), 2, 1);
    p2_est = P2 * points3D; p2_est = p2_est(1:2, :) ./ repmat(p2_est(3, :), 2, 1);
    p3_est = P3 * points3D; p3_est = p3_est(1:2, :) ./ repmat(p3_est(3, :), 2, 1);

    % minimize error with Gauss-Helmert
    N = size(x1, 2);
    param0 = [vec_u * o_u; vec_v * o_v; vec_w * o_w; paramT];
    obs = reshape([x1(1:2, :); x2(1:2, :); x3(1:2, :)], 6 * N, 1);
    obs_est = reshape([p1_est; p2_est; p3_est], 6 * N, 1);
    [~, param, ~, iter] = GaussHelmert(@ConstraintsGH_NordbergTFT, obs_est, param0, zeros(0, 1), obs, eye(6 * N));

    % recover orthogonal matrices from optimized parameters
    o_u = norm(param(1:3)); vec_u = param(1:3) / o_u;
    o_v = norm(param(4:6)); vec_v = param(4:6) / o_v;
    o_w = norm(param(7:9)); vec_w = param(7:9) / o_w;
    U = eye(3) + sin(o_u) * CrossProdMatrix(vec_u) + (1 - cos(o_u)) * CrossProdMatrix(vec_u) ^ 2;
    V = eye(3) + sin(o_v) * CrossProdMatrix(vec_v) + (1 - cos(o_v)) * CrossProdMatrix(vec_v) ^ 2;
    W = eye(3) + sin(o_w) * CrossProdMatrix(vec_w) + (1 - cos(o_w)) * CrossProdMatrix(vec_w) ^ 2;

    % sparse tensor from optimized parameters
    Ts = zeros(3, 3, 3);
    Ts([1, 7, 10, 12, 16, 19:22, 25]) = param(10:19);

    % trifocal tensor
    T = ComputeTensorfromMatrices(Ts, U', V', W');

    % denormalization
    T = TransformTFT(T, Normal1, Normal2, Normal3, 1);

    % Find orientation using calibration and TFT
    [R_t_2, R_t_3] = PoseEstfromTFT(T, CalM, Corresp);

    % Find 3D points by triangulation
    Reconst = Triangulate3DPoints({CalM(1:3, :) * eye(3, 4), CalM(4:6, :) * R_t_2, CalM(7:9, :) * R_t_3}, Corresp);
    Reconst = Reconst(1:3, :) ./ repmat(Reconst(4, :), 3, 1);

end


% This function implements constraints and parameters for the optimization
% of the Nordberg's parameterization with Gauss-Helmert.

function [f, g, A, B, C, D] = ConstraintsGH_NordbergTFT(obs, x, ~)

    % orthogonal matrices
    o_u = norm(x(1:3)); vec_u = x(1:3) / o_u;
    o_v = norm(x(4:6)); vec_v = x(4:6) / o_v;
    o_w = norm(x(7:9)); vec_w = x(7:9) / o_w;
    U = eye(3) + sin(o_u) * CrossProdMatrix(vec_u) + (1 - cos(o_u)) * CrossProdMatrix(vec_u) ^ 2;
    V = eye(3) + sin(o_v) * CrossProdMatrix(vec_v) + (1 - cos(o_v)) * CrossProdMatrix(vec_v) ^ 2;
    W = eye(3) + sin(o_w) * CrossProdMatrix(vec_w) + (1 - cos(o_w)) * CrossProdMatrix(vec_w) ^ 2;

    % sparse tensor
    paramT = x(10:19);
    Ts = zeros(3, 3, 3);
    param_ind = [1, 7, 10, 12, 16, 19:22, 25];
    Ts(param_ind) = paramT;

    % original tensor
    T = ComputeTensorfromMatrices(Ts, U', V', W');

    % observations
    obs = reshape(obs, 6, []);
    N = size(obs, 2);

    f = zeros(4 * N, 1); % constraints for tensor and observations (trilinearities)
    Ap = zeros(4 * N, 27); % jacobian of f w.r.t. the tensor T
    B = zeros(4 * N, 6 * N); % jacobian of f w.r.t. the observations

    for i = 1:N
        % points in the three images for correspondence i
        x1 = obs(1:2, i); x2 = obs(3:4, i); x3 = obs(5:6, i);

        % 4 trilinearities
        ind2 = 4 * (i - 1);
        S2 = [0 -1; -1 0; x2(2) x2(1)];
        S3 = [0 -1; -1 0; x3(2) x3(1)];
        f(ind2 + 1:ind2 + 4) = reshape(S2' * (x1(1) * T(:, :, 1) + x1(2) * T(:, :, 2) + T(:, :, 3)) * S3, 4, 1);

        % Jacobians for the trilinearities
        Ap(ind2 + 1:ind2 + 4, :) = kron(S3, S2)' * kron([x1; 1]', eye(9));
        B(ind2 + 1:ind2 + 4, 6 * (i - 1) + 1) = reshape(S2' * T(:, :, 1) * S3, 4, 1);
        B(ind2 + 1:ind2 + 4, 6 * (i - 1) + 2) = reshape(S2' * T(:, :, 2) * S3, 4, 1);
        B(ind2 + 1:ind2 + 4, 6 * (i - 1) + (3:4)) = kron(S3' * reshape(T(3, :, :), 3, 3) * [x1; 1], [0, 1; 1, 0]);
        B(ind2 + 1:ind2 + 4, 6 * (i - 1) + (5:6)) = kron([0, 1; 1, 0], S2' * reshape(T(:, 3, :), 3, 3) * [x1; 1]);
    end

    % Jacobian of T=Ts(U,V,W) w.r.t. the minimal parameterization
    J = zeros(27, 19);
    % derivatives of the parameterization w.r.t. the sparse tensor
    for i = 1:10
        e = zeros(3, 3, 3); e(param_ind(i)) = 1;
        J(:, i + 9) = reshape(ComputeTensorfromMatrices(e, U', V', W'), 27, 1);
    end

    % derivatives of the parameterization w.r.t. the orthogonal matrices
    dU = zeros(3, 3, 3); dV = zeros(3, 3, 3); dW = zeros(3, 3, 3);
    e = eye(3);

    for i = 1:3
        dU(:, :, i) = -vec_u(i) * sin(o_u) * eye(3) + vec_u(i) * cos(o_u) * CrossProdMatrix(vec_u) + ...
            sin(o_u) * (1 / o_u) * (CrossProdMatrix(e(:, i)) - vec_u(i) * CrossProdMatrix(vec_u)) + ...
            vec_u(i) * sin(o_u) * (vec_u * vec_u') + ...
            (1 - cos(o_u)) * (1 / o_u) * (vec_u * e(i, :) + e(:, i) * vec_u' - 2 * vec_u(i) * (vec_u * vec_u'));

        dV(:, :, i) = -vec_v(i) * sin(o_v) * eye(3) + vec_v(i) * cos(o_v) * CrossProdMatrix(vec_v) + ...
            sin(o_v) * (1 / o_v) * (CrossProdMatrix(e(:, i)) - vec_v(i) * CrossProdMatrix(vec_v)) + ...
            vec_v(i) * sin(o_v) * (vec_v * vec_v') + ...
            (1 - cos(o_v)) * (1 / o_v) * (vec_v * e(i, :) + e(:, i) * vec_v' - 2 * vec_v(i) * (vec_v * vec_v'));

        dW(:, :, i) = -vec_w(i) * sin(o_w) * eye(3) + vec_w(i) * cos(o_w) * CrossProdMatrix(vec_w) + ...
            sin(o_w) * (1 / o_w) * (CrossProdMatrix(e(:, i)) - vec_w(i) * CrossProdMatrix(vec_w)) + ...
            vec_w(i) * sin(o_w) * (vec_w * vec_w') + ...
            (1 - cos(o_w)) * (1 / o_w) * (vec_w * e(i, :) + e(:, i) * vec_w' - 2 * vec_w(i) * (vec_w * vec_w'));
    end

    for i = 1:3
        J(:, i) = reshape(ComputeTensorfromMatrices(Ts, dU(:, :, i)',V', W'), 27, 1);
        J(:, i + 3) = reshape(ComputeTensorfromMatrices(Ts, U', dV(:, :, i)',W'), 27, 1);
        J(:, i + 6) = reshape(ComputeTensorfromMatrices(Ts, U', V', dW(:, :, i)'), 27, 1);
    end

    A = Ap * J; % Jacobian of f w.r.t. the minimal parameterization

    g = sum(paramT .^ 2) - 1; % constraints on the minimal parameterization
    C = zeros(1, 19); % jacobian of g w.r.t. the minimal parameterization
    C(1, 10:19) = 2 * paramT';
    D = zeros(1, 0);

end
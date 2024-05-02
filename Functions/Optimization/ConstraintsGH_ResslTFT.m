% This function implements constraints and parameters for the optimization
% of the Ressl's parameterization for collinear cameras with Gauss-Helmert.

function [f, g, A, B, C, D] = ConstraintsGH_ResslTFT(x, p, ~, Ind)

    Ind2 = 1:3; Ind2 = Ind2(Ind2 ~= Ind);
    % recover Ressl's paremeters
    S = reshape(p(1:9), 3, 3);
    e21 = ones(3, 1); e21(Ind2) = p(10:11); e31 = p(18:20);
    mn = zeros(3); mn(:, Ind2) = reshape(p(12:17), 3, 2);
    % tensor
    T1 = (S(:, 1) * e21.' + e31 * mn(1, :)).';
    T2 = (S(:, 2) * e21.' + e31 * mn(2, :)).';
    T3 = (S(:, 3) * e21.' + e31 * mn(3, :)).';

    % other slices of the TFT
    J3 = [T1(3, :); T2(3, :); T3(3, :)];
    K3 = [T1(:, 3), T2(:, 3), T3(:, 3)];

    N = size(x, 1) / 6;

    % constraints evaluated in (p)
    g = [sum(e31 .^ 2) - 1; sum(S(:) .^ 2) - 1];

    % g jacobian w.r.t. p evaluated in p
    C = zeros(2, 20);
    C(1, 18:20) = 2 * e31.';
    C(2, 1:9) = 2 * S(:).';

    f = zeros(4 * N, 1); % constraints for tensor and observations (trilinearities)
    Ap = zeros(4 * N, 27); % jacobian of f w.r.t. the tensor T
    B = zeros(4 * N, 6 * N); % jacobian of f w.r.t. the observations

    for i = 1:N

        % points in the three images for correspondance i
        ind = 6 * (i - 1);
        x1 = x(ind + 1:ind + 2);
        x2 = x(ind + 3:ind + 4);
        x3 = x(ind + 5:ind + 6);

        % 4 trilinearities
        ind2 = 4 * (i - 1);
        S2 = [0 -1; -1 0; x2(2) x2(1)];
        S3 = [0 -1; -1 0; x3(2) x3(1)];
        f(ind2 + 1:ind2 + 4) = reshape(S2.' * (x1(1) * T1 + x1(2) * T2 + T3) * S3, 4, 1);

        % Jacobians for the trilinearities
        Ap(ind2 + 1:ind2 + 4, :) = kron(S3, S2).' * kron([x1; 1].', eye(9));
        B(ind2 + 1:ind2 + 4, ind + 1) = reshape(S2.' * T1 * S3, 4, 1);
        B(ind2 + 1:ind2 + 4, ind + 2) = reshape(S2.' * T2 * S3, 4, 1);
        B(ind2 + 1:ind2 + 4, ind + 3:ind + 4) = kron(S3.' * J3.' * [x1; 1], [0, 1; 1, 0]);
        B(ind2 + 1:ind2 + 4, ind + 5:ind + 6) = kron([0, 1; 1, 0], S2.' * K3 * [x1; 1]);
    end

    % jacobian for parametrization t=F(p) w.r.t. p evaluated in p
    D = zeros(27, 20);
    D(:, 1:9) = kron(eye(3), kron(eye(3), e21));
    aux = zeros(3, 2); aux(Ind2, :) = eye(2);
    D(:, 10:11) = [kron(S(:, 1), aux); kron(S(:, 2), aux); kron(S(:, 3), aux)];
    D(:, 12:14) = kron(eye(3), kron(e31, aux(:, 1)));
    D(:, 15:17) = kron(eye(3), kron(e31, aux(:, 2)));
    D(:, 18:20) = [kron(eye(3), mn(1, :).'); kron(eye(3), mn(2, :).'); kron(eye(3), mn(3, :).')];

    % jacobian of f w.r.t. the minimal parameterization
    A = Ap * D;

    D = zeros(2, 0);

end
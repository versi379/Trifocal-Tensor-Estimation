% This function implements constraints and parameters for the optimization
% of the FM with Gauss-Helmert.

function [f, g, A, B, C, D] = ConstraintsGH(x, p, ~)

    N = size(x, 1) / 4;
    x = reshape(x, 4, N);

    F = reshape(p, 3, 3);

    g = [det(F); sum(F(1:9) .^ 2) - 1];

    C = [F(5) * F(9) - F(6) * F(8), F(6) * F(7) - F(4) * F(9), F(4) * F(8) - F(5) * F(7), ...
             F(3) * F(8) - F(2) * F(9), F(1) * F(9) - F(3) * F(7), F(2) * F(7) - F(1) * F(8), ...
             F(2) * F(6) - F(3) * F(5), F(3) * F(4) - F(1) * F(6), F(1) * F(5) - F(2) * F(4);
         2 * reshape(F, 9, 1).'];

    f = zeros(N, 1);
    A = zeros(N, 9);
    B = zeros(N, 4 * N);

    for i = 1:N
        x1 = [x(1:2, i); 1]; x2 = [x(3:4, i); 1];
        f(i, :) = x2.' * F * x1;
        A(i, :) = [x1(1) * x2(1), x1(1) * x2(2), x1(1), x1(2) * x2(1), x1(2) * x2(2), x1(2), x2(1), x2(2), 1];
        B(i, 4 * (i - 1) + 1:4 * (i - 1) + 4) = [F(3) + F(1) * x2(1) + F(2) * x2(2), F(6) + F(4) * x2(1) + F(5) * x2(2), ...
                                                     F(7) + F(1) * x1(1) + F(4) * x1(2), F(8) + F(2) * x1(1) + F(5) * x1(2)];
    end

    D = zeros(2, 0);

end

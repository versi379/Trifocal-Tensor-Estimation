% This function implements constraints and parameters for the optimization
% of the Ponce-Hebert's parameterization with Gauss-Helmert.

function [f, g, A, B, C, D] = ConstraintsGH_PiTFT(x, pi, ~, ~)

    N = size(x, 1) / 6;

    % Pi vectors
    pi21 = pi(1:3); pi31 = pi(4:6); pi41 = pi(7:9);
    pi12 = pi(10:12); pi32 = pi(13:15); pi42 = pi(16:18);
    pi13 = pi(19:21); pi23 = pi(22:24); pi43 = pi(25:27);

    % Compute FMs
    F12 = pi41 * pi32.' - pi31 * pi42.';
    F13 = pi41 * pi23.' - pi21 * pi43.';
    F23 = pi42 * pi13.' - pi12 * pi43.';

    % Constraints for pi parametrization evaluated in pi
    g = [sum(pi41 .^ 2) - 1; sum(pi42 .^ 2) - 1; sum(pi43 .^ 2) - 1; ...
             sum(pi21 .^ 2) - 1; sum(pi32 .^ 2) - 1; sum(pi13 .^ 2) - 1; ...
             pi21.' * pi41; pi32.' * pi42; pi13.' * pi43];

    % Jacobian of g w.r.t. pi evaluated in pi
    C = zeros(9, 27);
    C(1, 7:9) = 2 * pi41.';
    C(2, 16:18) = 2 * pi42.';
    C(3, 25:27) = 2 * pi43.';
    C(4, 1:3) = 2 * pi21.';
    C(5, 13:15) = 2 * pi32.';
    C(6, 19:21) = 2 * pi13.';
    C(7, [1:3 7:9]) = [pi41.' pi21.'];
    C(8, [13:15 16:18]) = [pi42.' pi32.'];
    C(9, [19:21 25:27]) = [pi43.' pi13.'];

    % Epi-trilinear conditions evaluated in (x,pi) and jacobians
    f = zeros(4 * N, 1);
    A = zeros(4 * N, 27);
    B = zeros(4 * N, 6 * N);

    for i = 1:N
        % Points in the three images for correspondance i
        ind = 6 * (i - 1);
        x1 = x(ind + 1:ind + 2); p1 = [x1; 1];
        x2 = x(ind + 3:ind + 4); p2 = [x2; 1];
        x3 = x(ind + 5:ind + 6); p3 = [x3; 1];

        % Epipolar constraints and trilinearities
        ind2 = 4 * (i - 1);
        f(ind2 + 1:ind2 + 4) = [p1.' * F12 * p2; p1.' * F13 * p3; p2.' * F23 * p3; ...
                                    ((pi21.' * p1) * (pi32.' * p2) * (pi13.' * p3) - (pi31.' * p1) * (pi12.' * p2) * (pi23.' * p3))];

        % Jacobians for f
        A(ind2 + 1, 4:9) = [-pi42.' * p2 * p1.', pi32.' * p2 * p1.'];
        A(ind2 + 1, 13:18) = [pi41.' * p1 * p2.', -pi31.' * p1 * p2.'];
        A(ind2 + 2, [1:3 7:9]) = [-pi43.' * p3 * p1.', pi23.' * p3 * p1.'];
        A(ind2 + 2, 22:27) = [pi41.' * p1 * p3.', -pi21.' * p1 * p3.'];
        A(ind2 + 3, [10:12 16:18]) = [-pi43.' * p3 * p2.', pi13.' * p3 * p2.'];
        A(ind2 + 3, [19:21 25:27]) = [pi42.' * p2 * p3.', -pi12.' * p2 * p3.'];
        A(ind2 + 4, 1:6) = [p1.' * (pi32.' * p2) * (pi13.' * p3), -p1.' * (pi12.' * p2) * (pi23.' * p3)];
        A(ind2 + 4, 10:15) = [- (pi31.' * p1) * (pi23.' * p3) * p2.', (pi21.' * p1) * (pi13.' * p3) * p2.'];
        A(ind2 + 4, 19:24) = [(pi21.' * p1) * (pi32.' * p2) * p3.', - (pi31.' * p1) * (pi12.' * p2) * p3.'];

        B(ind2 + 1, ind + 1:ind + 2) = (p2.' * F12.') * [1 0; 0 1; 0 0]; B(ind2 + 1, ind + 3:ind + 4) = p1.' * F12 * [1 0; 0 1; 0 0];
        B(ind2 + 2, ind + 1:ind + 2) = (p3.' * F13.') * [1 0; 0 1; 0 0]; B(ind2 + 2, ind + 5:ind + 6) = p1.' * F13 * [1 0; 0 1; 0 0];
        B(ind2 + 3, ind + 3:ind + 4) = (p3.' * F23.') * [1 0; 0 1; 0 0]; B(ind2 + 3, ind + 5:ind + 6) = p2.' * F23 * [1 0; 0 1; 0 0];
        B(ind2 + 4, ind + 1:ind + 2) = (pi21 * (pi32.' * p2) * (pi13.' * p3) - pi31 * (pi12.' * p2) * (pi23.' * p3)).' * [1 0; 0 1; 0 0];
        B(ind2 + 4, ind + 3:ind + 4) = (pi32 * (pi21.' * p1) * (pi13.' * p3) - pi12 * (pi31.' * p1) * (pi23.' * p3)).' * [1 0; 0 1; 0 0];
        B(ind2 + 4, ind + 5:ind + 6) = (pi13 * (pi21.' * p1) * (pi32.' * p2) - pi23 * (pi31.' * p1) * (pi12.' * p2)).' * [1 0; 0 1; 0 0];
    end

    D = zeros(9, 0);

end

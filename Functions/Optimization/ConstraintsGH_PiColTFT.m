% This function implements constraints and parameters for the optimization
% of the Ponce-Hebert's parameterization for collinear cameras with Gauss-Helmert.

function [f, g, A, B, C, D] = ConstraintsGH_PiColTFT(x, pi, ~)

    N = size(x, 1) / 6;

    % Pi vectors
    pi21 = pi(1:3); pi31 = pi(4:6); pi41 = pi(7:9);
    pi12 = pi(10:12); pi32 = pi(13:15); pi42 = pi(16:18);
    w3 = pi(19:21); pi33 = pi(22:24); pi43 = pi(25:27);

    % Compute FMs
    F12 = pi41 * pi32.' - pi31 * pi42.';
    F13 = pi41 * pi33.' - pi31 * pi43.';
    F23 = pi42 * pi33.' - pi32 * pi43.';

    % Constraints for pi parametrization evaluated in pi
    g = [sum(pi21 .^ 2) - 1; sum(pi12 .^ 2) - 1; ...
             sum(w3 .^ 2) - 1; sum(pi33 .^ 2) - 1; sum(pi43 .^ 2) - 1; ...
             pi21.' * pi31; pi21.' * pi41; pi31.' * pi41; ...
             pi12.' * pi32; pi12.' * pi42; pi32.' * pi42];

    % Jacobian of g w.r.t. pi evaluated in pi
    C = zeros(11, 27);
    C(1, 1:3) = 2 * pi21.'; C(2, 10:12) = 2 * pi12.';
    C(3, 19:21) = 2 * w3.'; C(4, 22:24) = 2 * pi33.'; C(5, 25:27) = 2 * pi43.';
    C(6, 1:6) = [pi31.', pi21.']; C(9, 10:15) = [pi32.', pi12.'];
    C(7, [1:3, 7:9]) = [pi41.', pi21.']; C(10, [10:12, 16:18]) = [pi42.', pi12.'];
    C(8, 4:9) = [pi41.', pi31.']; C(11, 13:18) = [pi42.', pi32.'];

    % Epi-trilinear conditions evaluated in (x,pi) and jacobians
    f = zeros(5 * N, 1);
    A = zeros(5 * N, 27);
    B = zeros(5 * N, 6 * N);

    for i = 1:N
        % Points in the three images for correspondance i
        ind = 6 * (i - 1);
        x1 = x(ind + 1:ind + 2); p1 = [x1; 1];
        x2 = x(ind + 3:ind + 4); p2 = [x2; 1];
        x3 = x(ind + 5:ind + 6); p3 = [x3; 1];

        % Epipolar constraints and trilinearities
        ind2 = 5 * (i - 1);
        f(ind2 + 1:ind2 + 5) = [p1.' * F12 * p2; p1.' * F13 * p3; p2.' * F23 * p3; ...
                                    (pi31.' * p1) * (pi32.' * p2) * (w3.' * p3) + ((pi31.' * p1) * (pi12.' * p2) - (pi21.' * p1) * (pi32.' * p2)) * (pi33.' * p3); ...
                                    (pi41.' * p1) * (pi42.' * p2) * (w3.' * p3) + ((pi41.' * p1) * (pi12.' * p2) - (pi21.' * p1) * (pi42.' * p2)) * (pi43.' * p3)];

        % Jacobians for f
        A(ind2 + 1, [4:9, 13:18]) = [- (pi42.' * p2) * p1.', (pi32.' * p2) * p1.', ...
                                         (pi41.' * p1) * p2.', - (pi31.' * p1) * p2.'];
        A(ind2 + 2, [4:9, 22:27]) = [- (pi43.' * p3) * p1.', (pi33.' * p3) * p1.', ...
                                         (pi41.' * p1) * p3.', - (pi31.' * p1) * p3.'];
        A(ind2 + 3, [13:18, 22:27]) = [- (pi43.' * p3) * p2.', (pi33.' * p3) * p2.', ...
                                           (pi42.' * p2) * p3.', - (pi32.' * p2) * p3.'];
        A(ind2 + 4, [1:6, 10:15, 19:24]) = [p1.' * (pi32.' * p2) * (pi33.' * p3), ...
                                                p1.' * ((pi32.' * p2) * (w3.' * p3) + (pi12.' * p2) * (pi33.' * p3)), ...
                                                p2.' * (pi31.' * p1) * (pi33.' * p3), p2.' * ((pi31.' * p1) * (w3.' * p3) - (pi21.' * p1) * (pi33.' * p3)), ...
                                                p3.' * (pi31.' * p1) * (pi32.' * p2), p3.' * ((pi31.' * p1) * (pi12.' * p2) - (pi21.' * p1) * (pi32.' * p2))];
        A(ind2 + 5, [1:3, 7:12, 16:21, 25:27]) = [-p1.' * (pi42.' * p2) * (pi43.' * p3), ...
                                                      p1.' * ((pi42.' * p2) * (w3.' * p3) + (pi12.' * p2) * (pi43.' * p3)), ...
                                                      p2.' * (pi41.' * p1) * (pi43.' * p3), p2.' * ((pi41.' * p1) * (w3.' * p3) - (pi21.' * p1) * (pi43.' * p3)), ...
                                                      p3.' * (pi41.' * p1) * (pi42.' * p2), p3.' * ((pi41.' * p1) * (pi12.' * p2) - (pi21.' * p1) * (pi42.' * p2))];

        B(ind2 + 1, ind + (1:4)) = [(p2.' * F12.') * eye(3, 2), (p1.' * F12) * eye(3, 2)];
        B(ind2 + 2, ind + [1:2, 5:6]) = [(p3.' * F13.') * eye(3, 2), (p1.' * F13) * eye(3, 2)];
        B(ind2 + 3, ind + (3:6)) = [(p3.' * F23.') * eye(3, 2), (p2.' * F23) * eye(3, 2)];
        B(ind2 + 4, ind + (1:6)) = ...
            [(pi31.' * ((pi32.' * p2) * (w3.' * p3) + (pi12.' * p2) * (pi33.' * p3)) - pi21.' * (pi32.' * p2) * (pi33.' * p3)) * eye(3, 2), ...
             ((pi31.' * p1) * (pi32.') * (w3.' * p3) + ((pi31.' * p1) * (pi12.') - (pi21.' * p1) * (pi32.')) * (pi33.' * p3)) * eye(3, 2), ...
             ((pi31.' * p1) * (pi32.' * p2) * (w3.') + ((pi31.' * p1) * (pi12.' * p2) - (pi21.' * p1) * (pi32.' * p2)) * (pi33.')) * eye(3, 2)];
        B(ind2 + 5, ind + (1:6)) = ...
            [((pi41.') * (pi42.' * p2) * (w3.' * p3) + ((pi41.') * (pi12.' * p2) - (pi21.') * (pi42.' * p2)) * (pi43.' * p3)) * eye(3, 2), ...
             ((pi41.' * p1) * (pi42.') * (w3.' * p3) + ((pi41.' * p1) * (pi12.') - (pi21.' * p1) * (pi42.')) * (pi43.' * p3)) * eye(3, 2), ...
             ((pi41.' * p1) * (pi42.' * p2) * (w3.') + ((pi41.' * p1) * (pi12.' * p2) - (pi21.' * p1) * (pi42.' * p2)) * (pi43.')) * eye(3, 2)];
    end

    D = zeros(11, 0);

end

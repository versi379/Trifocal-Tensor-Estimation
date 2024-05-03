% This function computes the rotation and translation errors (angular errors).

function [rot_err, t_err] = AngErrors(R_t_true, R_t_est)

    R_true = R_t_true(:, 1:3); t_true = R_t_true(:, 4);
    R_est = R_t_est(:, 1:3); t_est = R_t_est(:, 4);

    % Determine angle difference between rotations
    rot_err = abs(180 * acos((trace(R_true.' * R_est) - 1) / 2) / pi);

    % Determine angle difference between translations
    t_err = abs(180 * acos(dot(t_est / norm(t_est), t_true / norm(t_true))) / pi);

end

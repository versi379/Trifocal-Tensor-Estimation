function [R_t_2, R_t_3, Reconst, T, iter] = LinearTFTPoseEstimation(Corresp, CalM)

    % Normalization of the data
    [x1, Normal1] = Normalize2DPoints(Corresp(1:2, :));
    [x2, Normal2] = Normalize2DPoints(Corresp(3:4, :));
    [x3, Normal3] = Normalize2DPoints(Corresp(5:6, :));

    % Model to estimate T: linear equations
    T = linearTFT(x1, x2, x3);

    % tensor denormalization
    T = transform_TFT(T, Normal1, Normal2, Normal3, 1);

    % Find orientation using calibration and TFT
    [R_t_2, R_t_3] = R_t_from_TFT(T, CalM, Corresp);

    % Find 3D points by triangulation
    Reconst = triangulation3D({CalM(1:3, :) * eye(3, 4), CalM(4:6, :) * R_t_2, CalM(7:9, :) * R_t_3}, Corresp);
    Reconst = Reconst(1:3, :) ./ repmat(Reconst(4, :), 3, 1);

    iter = 0;

end

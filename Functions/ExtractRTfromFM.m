% This function extracts rotation and translation from the FM.
function [R_f, t_f] = ExtractRTfromFM(F, K1, K2, x1, x2)

    E = K2.' * F * K1;
    W = [0 -1 0; 1 0 0; 0 0 1];
    [U, ~, V] = svd(E);
    R = U * W * V.'; Rp = U * W.' * V.';
    R = R * sign(det(R)); Rp = Rp * sign(det(Rp));
    t = U(:, 3);

    num_points_seen = 0;

    for k = 1:4

        if k == 2 || k == 4
            t = -t;
        elseif k == 3
            R = Rp;
        end

        X1 = Triangulate3DPoints({[K1 [0; 0; 0]], K2 * [R, t]}, [x1; x2]); X1 = X1 ./ repmat(X1(4, :), 4, 1);
        X2 = [R t] * X1;

        if sum(sign(X1(3, :)) + sign(X2(3, :))) >= num_points_seen
            R_f = R; t_f = t;
            num_points_seen = sum(sign(X1(3, :)) + sign(X2(3, :)));
        end

    end

end

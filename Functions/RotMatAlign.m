% This function computes the rotation matrix that rotates
% vector u to align with vector v.

function R = RotMatAlign(u, v)

    if size(u, 1) == 1
        u = u.';
    end

    if size(v, 1) == 1
        v = v.';
    end

    u = u / norm(u); v = v / norm(v);
    w = cross(u, v);
    s = norm(w);
    c = dot(u, v);
    w = w / s;
    R = c * eye(3) + s * CrossProdMatrix(w) + (1 - c) * (w * w.');

end
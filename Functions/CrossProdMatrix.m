% This function computes the 3x3 matrix corresponding to the
% cross product of vector v.

function M = CrossProdMatrix(v)
    M = [0 -v(3) v(2); v(3) 0 -v(1); -v(2) v(1) 0];
end

function T = TFTfromProj(P1, P2, P3)

    T = zeros(3, 3, 3);

    for i = 1:3

        for j = 1:3

            for k = 1:3
                T(j, k, i) = (-1) ^ (i + 1) * det([P1([1:(i - 1) (i + 1):3], :); P2(j, :); P3(k, :)]);
            end

        end

    end

    T = T / norm(T(:));
end

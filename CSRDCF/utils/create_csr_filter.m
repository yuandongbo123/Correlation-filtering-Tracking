function H = create_csr_filter(img, Y, P)
% CREATE_CSR_FILTER
% create filter with Augmented Lagrangian iterative optimization method   ADMM
% input parameters:
% img: image patch (already normalized)
% Y: gaussian shaped labels (note that the peak must be at the top-left corner)
% P: padding mask (zeros around, 1 around the center), shape: box
% lambda: regularization parameter, i.e. 10e-2

mu = 5;
beta =  3;
mu_max = 20;
max_iters = 4;%这里减少迭代的次数=2并不会对跟踪的速度有太大的影响
lambda = mu/100;

F = fft2(img);%这里的img是特征50x50x29;  Y是对应的gaussion标签
Sxy = bsxfun(@times, F, conj(Y));
Sxx = F.*conj(F);%  对应论文里面的公式

% mask filter  这里的P就是滤波器
H = fft2(bsxfun(@times, ifft2(bsxfun(@rdivide, Sxy, (Sxx + lambda))), P));%这里好像是按照KCF滤波器的公式取优化的
% initialize lagrangian multiplier
L = zeros(size(H));

iter = 1;
while true
    G = (Sxy +mu*H - L) ./ (Sxx + mu);%这里的G就是hc
    H = fft2(real((1/(lambda + mu)) * bsxfun(@times, P, ifft2(mu*G + L))));%求hm，

    % stop optimization after fixed number of steps
    if iter >= max_iters
        break;
    end
    
    % update variables for next iteration
    L = L + mu*(G - H);
    mu = min(mu_max, beta*mu);
    iter = iter + 1;
end

end  % endfunction

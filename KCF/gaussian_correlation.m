function kf = gaussian_correlation(xf, yf, sigma)
%快速计算核函数

%在tracker.m中xf是 
%   GAUSSIAN_CORRELATION Gaussian Kernel at all shifts, i.e. kernel correlation.
%   Evaluates a Gaussian kernel with bandwidth SIGMA for all relative
%   shifts between input images X and Y, which must both be MxN.
%   用输入图像X和Y之间的所有相对位移来评估带宽为SIGMA的高斯内核
%   They must also be periodic (ie., pre-processed with a cosine window). 他们也必须是定期的
%   The result is an MxN map of responses.结果是一个MxN响应图。  
%   Inputs and output are all in the Fourier domain.  
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/            
	
	N = size(xf,1) * size(xf,2);        
	xx = xf(:)' * xf(:) / N;  %squared norm of x x的平方规范
	yy = yf(:)' * yf(:) / N;  %squared norm of y
	
	%cross-correlation term in Fourier domain   傅立叶域中的互相关项
	xyf = xf .* conj(yf);  %：y=conj(x)函数计算复数x的共轭值         
	xy = sum(real(ifft2(xyf)), 3); 
    %to spatial domain
    %ifft2反傅里叶变换 real（Z） 取Z的实数部分
    
	%calculate gaussian response for all positions, then go back to the Fourier domain
	%计算所有位置的高斯响应，然后返回傅立叶域
    
	kf = fft2(exp(-1 / sigma^2 * max(0, (xx + yy - 2 * xy) / numel(xf))));
    %傅里叶变换  公式里sigma就是方差
end


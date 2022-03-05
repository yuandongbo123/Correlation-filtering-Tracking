function kf = linear_correlation(xf, yf)
%LINEAR_CORRELATION Linear Kernel at all shifts, i.e. correlation.
%LINEAR_CORRELATION所有位移的线性内核，即相关。
%   Computes the dot-product for all relative shifts between input images
%   对所有输入图像及相关转移计算其点积，
%   X and Y, which must both be MxN. They must also be periodic定期
%   (ie.,pre-processed with a cosine window). 
%   用cos窗解决的是循环导致的边界问题。
   
%   The result is an MxN map of responses.
%   Inputs and output are all in the Fourier domain.
%   输入和输出
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/
	
	%cross-correlation term in Fourier domain傅立叶域中的互相关项
	kf = sum(xf .* conj(yf), 3) / numel(xf);
    %conj函数：用于计算复数的共轭值
    %用法说明：y=conj(x)函数计算复数x的共轭值。输出结果y的维数跟输入x的维数一致，返回值为：real(y)-i*imag(y)
end


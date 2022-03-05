function kf = gaussian_correlation(xf, yf, sigma)
%���ټ���˺���

%��tracker.m��xf�� 
%   GAUSSIAN_CORRELATION Gaussian Kernel at all shifts, i.e. kernel correlation.
%   Evaluates a Gaussian kernel with bandwidth SIGMA for all relative
%   shifts between input images X and Y, which must both be MxN.
%   ������ͼ��X��Y֮����������λ������������ΪSIGMA�ĸ�˹�ں�
%   They must also be periodic (ie., pre-processed with a cosine window). ����Ҳ�����Ƕ��ڵ�
%   The result is an MxN map of responses.�����һ��MxN��Ӧͼ��  
%   Inputs and output are all in the Fourier domain.  
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/            
	
	N = size(xf,1) * size(xf,2);        
	xx = xf(:)' * xf(:) / N;  %squared norm of x x��ƽ���淶
	yy = yf(:)' * yf(:) / N;  %squared norm of y
	
	%cross-correlation term in Fourier domain   ����Ҷ���еĻ������
	xyf = xf .* conj(yf);  %��y=conj(x)�������㸴��x�Ĺ���ֵ         
	xy = sum(real(ifft2(xyf)), 3); 
    %to spatial domain
    %ifft2������Ҷ�任 real��Z�� ȡZ��ʵ������
    
	%calculate gaussian response for all positions, then go back to the Fourier domain
	%��������λ�õĸ�˹��Ӧ��Ȼ�󷵻ظ���Ҷ��
    
	kf = fft2(exp(-1 / sigma^2 * max(0, (xx + yy - 2 * xy) / numel(xf))));
    %����Ҷ�任  ��ʽ��sigma���Ƿ���
end


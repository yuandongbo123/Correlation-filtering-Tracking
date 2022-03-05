function kf = linear_correlation(xf, yf)
%LINEAR_CORRELATION Linear Kernel at all shifts, i.e. correlation.
%LINEAR_CORRELATION����λ�Ƶ������ںˣ�����ء�
%   Computes the dot-product for all relative shifts between input images
%   ����������ͼ�����ת�Ƽ���������
%   X and Y, which must both be MxN. They must also be periodic����
%   (ie.,pre-processed with a cosine window). 
%   ��cos���������ѭ�����µı߽����⡣
   
%   The result is an MxN map of responses.
%   Inputs and output are all in the Fourier domain.
%   ��������
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/
	
	%cross-correlation term in Fourier domain����Ҷ���еĻ������
	kf = sum(xf .* conj(yf), 3) / numel(xf);
    %conj���������ڼ��㸴���Ĺ���ֵ
    %�÷�˵����y=conj(x)�������㸴��x�Ĺ���ֵ��������y��ά��������x��ά��һ�£�����ֵΪ��real(y)-i*imag(y)
end


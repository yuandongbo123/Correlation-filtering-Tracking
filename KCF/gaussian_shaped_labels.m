function labels = gaussian_shaped_labels(sigma, sz)
%gaussian_shaped_labels.m�Ǹ�ѭ������������ÿһ���������ϱ�ǩ����ǩֵ��0~1
%KCF ��ֱ��ʹ��[0,1]������������������ֵ
%  sigma��ʵ��  ��output_sigma_factor �������Ŀ���С�Ļع�Ŀ��Ŀռ����
%  sz��ʵ��     floor(window_sz / cell_size)
%GAUSSIAN_SHAPED_LABELS
%   Gaussian-shaped labels for all shifts of a sample.%��˹�α�ǩΪ������λ��������
%   LABELS = GAUSSIAN_SHAPED_LABELS(SIGMA, SZ)
%   ������ͼ��X��Y֮����������λ������������ΪSIGMA�ĸ�˹�ں�
%   Creates an array of labels (regression targets) for all shifts of a
%   sample of dimensions SZ.Ϊ�ߴ�SZ������������λ�ƴ���һ���ǩ���ع�Ŀ�꣩��
%   The output will have size SZ, representingone label for each possible shift. 
%   ��������д�СSZ������ÿ��������λ�ı�ǩ��
%   The labels will be Gaussian-shaped,with the peak at 0-shift (top-left element of the array),
%   decaying as the distance increases, and wrapping around at the borders.
%   ��ǩ���Ǹ�˹�Σ���0λ�ƵĶ��㣨���е����Ϸ�Ԫ�أ������ž�������Ӷ�˥�������ڱ߽���Χ���ơ�
%   The Gaussian function has spatial bandwidth SIGMA.   ��˹�������пռ����SIGMA
% 	as a simple example, the limit sigma = 0 would be a Dirac delta,
	%evaluate a Gaussian with the peak at the center element
    %������Ԫ�صķ�ֵ������˹
    
    %�ú�������һ����СΪsz�ĸ�˹��ǩ���󣬴�ͳ�������궨�����У����������Ϊ1�����������Ϊ0
    %���ڸ����в��ܺܺõı�ʾ������Ŀ������ƶȡ�KCF����ǩ��0-1�任Ϊ[0,1]����m
    
	[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
    %ndgrid���ɶ�ά�����Ͳ�ֵ������     floor�ú������Զ���ֵ����ȡ������
        
	labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2));
    
	%move the peak to the top-left, with wrap-around
    
	%labels = circshift(labels, -floor(sz(1:2) / 2) + 1);%��ֵ���ƶ������Ͻ�
    %circshift������matlab�б�ʾѭ����λ�ĺ���
    
	%sanity check: make sure it's really at top-left
	%assert(labels(1,1) == 1)
    %matlab��assert���������ж�һ��expression�Ƿ�������粻�����򱨴�
end


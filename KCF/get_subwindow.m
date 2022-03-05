function out = get_subwindow(im, pos, sz)%imͼ��pos����λ�ã�sz�ߴ��С
              %get_subwindow���ͼ����ȤĿ���Ӵ��ں�����

%GET_SUBWINDOW Obtain sub-window from image, with replication-padding.
%   GET_SUBWINDOWͨ����������ͼ���ȡ�Ӵ��ڡ�
%   Returns sub-window of image IM centered at POS ([y, x] coordinates),
%   with size SZ ([height, width]).
%   ��POS��[y��x]���꣩Ϊ���ĵ�ͼ��IM���Ӵ��ڷ��أ��ߴ�ΪSZ��[height��width]����
%   If any pixels are outside of the image,they will replicate the values at the borders.
%   ����κ�������ͼ��֮�⣬���ǽ����Ʊ߽��ϵ�ֵ��
%   get_subwindow��ͼ���л��һ���Ӵ��ڣ��ø������
%   ������POSΪ���ĵ�ͼ��IM�ĳߴ�ΪSZ���Ӵ���
%   ����κ�������ͼ��֮�⣬���ǽ����Ʊ߽��ϵ�ֵ��
%

    if isscalar(sz),  %square sub-window  Isscalar �ж�Ϊ����
		sz = [sz, sz];
	end
	xs = floor(pos(2)) + (1:sz(2)) - floor(sz(2)/2);%����ȡ��
	%xs = floor(pos(1)) + (1:sz(1)) - floor(sz(1)/2);%floorȡ��
	ys = floor(pos(1)) + (1:sz(1)) - floor(sz(1)/2);
	
	%check for out-of-bounds coordinates, and set them to the values at the borders
    %���������꣬����������Ϊ�߽��ϵ�ֵ
	xs(xs < 1) = 1;
	ys(ys < 1) = 1;
	xs(xs > size(im,2)) = size(im,2);%c=size(A,2) ����䷵�ص�ʱ����A��������
	ys(ys > size(im,1)) = size(im,1);%c=size(A,1) ����䷵�ص�ʱ����A��������
	
	%extract image
	out = im(ys, xs, :);

end


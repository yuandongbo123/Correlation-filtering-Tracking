function out = get_subwindow(im, pos, sz)%im图像，pos中心位置，sz尺寸大小
              %get_subwindow获得图像兴趣目标子窗口函数，

%GET_SUBWINDOW Obtain sub-window from image, with replication-padding.
%   GET_SUBWINDOW通过复制填充从图像获取子窗口。
%   Returns sub-window of image IM centered at POS ([y, x] coordinates),
%   with size SZ ([height, width]).
%   以POS（[y，x]坐标）为中心的图像IM的子窗口返回，尺寸为SZ（[height，width]）。
%   If any pixels are outside of the image,they will replicate the values at the borders.
%   如果任何像素在图像之外，它们将复制边界上的值。
%   get_subwindow从图像中获得一个子窗口，用复制填充
%   返回以POS为中心的图像IM的尺寸为SZ的子窗口
%   如果任何像素在图像之外，它们将复制边界上的值。
%

    if isscalar(sz),  %square sub-window  Isscalar 判断为标量
		sz = [sz, sz];
	end
	xs = floor(pos(2)) + (1:sz(2)) - floor(sz(2)/2);%向下取整
	%xs = floor(pos(1)) + (1:sz(1)) - floor(sz(1)/2);%floor取整
	ys = floor(pos(1)) + (1:sz(1)) - floor(sz(1)/2);
	
	%check for out-of-bounds coordinates, and set them to the values at the borders
    %检查外界坐标，并将其设置为边界上的值
	xs(xs < 1) = 1;
	ys(ys < 1) = 1;
	xs(xs > size(im,2)) = size(im,2);%c=size(A,2) 该语句返回的时矩阵A的列数。
	ys(ys > size(im,1)) = size(im,1);%c=size(A,1) 该语句返回的时矩阵A的行数。
	
	%extract image
	out = im(ys, xs, :);

end


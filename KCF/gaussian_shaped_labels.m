function labels = gaussian_shaped_labels(sigma, sz)
%gaussian_shaped_labels.m是给循环样本矩阵中每一个样本加上标签，标签值是0~1
%KCF 是直接使用[0,1]内连续的数作样本的值
%  sigma的实参  是output_sigma_factor 是相对于目标大小的回归目标的空间带宽
%  sz的实参     floor(window_sz / cell_size)
%GAUSSIAN_SHAPED_LABELS
%   Gaussian-shaped labels for all shifts of a sample.%高斯形标签为所有移位的样本。
%   LABELS = GAUSSIAN_SHAPED_LABELS(SIGMA, SZ)
%   用输入图像X和Y之间的所有相对位移来评估带宽为SIGMA的高斯内核
%   Creates an array of labels (regression targets) for all shifts of a
%   sample of dimensions SZ.为尺寸SZ的样本的所有位移创建一组标签（回归目标）。
%   The output will have size SZ, representingone label for each possible shift. 
%   输出将具有大小SZ，代表每个可能移位的标签。
%   The labels will be Gaussian-shaped,with the peak at 0-shift (top-left element of the array),
%   decaying as the distance increases, and wrapping around at the borders.
%   标签将是高斯形，在0位移的顶点（阵列的左上方元素），随着距离的增加而衰减，并在边界周围环绕。
%   The Gaussian function has spatial bandwidth SIGMA.   高斯函数具有空间带宽SIGMA
% 	as a simple example, the limit sigma = 0 would be a Dirac delta,
	%evaluate a Gaussian with the peak at the center element
    %用中心元素的峰值评估高斯
    
    %该函数生成一个大小为sz的高斯标签矩阵，传统的样本标定方法中，正样本标记为1，负样本标记为0
    %但在跟踪中不能很好的表示样本对目标的相似度。KCF将标签从0-1变换为[0,1]距离m
    
	[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
    %ndgrid生成多维函数和插值的数组     floor该函数可以对数值进行取整运算
        
	labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2));
    
	%move the peak to the top-left, with wrap-around
    
	%labels = circshift(labels, -floor(sz(1:2) / 2) + 1);%峰值点移动到左上角
    %circshift函数是matlab中表示循环移位的函数
    
	%sanity check: make sure it's really at top-left
	%assert(labels(1,1) == 1)
    %matlab中assert函数用来判断一个expression是否成立，如不成立则报错
end


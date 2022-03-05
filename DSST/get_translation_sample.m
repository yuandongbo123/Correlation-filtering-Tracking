function out = get_translation_sample(im, pos, model_sz, currentScaleFactor, cos_window)

% out = get_subwindow(im, pos, model_sz, currentScaleFactor, cos_window)
% 
% Extracts the a sample for the translation filter at the current
% location and scale.

if isscalar(model_sz),  %square sub-window 判断是否输出为标量
    model_sz = [model_sz, model_sz];
end
%滤波器的大小(即第一帧目标大小+padding)*尺度因子=patch_sz，根据patch_sz从图像中提取样本im_patch
%提取im_patch再转换到滤波器的大小（最终计算response的时候，样本大小和滤波器大小必须相等）
patch_sz = floor(model_sz * currentScaleFactor);

%make sure the size is not to small
if patch_sz(1) < 1
    patch_sz(1) = 2;
end
if patch_sz(2) < 1
    patch_sz(2) = 2;
end

%pos是坐标在目标中的真实位置；xs和ys分别代表目标区域内像素的x和y坐标值
xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);
% check for out-of-bounds coordinates, and set them to the values at
% the borders 这里的防止越界的处理非常重要
xs(xs < 1) = 1;
ys(ys < 1) = 1;
xs(xs > size(im,2)) = size(im,2);
ys(ys > size(im,1)) = size(im,1);

% extract image
im_patch = im(ys, xs, :);%row,col,channel

% resize image to model size  model size是第一帧的加了padding之后的图像大小
im_patch = mexResize(im_patch, model_sz, 'auto');%将patch resize到指定的大小

% compute feature map
out = get_feature_map(im_patch);

% apply cosine window
out = bsxfun(@times, cos_window, out);
end


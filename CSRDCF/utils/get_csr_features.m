function [out num_feat_ch] = get_csr_features(img, c, scale, template_size, ...
    resize_sz, cos_win, feature_type, w2c, cell_size)
 %cos_win= rescale_template_size/cell_size;
 % extract features  c=没有resize的目标中心的坐标 template_size=原始图的搜索区域大小  rescale_template_size=resize之后的搜索区域的大小

% calculate size of the patch
w = floor(scale*template_size(1));%计算 原始搜索区域的宽和高
h = floor(scale*template_size(2));
% extract indexes %c代表
xs = floor(c(1)) + (1:w) - floor(w/2);
ys = floor(c(2)) + (1:h) - floor(h/2);
% indexes outside of image: use border pixels
xs(xs < 1) = 1;
ys(ys < 1) = 1;
xs(xs > size(img,2)) = size(img,2);
ys(ys > size(img,1)) = size(img,1);
% extract from image
im = img(ys, xs, :);%这里是从原始图像中把原始的搜索区域给抠出来
% resize to reference size
im = imresize(im, resize_sz([2, 1]));%原始的搜索区域如果面积小于40000，则resize到200x200,反之则保持不变

% hog features
nHogChan = 18;%注意这里的hog特征只是提取了18个维度上的

% compute num. feature channels
num_feat_ch = 0;
feat_gray = false; feat_hog = false; feat_cn = false; feat_color = false; feat_hsv = false;
if sum(strcmp(feature_type, 'hog'))
    num_feat_ch = num_feat_ch + nHogChan;%hog特征18个通道
    feat_hog = true;
end
if sum(strcmp(feature_type, 'gray'))
    num_feat_ch = num_feat_ch + 1;  %hog特征是1个通道
    feat_gray = true;
end
if sum(strcmp(feature_type, 'cn'))
    num_feat_ch = num_feat_ch + size(w2c,2);%cn特征只使用了10个通道
    feat_cn = true;
end

if feat_hog%如果使用了hog特征就要考虑cell的大小
    out_size = floor([size(im, 1) size(im, 2)] ./ cell_size);
else %如果没有使用hog特征，则outsize就是resize之后的搜索区域
    out_size = [size(im, 1) size(im, 2)];
end

out = zeros(out_size(1), out_size(2), num_feat_ch);%num_feat_ch代表特征的维度
channel_id = 1;
%---------------------------------提取hog特征，gray特征和cn特征---------------------------------------%
% extract features from image  
if feat_hog
    % extract HoG features
    nOrients = 9;
	hog_image = fhog(single(im), cell_size, nOrients);
    % put HoG features into output structure
    out(:,:,channel_id:(channel_id + nHogChan - 1)) = hog_image(:,:,1:nHogChan);
    channel_id = channel_id + nHogChan;
end

if feat_gray
    % prepare grayscale patch
	if size(im,3) > 1
		gray_patch = rgb2gray(im);
	else
		gray_patch = im;
    end
    % resize it to out size
	gray_patch = imresize(gray_patch, out_size);
    % put grayscale channel into output structure
    out(:, :, channel_id) = single((gray_patch / 255) - 0.5);
    channel_id = channel_id + 1;
end

if feat_cn
    % extract ColorNames features
    CN = im2c(single(im), w2c, -2);
    CN = imresize(CN, out_size);
    % put colornames features into output structure
    out(:,:,channel_id:(channel_id + size(w2c, 2) - 1)) = CN;
    channel_id = channel_id + size(w2c,2);
end
%---------------------------------提取hog特征，gray特征和cn特征---------------------------------------%
% multiply with cosine window
if ~isempty(cos_win)
    out = bsxfun(@times, out, cos_win);% out=50x50x29  cos_win=50x50 , 对提取的特征添加cos_window;
end

end  % endfunction

function tracker = create_csr_tracker(img, init_bbox, init_params)

    if nargin < 3
        init_params = read_default_csr_parameters();
    end

    % transform polygon to axis-aligned bbox
    if numel(init_bbox) > 4
        bb8 = round(init_bbox(:));
        x1 = round(min(bb8(1:2:end))); % 将带有旋转的矩形框转换成正矩形框
        x2 = round(max(bb8(1:2:end)));
        y1 = round(min(bb8(2:2:end)));
        y2 = round(max(bb8(2:2:end)));
        bb = round([x1, y1, x2 - x1, y2 - y1]);%x,y,w,h
        %初始化
        init_mask = poly2mask(bb8(1:2:end)-bb(1), bb8(2:2:end)-bb(2), bb(4), bb(3));
    else%imshow(init_mask);
        bb = round(init_bbox);
        init_mask = ones(bb(4), bb(3));% h,w,全是1
    end

    % filter parameters
    padding = init_params.padding;  % padding=3
    learning_rate = init_params.learning_rate;  % learning rate for updating filter= 0.02
    feature_type  = init_params.feature_type;%hog cn gray

    % load and store pre-computed lookup table for colornames
    w2c = [];
    if sum(strcmp(feature_type, 'cn'))%加载cn模型
        w2c = load('w2crs.mat');
        w2c = w2c.w2crs;
    end

    % segmentation parameters
    hist_lr = init_params.hist_lr;  % 0.04
    nbins = init_params.nbins;  % N bins for segmentation 16
    seg_colorspace = init_params.seg_colorspace;     % 'rgb' or 'hsv' hsv kongjain
    use_segmentation = init_params.use_segmentation;  % false to disable use of segmentation
    mask_diletation_type =  init_params.mask_diletation_type;  % for function strel (square, disk, ...) disk
    mask_diletation_sz = init_params.mask_diletation_sz;

    % check if grayscale image (only 1 channel) or
    % check if grayscale image (3 the same channels)
    img0 = bsxfun(@minus, double(img), mean(img,3));%减去均值
    if size(img,3) < 3 || sum(abs(img0(:))) < 10    %如果只有一个通道或者三个通道相同
        use_segmentation = false;
        % also do not use colornames
        [isused_cn, cn_idx] = ismember('cn', feature_type);
        if isused_cn
            feature_type(cn_idx) = [];
        end
    end

    % features parameters
    cell_size = 1.0;
    if sum(strcmp(feature_type, 'hog'))
        cell_size = min(4, max(1, ceil((bb(3)*bb(4))/400)));% cell_size=4
    end

    % size parameters
    % reference target size: [width, height]
    base_target_sz = [bb(3), bb(4)];% [w,h] 45x111 目标区域实际的宽和高
    % reference template size: [w, h], does not change during tracking
    template_size = floor(base_target_sz + padding*sqrt(prod(base_target_sz)));% 45 x 111  3倍padding的大小
    template_size = mean(template_size);%对搜索区域的大小进行求均值 获取矩形区域的搜索区域
    template_size = [template_size, template_size];%  template_size  是扩大搜索区域再开方之后的大小
    %template_size=290,290,如果实际的搜索区域大于200x200的话就进行resize，实际上是进行缩放
    % rescale template after extracting to have fixed area
    rescale_ratio = sqrt((200^2) / (template_size(1) * template_size(2)));%对目标进行缩放resize=0.6897
    if rescale_ratio > 1  % if already smaller - do not rescale
        rescale_ratio = 1;
    end
    %rescale_template_size这个是对区域进行缩放，如果搜索区域大于200x200就缩放得到这个尺度，如果小于200x200则搜索区域不变 rescale_scale=1
    rescale_template_size = floor(rescale_ratio * template_size);% zhe li zhi dui mu biao jin xing suo fang 200x200

    % position of the target center
    c = bb([1,2]) + base_target_sz/2;% 求目标的中心坐标

    % create gaussian shaped labels
    sigma = init_params.y_sigma;%sigma=1
    Y = fft2(gaussian_shaped_labels(1,sigma, floor(rescale_template_size([2,1]) / cell_size)));
    %imshow(Y); %这个标签Y的大小是对应搜索区域的 area/cell 之后的大小
    %store pre-computed cosine window
    cos_win = hann(size(Y,1)) * hann(size(Y,2))';%添加cos_window
    %cos_win的大小和Y 的大小是一样的
    % scale adaptation parameters (from DSST)
    currentScaleFactor = init_params.currentScaleFactor;
    n_scales = init_params.n_scales;
    scale_model_factor = init_params.scale_model_factor;
    scale_sigma_factor = init_params.scale_sigma_factor;
    scale_step = init_params.scale_step;
    scale_model_max_area = init_params.scale_model_max_area;
    scale_sigma = sqrt(n_scales) * scale_sigma_factor;
    scale_lr = init_params.scale_lr;     % learning rate parameter

    %label function for the scales
    ss = (1:n_scales) - ceil(n_scales/2);
    ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
    ysf = single(fft(ys));

    if mod(n_scales,2) == 0
        scale_window = single(hann(n_scales+1));
        scale_window = scale_window(2:end);
    else
        scale_window = single(hann(n_scales));
    end

    ss = 1:n_scales;
    scaleFactors = scale_step.^(ceil(n_scales/2) - ss);%1-33个尺度
    %以上这些都是DSST的尺度设计，
    template_size_ = template_size;%这个是没有resize的模板的大小  template_size=290 x 290是没有resize的大小   
    if scale_model_factor^2 * prod(template_size_) > scale_model_max_area
        scale_model_factor = sqrt(scale_model_max_area/prod(template_size_));
    end %在进行尺度估计的时候搜索区域不能太大，最大区域是512；否则就对这个区域进行缩放 

    scale_model_sz = floor(template_size_ * scale_model_factor);%获得尺度估计的model的大小，
    scaleSizeFactors = scaleFactors;  %获取33个尺度因子，并设置最大最小尺度因子
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ template_size_)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(img,1) size(img,2)] ./ base_target_sz)) / log(scale_step));

    % create dummy mask (approximation for segmentation)
    % size of the object in feature space
    obj_size = floor(rescale_ratio * (base_target_sz/cell_size));%目标区域进行resize之后的大小
    x0 = floor((size(Y,2)-obj_size(1))/2);
    y0 = floor((size(Y,1)-obj_size(2))/2);%这里是响应图的右上角的点的坐标
    x1 = x0 + obj_size(1);%宽
    y1 = y0 + obj_size(2);%高   这是响应图的左下角的坐标
    target_dummy_mask = zeros(size(Y));% Y是缩放之后的目标/cell_size
    target_dummy_mask(y0:y1, x0:x1) = 1;%把目标区域设置为1，非目标区域设置为0
    target_dummy_mask = single(target_dummy_mask);%imshow(target_dummy_mask);

    target_dummy_area = sum(target_dummy_mask(:));%求取目标区域的面积
    
    if use_segmentation
        % convert image in desired colorspace
        if strcmp(seg_colorspace, 'rgb')
            seg_img = img;
        elseif strcmp(seg_colorspace, 'hsv')
            seg_img = rgb2hsv(img);
            seg_img = seg_img * 255;
        else
            error('Unknown colorspace parameter');
        end

        % object rectangle region (to zero-based coordinates)
        obj_reg = [bb(1), bb(2), bb(1)+bb(3), bb(2)+bb(4)] - [1 1 1 1];

        % extract histograms
        hist_fg = mex_extractforeground(seg_img, obj_reg, nbins);
        hist_bg = mex_extractbackground(seg_img, obj_reg, nbins);

        % extract masked patch: mask out parts outside image
        [seg_patch, valid_pixels_mask] = get_patch(seg_img, c, currentScaleFactor, template_size);

        % segmentation
        [fg_p, bg_p] = get_location_prior([1 1 size(seg_patch, 2) size(seg_patch, 1)], base_target_sz, [size(seg_patch,2), size(seg_patch, 1)]);
        [~, fg, ~] = mex_segment(seg_patch, hist_fg, hist_bg, nbins, fg_p, bg_p);

        % cut out regions outside from image
        mask = single(fg).*single(valid_pixels_mask);
        mask = binarize_softmask(mask);

        % use mask from init pose
        init_mask_padded = zeros(size(mask));
        pm_x0 = floor(size(init_mask_padded,2) / 2 - size(init_mask,2) / 2);
        pm_y0 = floor(size(init_mask_padded,1) / 2 - size(init_mask,1) / 2);
        init_mask_padded(pm_y0:pm_y0+size(init_mask,1)-1, pm_x0:pm_x0+size(init_mask,2)-1) = init_mask;
        mask = mask.*single(init_mask_padded);

        % resize to filter size
        mask = imresize(mask, size(Y), 'nearest');

        % check if mask is too small (probably segmentation is not ok then)
        if mask_normal(mask, target_dummy_area)
            if mask_diletation_sz > 0
                D = strel(mask_diletation_type, mask_diletation_sz);
                mask = imdilate(mask, D);
            end
        else
            mask = target_dummy_mask;
        end

    else
        mask = target_dummy_mask;%imshow(mask);搜索区域中只有目标区域是1，非目标区域是0
    end
    %cos_win= rescale_template_size/cell_size;
    % extract features  c=没有resize的目标中心的坐标 template_size=原始图的搜索区域大小  rescale_template_size=resize之后的搜索区域的大小
    f = get_csr_features(img, c, currentScaleFactor, template_size, ...
                rescale_template_size, cos_win, feature_type, w2c, cell_size);
    %f=50x50x29(在篮球测试的例程中)
    % create filter - using segmentation mask
    H = create_csr_filter(f, Y, single(mask));%论文创新点1，通过mask来确定目标区域和非目标区域，类似域目标分割
    %H是多通道的50x50x29
    % calculate per-channel feature weights   %论文创新点2 计算每个通道的权重
    response = real(ifft2(fft2(f).*conj(H)));%特征和滤波器卷积
    chann_w = max(reshape(response, [size(response,1)*size(response,2), size(response,3)]), [], 1);%找出每一列的最大值
    % normalize: sum = 1
    chann_w = chann_w / sum(chann_w);%归一化

    % make a scale search model aswell
    xs = get_scale_subwindow(img, c([2,1]), base_target_sz([2,1]), ...
        currentScaleFactor * scaleSizeFactors, scale_window, scale_model_sz([2,1]), []);
    % fft over the scale dim
    xsf = fft(xs,[],2);
    new_sf_num = bsxfun(@times, ysf, conj(xsf));
    new_sf_den = sum(xsf .* conj(xsf), 1);

    % store all important const's and variables to the tracker structure
    tracker.feature_type = feature_type;
    tracker.padding = padding;
    tracker.learning_rate = learning_rate;  % filter learning rate
    tracker.cell_size = cell_size;
    tracker.H = H;
    tracker.weight_lr = init_params.channels_weight_lr;
    tracker.use_channel_weights = init_params.use_channel_weights;
    tracker.chann_w = chann_w;
    tracker.Y = Y;
    tracker.mask_diletation_type = mask_diletation_type;
    tracker.mask_diletation_sz = mask_diletation_sz;
    tracker.target_dummy_mask = target_dummy_mask;
    tracker.target_dummy_area = target_dummy_area;
    tracker.use_segmentation = use_segmentation;
    tracker.bb = bb;
    tracker.cos_win = cos_win;
    tracker.w2c = w2c;
    tracker.template_size = template_size;
    tracker.obj_size = obj_size;
    tracker.c = c;
    tracker.nbins = nbins;
    tracker.currentScaleFactor = currentScaleFactor;
    tracker.rescale_template_size = rescale_template_size;
    tracker.rescale_ratio = rescale_ratio;
    if use_segmentation
        tracker.hist_fg = hist_fg;
        tracker.hist_bg = hist_bg;
        tracker.hist_lr = hist_lr;
        tracker.seg_colorspace = seg_colorspace;
    end
    tracker.ysf = ysf;
    tracker.sf_num = new_sf_num;
    tracker.sf_den = new_sf_den;
    tracker.scale_lr = scale_lr;
    tracker.base_target_sz = base_target_sz;
    tracker.scaleSizeFactors = scaleSizeFactors;
    tracker.scale_window = scale_window;
    tracker.scale_model_sz = scale_model_sz;
    tracker.scaleFactors = scaleFactors;
    tracker.min_scale_factor = min_scale_factor;
    tracker.max_scale_factor = max_scale_factor;
    tracker.mask = mask;
    
    tracker.H_prev = H;
    
    tracker.img_prev = img;

end  % endfunction

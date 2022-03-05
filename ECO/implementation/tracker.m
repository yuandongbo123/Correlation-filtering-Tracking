function results = tracker(params)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get sequence info     %序列信息
[seq, im] = get_sequence_info(params.seq,params.video_path); % im = 第一帧初始图像，并且会根据后面帧的变化更新
params = rmfield(params, 'seq');
if isempty(im)
    seq.rect_position = [];
    [seq, results] = get_sequence_results(seq);
    return;
end

% Init position
pos = seq.init_pos(:)';
target_sz = seq.init_sz(:)';
params.init_sz = target_sz;

% Feature settings
features = params.t_features;

% Set default parameters
params = init_default_params(params);

% Global feature parameters
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end
global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;

% Correct max number of samples
params.nSamples = min(params.nSamples, seq.num_frames);

% Define data types
if params.use_gpu
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end
params.data_type_complex = complex(params.data_type);
global_fparams.data_type = params.data_type;

init_target_sz = target_sz;

% Check if color image
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end

% Check if mexResize is available and show warning otherwise.
%检查mexResize是否可用，否则显示警告。
params.use_mexResize = true;
global_fparams.use_mexResize = true;
try
    [~] = mexResize(ones(5,5,3,'uint8'), [3 3], 'auto');
catch err
    warning('ECO:tracker', 'Error when using the mexResize function. Using Matlab''s interpolation function instead, which is slower.\nTry to run the compile script in "external_libs/mexResize/".\n\nThe error was:\n%s', getReport(err));
    params.use_mexResize = false;
    global_fparams.use_mexResize = false;
end

% Calculate search area and initial scale factor
%计算搜索区域和初始比例因子
search_area = prod(init_target_sz * params.search_area_scale); %prod - 数组元素的乘积 prod(A,dim) 默认dim=1
if search_area > params.max_image_sample_size
    currentScaleFactor = sqrt(search_area / params.max_image_sample_size);
elseif search_area < params.min_image_sample_size
    currentScaleFactor = sqrt(search_area / params.min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor( base_target_sz * params.search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * params.search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * params.search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*2]; % for testing
end

[features, global_fparams, feature_info] = init_features(features, global_fparams, is_color_image, img_sample_sz, 'odd_cells');

% Set feature info 
%得到特征信息
img_support_sz = feature_info.img_support_sz;
feature_sz = feature_info.data_sz;
feature_dim = feature_info.dim;
num_feature_blocks = length(feature_dim);

% Get feature specific parameters
%获取特征特定参数
feature_params = init_feature_params(features, feature_info);
feature_extract_info = get_feature_extract_info(features);

% Set the sample feature dimension
%设置样本特征尺寸
if params.use_projection_matrix
    sample_dim = feature_params.compressed_dim;
else
    sample_dim = feature_dim;
end

% Size of the extracted feature maps %提取的特征图的大小
feature_sz_cell = permute(mat2cell(feature_sz, ones(1,num_feature_blocks), 2), [2 3 1]);
% permute - 置换数组维度 ,B = permute(A,dimorder)
%  此 MATLAB 函数 按照向量 dimorder 指定的顺序重新排列数组的维度。例如，permute(A,[2 1]) 交换矩阵 A 的行和列维度。
% Number of Fourier coefficients to save for each filter layer. This will
%为每个滤光层节省的傅里叶系数的数目。这将是奇数
% be an odd number.
filter_sz = feature_sz + mod(feature_sz+1, 2);
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% The size of the label function DFT. Equal to the maximum filter size.
%标签函数DFT的大小。等于过滤器的最大尺寸。
[output_sz, k1] = max(filter_sz, [], 1);
k1 = k1(1);

% Get the remaining block indices
%获取剩下的块索引
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];

% How much each feature block has to be padded to the obtain output_sz
%每个特征块需要填充多少到获取的output_sz
pad_sz = cellfun(@(filter_sz) (output_sz - filter_sz) / 2, filter_sz_cell, 'uniformoutput', false);
%cellfun - 对元胞数组中的每个元胞应用函数

% Compute the Fourier series indices and their transposes
%计算傅里叶级数指标及其转置
ky = cellfun(@(sz) (-ceil((sz(1) - 1)/2) : floor((sz(1) - 1)/2))', filter_sz_cell, 'uniformoutput', false);
kx = cellfun(@(sz) -ceil((sz(2) - 1)/2) : 0, filter_sz_cell, 'uniformoutput', false);
% cellfun - 对元胞数组中的每个元胞应用函数 %对于 C 的第 i 个元素来说，A(i) = func(C{i})
% construct the Gaussian label function using Poisson formula
%用泊松公式构造高斯标号函数
sig_y = sqrt(prod(floor(base_target_sz))) * params.output_sigma_factor * (output_sz ./ img_support_sz);
yf_y = cellfun(@(ky) single(sqrt(2*pi) * sig_y(1) / output_sz(1) * exp(-2 * (pi * sig_y(1) * ky / output_sz(1)).^2)), ky, 'uniformoutput', false);
yf_x = cellfun(@(kx) single(sqrt(2*pi) * sig_y(2) / output_sz(2) * exp(-2 * (pi * sig_y(2) * kx / output_sz(2)).^2)), kx, 'uniformoutput', false);
yf = cellfun(@(yf_y, yf_x) cast(yf_y * yf_x, 'like', params.data_type), yf_y, yf_x, 'uniformoutput', false);

% construct cosine window
%构建余弦窗
cos_window = cellfun(@(sz) hann(sz(1)+2)*hann(sz(2)+2)', feature_sz_cell, 'uniformoutput', false);
cos_window = cellfun(@(cos_window) cast(cos_window(2:end-1,2:end-1), 'like', params.data_type), cos_window, 'uniformoutput', false);

% Compute Fourier series of interpolation function
%计算插值函数的傅里叶级数
[interp1_fs, interp2_fs] = cellfun(@(sz) get_interp_fourier(sz, params), filter_sz_cell, 'uniformoutput', false);

% Get the reg_window_edge parameter
%获取reg_window_edge参数
reg_window_edge = {};
for k = 1:length(features)
    if isfield(features{k}.fparams, 'reg_window_edge')
        reg_window_edge = cat(3, reg_window_edge, permute(num2cell(features{k}.fparams.reg_window_edge(:)), [2 3 1]));
    else
        reg_window_edge = cat(3, reg_window_edge, cell(1, 1, length(features{k}.fparams.nDim)));
    end
end

% Construct spatial regularization filter
%构造空间正则化滤波器
reg_filter = cellfun(@(reg_window_edge) get_reg_filter(img_support_sz, base_target_sz, params, reg_window_edge), reg_window_edge, 'uniformoutput', false);

% Compute the energy of the filter (used for preconditioner)
%计算过滤器的能量(用于预处理器)
reg_energy = cellfun(@(reg_filter) real(reg_filter(:)' * reg_filter(:)), reg_filter, 'uniformoutput', false);

if params.use_scale_filter
    [nScales, scale_step, scaleFactors, scale_filter, params] = init_scale_filter(params);
else
    % Use the translation filter to estimate the scale.
    %使用翻译过滤器来估计规模。
    nScales = params.number_of_scales;
    scale_step = params.scale_step;
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
    scaleFactors = scale_step .^ scale_exp;
end

if nScales > 0
    %force reasonable scale changes
    %力合理规模变化
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

% Set conjugate gradient uptions
%设置共轭梯度
init_CG_opts.CG_use_FR = true;
init_CG_opts.tol = 1e-6;
init_CG_opts.CG_standard_alpha = true;
init_CG_opts.debug = params.debug;
CG_opts.CG_use_FR = params.CG_use_FR;
CG_opts.tol = 1e-6;
CG_opts.CG_standard_alpha = params.CG_standard_alpha;
CG_opts.debug = params.debug;
if params.CG_forgetting_rate == Inf || params.learning_rate >= 1
    CG_opts.init_forget_factor = 0;
else
    CG_opts.init_forget_factor = (1-params.learning_rate)^params.CG_forgetting_rate;
end

seq.time = 0;

% Initialize and allocate
%初始化和分配
prior_weights = zeros(params.nSamples,1, 'single');
sample_weights = cast(prior_weights, 'like', params.data_type);
samplesf = cell(1, 1, num_feature_blocks);
if params.use_gpu
    % In the GPU version, the data is stored in a more normal way since we
    %在GPU版本中，数据以一种更正常的方式存储
    % dont have to use mtimesx. 不必使用mtimesx。
    for k = 1:num_feature_blocks
        samplesf{k} = zeros(filter_sz(k,1),(filter_sz(k,2)+1)/2,sample_dim(k),params.nSamples, 'like', params.data_type_complex);
    end
else
    for k = 1:num_feature_blocks
        samplesf{k} = zeros(params.nSamples,sample_dim(k),filter_sz(k,1),(filter_sz(k,2)+1)/2, 'like', params.data_type_complex);
    end
end

% Allocate
scores_fs_feat = cell(1,1,num_feature_blocks);

% Distance matrix stores the square of the euclidean distance between each pair of
%距离矩阵存储每对之间的欧几里得距离的平方
% samples. Initialise it to inf
distance_matrix = inf(params.nSamples, 'single');

% Kernel matrix, used to update distance matrix
%核矩阵，用于更新距离矩阵
gram_matrix = inf(params.nSamples, 'single');

latest_ind = [];
frames_since_last_train = inf;
num_training_samples = 0;

% Find the minimum allowed sample weight. Samples are discarded if their weights become lower
%找出最小允许的样本重量。如果样本的重量降低，它们就会被丢弃
params.minimum_sample_weight = params.learning_rate*(1-params.learning_rate)^(2*params.nSamples);

res_norms = [];
residuals_pcg = [];

while true
    % Read image
    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq,params.video_path);
        if isempty(im)
            break;
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
        seq.frame = 1;
    end

    tic();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Target localization step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Do not estimate translation and scaling on the first frame, since we 
    %不要在第一帧上估计平移和缩放，因为我们
    % just want to initialize the tracker there
    %这里只需要初始化跟踪器
    if seq.frame > 1
        old_pos = inf(size(pos));
        iter = 1;
        
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            % Extract features at multiple resolutions
            %在多个分辨率提取特征
            sample_pos = round(pos); %round - 四舍五入为最近的小数或整数
            det_sample_pos = sample_pos;
            sample_scale = currentScaleFactor*scaleFactors;
            xt = extract_features(im, sample_pos, sample_scale, features, global_fparams, feature_extract_info);
                        
            % Project sample
            xt_proj = project_sample(xt, projection_matrix);
            
            % Do windowing of features
            xt_proj = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt_proj, cos_window, 'uniformoutput', false);
            
            % Compute the fourier series
            %计算傅里叶级数
            xtf_proj = cellfun(@cfft2, xt_proj, 'uniformoutput', false);
            
            % Interpolate features to the continuous domain
            xtf_proj = interpolate_dft(xtf_proj, interp1_fs, interp2_fs);
            
            % Compute convolution for each feature block in the Fourier domain
            %将特征插值到连续区域
            % and the sum over all blocks.
            %然后对所有块求和。
            scores_fs_feat{k1} = sum(bsxfun(@times, hf_full{k1}, xtf_proj{k1}), 3);
            scores_fs_sum = scores_fs_feat{k1};
            for k = block_inds
                scores_fs_feat{k} = sum(bsxfun(@times, hf_full{k}, xtf_proj{k}), 3);
                scores_fs_sum(1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end-pad_sz{k}(2),1,:) = ...
                    scores_fs_sum(1+pad_sz{k}(1):end-pad_sz{k}(1), 1+pad_sz{k}(2):end-pad_sz{k}(2),1,:) + ...
                    scores_fs_feat{k};
            end
            
            % Also sum over all feature blocks.
            %还要对所有特征块进行求和。
            % Gives the fourier coefficients of the convolution response.
            %给出了卷积响应的傅里叶系数。
            scores_fs = permute(gather(scores_fs_sum), [1 2 4 3]);
            
            % Optimize the continuous score function with Newton's method.
            %用牛顿法优化连续积分函数。
            [trans_row, trans_col, scale_ind] = optimize_scores(scores_fs, params.newton_iterations);
            
            % Compute the translation vector in pixel-coordinates and round
            %以像素坐标和圆形计算平移向量
            % to the closest integer pixel.
            %到最近的整数像素。
            translation_vec = [trans_row, trans_col] .* (img_support_sz./output_sz) * currentScaleFactor * scaleFactors(scale_ind);
            scale_change_factor = scaleFactors(scale_ind);
            
            % update position 更新位置
            old_pos = pos;
            pos = sample_pos + translation_vec;
            
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end
            
            % Do scale tracking with the scale filter 用比例滤波器进行比例跟踪
            if nScales > 0 && params.use_scale_filter
                scale_change_factor = scale_filter_track(im, pos, base_target_sz, currentScaleFactor, scale_filter, params);
            end 
            
            % Update the scale 更新的尺度
            currentScaleFactor = currentScaleFactor * scale_change_factor;
            
            % Adjust to make sure we are not to large or to small
            %调整，以确保我们不是大或小
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
            
            iter = iter + 1;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Model update step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    % Extract sample and init projection matrix
    %提取样本并初始化投影矩阵
    if seq.frame == 1
        % Extract image region for training sample
        %提取图像区域作为训练样本
        sample_pos = round(pos);
        sample_scale = currentScaleFactor;
        xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
        
        % Do windowing of features
        %做窗口功能
        xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl, cos_window, 'uniformoutput', false);
        
        % Compute the fourier series
        %计算傅里叶级数
        xlf = cellfun(@cfft2, xlw, 'uniformoutput', false);
        
        % Interpolate features to the continuous domain
        xlf = interpolate_dft(xlf, interp1_fs, interp2_fs);
        
        % New sample to be added
        %要添加的新样本
        xlf = compact_fourier_coeff(xlf);
        
        % Shift sample 改变样品
        shift_samp = 2*pi * (pos - sample_pos) ./ (sample_scale * img_support_sz);
        xlf = shift_sample(xlf, shift_samp, kx, ky);
         
        % Init the projection matrix 初始化投影矩阵
        projection_matrix = init_projection_matrix(xl, sample_dim, params);
        
        % Project sample 项目样例
        xlf_proj = project_sample(xlf, projection_matrix);
        
        clear xlw
    elseif params.learning_rate > 0
        if ~params.use_detection_sample
            % Extract image region for training sample
            %提取图像区域作为训练样本
            sample_pos = round(pos);
            sample_scale = currentScaleFactor;
            xl = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_extract_info);
            
            % Project sample
            %项目样例
            xl_proj = project_sample(xl, projection_matrix);
            
            % Do windowing of features
            %做窗口功能
            xl_proj = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl_proj, cos_window, 'uniformoutput', false);
            
            % Compute the fourier series
            %计算傅里叶级数
            xlf1_proj = cellfun(@cfft2, xl_proj, 'uniformoutput', false);
            
            % Interpolate features to the continuous domain
            %将特征插值到连续区域
            xlf1_proj = interpolate_dft(xlf1_proj, interp1_fs, interp2_fs);
            
            % New sample to be added
            %要添加的新样本
            xlf_proj = compact_fourier_coeff(xlf1_proj);
        else
            if params.debug
                % Only for visualization
                %只对可视化
                xl = cellfun(@(xt) xt(:,:,:,scale_ind), xt, 'uniformoutput', false);
            end
            
            % Use the sample that was used for detection
            %使用用于检测的样品
            sample_scale = sample_scale(scale_ind);
            xlf_proj = cellfun(@(xf) xf(:,1:(size(xf,2)+1)/2,:,scale_ind), xtf_proj, 'uniformoutput', false);
        end
        
        % Shift the sample so that the target is centered
        %移动样本，使目标居中
        shift_samp = 2*pi * (pos - sample_pos) ./ (sample_scale * img_support_sz);
        xlf_proj = shift_sample(xlf_proj, shift_samp, kx, ky);
    end
    
    % The permuted sample is only needed for the CPU implementation
    %排列示例仅用于CPU实现
    if ~params.use_gpu
        xlf_proj_perm = cellfun(@(xf) permute(xf, [4 3 1 2]), xlf_proj, 'uniformoutput', false);
    end
        
    if params.use_sample_merge
        % Update the samplesf to include the new sample. The distance 更新样例以包含新样例。 的距离  
        % matrix, kernel matrix and prior weight are also updated  对矩阵、核矩阵和先验权值进行了更新  
        if params.use_gpu
            [merged_sample, new_sample, merged_sample_id, new_sample_id, distance_matrix, gram_matrix, prior_weights] = ...
                update_sample_space_model_gpu(samplesf, xlf_proj, distance_matrix, gram_matrix, prior_weights,...
                num_training_samples,params);
        else
            [merged_sample, new_sample, merged_sample_id, new_sample_id, distance_matrix, gram_matrix, prior_weights] = ...
                update_sample_space_model(samplesf, xlf_proj_perm, distance_matrix, gram_matrix, prior_weights,...
                num_training_samples,params);
        end
        
        if num_training_samples < params.nSamples
            num_training_samples = num_training_samples + 1;
        end
    else
        % Do the traditional adding of a training sample and weight update
        %传统的增加训练样本和重量更新 吃c -cot
        % of C-COT
        [prior_weights, replace_ind] = update_prior_weights(prior_weights, gather(sample_weights), latest_ind, seq.frame, params);
        latest_ind = replace_ind;
        
        merged_sample_id = 0;
        new_sample_id = replace_ind;
        if params.use_gpu
            new_sample = xlf_proj;
        else
            new_sample = xlf_proj_perm;
        end
    end
    
    if seq.frame > 1 && params.learning_rate > 0 || seq.frame == 1 && ~params.update_projection_matrix
        % Insert the new training sample
        %插入新的训练样本
        for k = 1:num_feature_blocks
            if params.use_gpu
                if merged_sample_id > 0
                    samplesf{k}(:,:,:,merged_sample_id) = merged_sample{k};
                end
                if new_sample_id > 0
                    samplesf{k}(:,:,:,new_sample_id) = new_sample{k};
                end
            else
                if merged_sample_id > 0
                    samplesf{k}(merged_sample_id,:,:,:) = merged_sample{k};
                end
                if new_sample_id > 0
                    samplesf{k}(new_sample_id,:,:,:) = new_sample{k};
                end
            end
        end
    end

    sample_weights = cast(prior_weights, 'like', params.data_type);
           
    train_tracker = (seq.frame < params.skip_after_frame) || (frames_since_last_train >= params.train_gap);
    
    if train_tracker     
        % Used for preconditioning
        %用于预处理
        new_sample_energy = cellfun(@(xlf) abs(xlf .* conj(xlf)), xlf_proj, 'uniformoutput', false);
        
        if seq.frame == 1
            % Initialize stuff for the filter learning
            %为过滤器学习初始化东西
            
            % Initialize Conjugate Gradient parameters
            %初始化共轭梯度参数
            sample_energy = new_sample_energy;
            CG_state = [];
            
            if params.update_projection_matrix
                % Number of CG iterations per GN iteration 
                %每GN迭代的CG迭代数
                init_CG_opts.maxit = ceil(params.init_CG_iter / params.init_GN_iter);
            
                hf = cell(2,1,num_feature_blocks);
                proj_energy = cellfun(@(P, yf) 2*sum(abs(yf(:)).^2) / sum(feature_dim) * ones(size(P), 'like', params.data_type), projection_matrix, yf, 'uniformoutput', false);
            else
                CG_opts.maxit = params.init_CG_iter; % Number of initial iterations if projection matrix is not updated
                                                                               %未更新投影矩阵时的初始迭代次数
                hf = cell(1,1,num_feature_blocks);
            end
            
            % Initialize the filter with zeros
            for k = 1:num_feature_blocks
                hf{1,1,k} = zeros([filter_sz(k,1) (filter_sz(k,2)+1)/2 sample_dim(k)], 'like', params.data_type_complex);
            end
        else
            CG_opts.maxit = params.CG_iter;
            
            % Update the approximate average sample energy using the learning
            %利用学习更新近似平均样本能量
            % rate. This is only used to construct the preconditioner.
            %率。这只用于构造前置条件。
            sample_energy = cellfun(@(se, nse) (1 - params.learning_rate) * se + params.learning_rate * nse, sample_energy, new_sample_energy, 'uniformoutput', false);
        end
        
        % Do training
        if seq.frame == 1 && params.update_projection_matrix
            if params.debug
                projection_matrix_init = projection_matrix;
            end
            
            % Initial Gauss-Newton optimization of the filter and
            %初始的高斯-牛顿优化滤波器和
            % projection matrix.
            %投影矩阵。
            if params.use_gpu
                [hf, projection_matrix, res_norms] = train_joint_gpu(hf, projection_matrix, xlf, yf, reg_filter, sample_energy, reg_energy, proj_energy, params, init_CG_opts);
            else
                [hf, projection_matrix, res_norms] = train_joint(hf, projection_matrix, xlf, yf, reg_filter, sample_energy, reg_energy, proj_energy, params, init_CG_opts);
            end
            
            % Re-project and insert training sample 重新规划和插入训练样本
            xlf_proj = project_sample(xlf, projection_matrix);
            for k = 1:num_feature_blocks
                if params.use_gpu
                    samplesf{k}(:,:,:,1) = xlf_proj{k};
                else
                    samplesf{k}(1,:,:,:) = permute(xlf_proj{k}, [4 3 1 2]);
                end
            end
            
            % Update the gram matrix since the sample has changed 更新克矩阵，因为样品已经改变
            if strcmp(params.distance_matrix_update_type, 'exact')
                % Find the norm of the reprojected sample 找到重投影样本的规范
                new_train_sample_norm =  0;
                
                for k = 1:num_feature_blocks
                    new_train_sample_norm = new_train_sample_norm + real(gather(2*(xlf_proj{k}(:)' * xlf_proj{k}(:))));% - reshape(xlf_proj{k}(:,end,:,:), [], 1, 1)' * reshape(xlf_proj{k}(:,end,:,:), [], 1, 1));
                end
                
                gram_matrix(1,1) = new_train_sample_norm;
            end
            
            if params.debug
                norm_proj_mat_init = sqrt(sum(cellfun(@(P) gather(norm(P(:))^2), projection_matrix_init)));
                norm_proj_mat = sqrt(sum(cellfun(@(P) gather(norm(P(:))^2), projection_matrix)));
                norm_proj_mat_change = sqrt(sum(cellfun(@(P,P2) gather(norm(P(:) - P2(:))^2), projection_matrix_init, projection_matrix)));
                fprintf('Norm init: %f, Norm final: %f, Matrix change: %f\n', norm_proj_mat_init, norm_proj_mat, norm_proj_mat_change / norm_proj_mat_init);
            end
        else
            % Do Conjugate gradient optimization of the filter 滤波器的共轭梯度优化
            if params.use_gpu
                [hf, res_norms, CG_state] = train_filter_gpu(hf, samplesf, yf, reg_filter, sample_weights, sample_energy, reg_energy, params, CG_opts, CG_state);
            else
                [hf, res_norms, CG_state] = train_filter(hf, samplesf, yf, reg_filter, sample_weights, sample_energy, reg_energy, params, CG_opts, CG_state);
            end
        end
        
        % Reconstruct the full Fourier series 重建整个傅里叶级数
        hf_full = full_fourier_coeff(hf);
        
        frames_since_last_train = 0;
    else
        frames_since_last_train = frames_since_last_train+1;
    end
    
    % Update the scale filter 更新缩放过滤器
    if nScales > 0 && params.use_scale_filter
        scale_filter = scale_filter_update(im, pos, base_target_sz, currentScaleFactor, scale_filter, params);
    end
    
    % Update the target size (only used for computing output box) 更新目标大小(仅用于计算输出框)
    target_sz = base_target_sz * currentScaleFactor;
    
    %save position and calculate FPS 保存位置并计算FPS
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);
    
    seq.time = seq.time + toc();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Visualization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % debug visualization
    if params.debug
        figure(20)
%         set(gcf,'units','normalized','outerposition',[0 0 1 1]);
        subplot_cols = num_feature_blocks;
        subplot_rows = 3;%ceil(feature_dim/subplot_cols);
        for disp_layer = 1:num_feature_blocks;
            subplot(subplot_rows,subplot_cols,disp_layer);
            imagesc(mean(abs(sample_fs(conj(hf_full{disp_layer}))), 3)); 
            colorbar;
            axis image;
            subplot(subplot_rows,subplot_cols,disp_layer+subplot_cols);
            imagesc(mean(abs(xl{disp_layer}), 3)); 
            colorbar;
            axis image;
            if seq.frame > 1
                subplot(subplot_rows,subplot_cols,disp_layer+2*subplot_cols);
                imagesc(fftshift(sample_fs(scores_fs_feat{disp_layer}(:,:,1,scale_ind))));
                colorbar;
                axis image;
            end
        end
        
        if train_tracker
            residuals_pcg = [residuals_pcg; res_norms];
            res_start_ind = max(1, length(residuals_pcg)-300);
            figure(99);plot(res_start_ind:length(residuals_pcg), residuals_pcg(res_start_ind:end));
            axis([res_start_ind, length(residuals_pcg), 0, min(max(residuals_pcg(res_start_ind:end)), 0.2)]);
        end
    end
    
    % visualization
    if params.visualization
        rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]); %repmat - 重复数组副本
        end
        if seq.frame == 1,  %first frame, create GUI
            fig_handle = figure('Name', 'Tracking');
%             set(fig_handle, 'Position', [100, 100, size(im,2), size(im,1)]);
            imagesc(im_to_show); %imagesc - 显示使用经过标度映射的颜色的图像
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(seq.frame), 'color', [0 1 1]);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
            
%             output_name = 'Video_name';
%             opengl software;
%             writer = VideoWriter(output_name, 'MPEG-4');
%             writer.FrameRate = 5;
%             open(writer);
        else
            % Do visualization of the sampled confidence scores overlayed
            %是否将抽样的置信度分数进行可视化
            resp_sz = round(img_support_sz*currentScaleFactor*scaleFactors(scale_ind));
            xs = floor(det_sample_pos(2)) + (1:resp_sz(2)) - floor(resp_sz(2)/2);
            ys = floor(det_sample_pos(1)) + (1:resp_sz(1)) - floor(resp_sz(1)/2);
            
            % To visualize the continuous scores, sample them 10 times more
            % dense than output_sz. 
            sampled_scores_display = fftshift(sample_fs(scores_fs(:,:,scale_ind), 10*output_sz));
            %fftshift - 将零频分量移到频谱中心
            figure(fig_handle);
%             set(fig_handle, 'Position', [100, 100, 100+size(im,2), 100+size(im,1)]);
            imagesc(im_to_show);
            hold on;
            resp_handle = imagesc(xs, ys, sampled_scores_display); colormap hsv; %弄上hsv的颜色空间
                                                                                                                     %imagesc - 显示使用经过标度映射的颜色的图像
            alpha(resp_handle, 0.3); %alpha - 向坐标区中的对象添加透明度
            rectangle('Position',rect_position_vis, 'EdgeColor','r', 'LineWidth',2);
            text(10, 10, int2str(seq.frame), 'color', [0 1 1]);
            hold off;
            
%                 axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        end
        
        drawnow
%         if frame > 1
%             if frame < inf
%                 writeVideo(writer, getframe(gcf));
%             else
%                 close(writer);
%             end
%         end
%          pause
    end
end

% close(writer);

[seq, results] = get_sequence_results(seq);

disp(['fps: ' num2str(results.fps)])


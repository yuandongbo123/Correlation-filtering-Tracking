function results = testing_ECO_HC(seq,video_path, res_path, bSaveImage, parameters)

% Feature specific parameters
hog_params.cell_size = 6;
hog_params.compressed_dim = 10;

grayscale_params.colorspace='gray';
grayscale_params.cell_size = 1;

cn_params.tablename = 'CNnorm';
cn_params.useForGray = false;
cn_params.cell_size = 4;
cn_params.compressed_dim = 3;

ic_params.tablename = 'intensityChannelNorm6';
ic_params.useForColor = false;
ic_params.cell_size = 4;
ic_params.compressed_dim = 3;

% Which features to include
params.t_features = {
    struct('getFeature',@get_fhog,'fparams',hog_params),...
    ...struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...
    struct('getFeature',@get_table_feature, 'fparams',cn_params),...
    struct('getFeature',@get_table_feature, 'fparams',ic_params),...
};

% Global feature parameters1s
params.t_global.normalize_power = 2;    % Lp normalization with this %p Lp和p的归一化
params.t_global.normalize_size = true;  % Also normalize with respect to the spatial size of the feature %也标准化与空间大小的特征
params.t_global.normalize_dim = true;   % Also normalize with respect to the dimensionality of the feature %也对特征的维数进行归一化

% Image sample parameters   % 图像样本参数
params.search_area_shape = 'square';    % The shape of the samples %样品的形状
params.search_area_scale = 4.0;         % The scaling of the target size to get the search area %缩放目标大小得到搜索区域
params.min_image_sample_size = 150^2;   % Minimum area of image samples %图像样本的最小面积
params.max_image_sample_size = 200^2;   % Maximum area of image samples %图像样本的最大面积

% Detection parameters
params.refinement_iterations = 1;       % Number of iterations used to refine the resulting position in a frame %用于在一帧中细化结果位置的迭代次数
params.newton_iterations = 5;           % The number of Newton iterations used for optimizing the detection score % 用于优化检测分数的牛顿迭代次数
params.clamp_position = false;          % Clamp the target position to be inside the image %夹紧目标位置，使其位于图像内部

% Learning parameters
params.output_sigma_factor = 1/16;		% Label function sigma %标签功能σ
params.learning_rate = 0.009;	 	 	% Learning rate %学习速率
params.nSamples = 30;                   % Maximum number of stored training samples %存储训练样本的最大数目
params.sample_replace_strategy = 'lowest_prior';    % Which sample to replace when the memory is full %当内存满时，更换哪个样品
params.lt_size = 0;                     % The size of the long-term memory (where all samples have equal weight)  %长期记忆的大小(所有样本的权重相等)
params.train_gap = 5;                   % The number of intermediate frames with no training (0 corresponds to training every frame) %没有训练的中间帧数(0表示每帧训练)
params.skip_after_frame = 10;           % After which frame number the sparse update scheme should start (1 is directly) %在哪个帧编号之后稀疏更新方案应该开始(1直接)
params.use_detection_sample = true;     % Use the sample that was extracted at the detection stage also for learning %使用在检测阶段提取的样本也用于学习

% Factorized convolution parameters %映像卷积参数
params.use_projection_matrix = true;    % Use projection matrix, i.e. use the factorized convolution formulation %使用投影矩阵，即使用分解卷积公式
params.update_projection_matrix = true; % Whether the projection matrix should be optimized or not %是否需要优化投影矩阵
params.proj_init_method = 'pca';        % Method for initializing the projection matrix %初始化投影矩阵的方法
params.projection_reg = 1e-7;	 	 	% Regularization paremeter of the projection matrix %投影矩阵的正则化参数

% Generative sample space model parameters %生成样本空间模型参数
params.use_sample_merge = true;                 % Use the generative sample space model to merge samples %使用生成样本空间模型来合并样本
params.sample_merge_type = 'Merge';             % Strategy for updating the samples %更新样本的策略
params.distance_matrix_update_type = 'exact';   % Strategy for updating the distance matrix %距离矩阵的更新策略

% Conjugate Gradient parameters %共轭梯度参数
params.CG_iter = 5;                     % The number of Conjugate Gradient iterations in each update after the first frame %在第一帧后的每次更新中共轭梯度迭代的次数
params.init_CG_iter = 10*15;            % The total number of Conjugate Gradient iterations used in the first frame %第一帧使用的共轭梯度迭代的总数
params.init_GN_iter = 10;               % The number of Gauss-Newton iterations used in the first frame (only if the projection matrix is updated) %第一帧使用的高斯-牛顿迭代次数(仅当更新投影矩阵时)
params.CG_use_FR = false;               % Use the Fletcher-Reeves (true) or Polak-Ribiere (false) formula in the Conjugate Gradient %在共轭梯度中使用Fletcher-Reeves(真)或Polak-Ribiere(假)公式
params.CG_standard_alpha = true;        % Use the standard formula for computing the step length in Conjugate Gradient
params.CG_forgetting_rate = 50;	 	 	% Forgetting rate of the last conjugate direction %用共轭梯度法的标准公式计算步长
params.precond_data_param = 0.75;       % Weight of the data term in the preconditioner %前置条件中数据项的权重
params.precond_reg_param = 0.25;	 	% Weight of the regularization term in the preconditioner %正则化项在预处理器中的权重
params.precond_proj_param = 40;	 	 	% Weight of the projection matrix part in the preconditioner %预调节器中投影矩阵部分的权值

% Regularization window parameters %正则化参数窗口
params.use_reg_window = true;           % Use spatial regularization or not %是否使用空间正则化
params.reg_window_min = 1e-4;			% The minimum value of the regularization window %正则化窗口的最小值
params.reg_window_edge = 10e-3;         % The impact of the spatial regularization %空间规格化的影响
params.reg_window_power = 2;            % The degree of the polynomial to use (e.g. 2 is a quadratic window) %要使用的多项式的次数(例如2是一个二次窗口)
params.reg_sparsity_threshold = 0.05;   % A relative threshold of which DFT coefficients that should be set to zero %一种相对阈值，其DFT系数应设为零

% Interpolation parameters %插值参数
params.interpolation_method = 'bicubic';    % The kind of interpolation kernel %一类插值核
params.interpolation_bicubic_a = -0.75;     % The parameter for the bicubic interpolation kernel %双三次插值核的参数
params.interpolation_centering = true;      % Center the kernel at the feature sample %将内核置于特性样本的中心
params.interpolation_windowing = false;     % Do additional windowing on the Fourier coefficients of the kernel %对核函数的傅里叶系数做额外的窗口操作吗

% Scale parameters for the translation model %转换模型的比例参数
% Only used if: params.use_scale_filter = false %仅在:params时使用。use_scale_filter = false
params.number_of_scales = 7;            % Number of scales to run the detector %运行检测器的刻度数
params.scale_step = 1.01;               % The scale factor %规模因素

% Scale filter parameters %尺度滤波器参数
% Only used if: params.use_scale_filter = true
params.use_scale_filter = true;         % Use the fDSST scale filter or not (for speed) %是否使用fDSST规模过滤器(为了速度)
params.scale_sigma_factor = 1/16;       % Scale label function sigma %标度函数
params.scale_learning_rate = 0.025;		% Scale filter learning rate %尺度滤波学习率
params.number_of_scales_filter = 17;    % Number of scales %数量的尺度
params.number_of_interp_scales = 33;    % Number of interpolated scales %插值尺度数
params.scale_model_factor = 1.0;        % Scaling of the scale model %缩放模型
params.scale_step_filter = 1.02;        % The scale factor for the scale filter %比例滤波器的比例因子
params.scale_model_max_area = 32*16;    % Maximume area for the scale sample patch %尺度样块的最大面积
params.scale_feature = 'HOG4';          % Features for the scale filter (only HOG4 supported) %缩放过滤器的功能(仅支持HOG4)
params.s_num_compressed_dim = 'MAX';    % Number of compressed feature dimensions in the scale filter %尺度滤波器中压缩特征维数
params.lambda = 1e-2;					% Scale filter regularization 尺度滤波器正规化
params.do_poly_interp = true;           % Do 2nd order polynomial interpolation to obtain more accurate scale %做二次多项式插值以获得更精确的比例

% Visualization
params.visualization = 1;               % Visualiza tracking and detection scores %可视化跟踪和检测分数
params.debug = 0;                       % Do full debug visualization  %执行完整的调试可视化

% GPU
params.use_gpu = false;                 % Enable GPU or not
params.gpu_id = [];                     % Set the GPU id, or leave empty to use default

% Initialize
params.seq = seq;
params.video_path = video_path;
% Run tracker
results = tracker(params);

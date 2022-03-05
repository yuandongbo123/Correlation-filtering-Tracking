%  This function takes care of setting up parameters,loading video information and computing precisions.
%  该功能负责设置参数，加载视频信息和计算精度。
%   For the actual tracking code,check out the TRACKER function.
%   对于实际的跟踪代码，请查看TRACKER功能。
%  RUN_TRACKER  %    Without any parameters, will ask you to choose a video, track using
%    the Gaussian KCF on HOG, and show the results in an interactive figure.
%    没有任何参数，会要求您选择一个视频，跟踪使用HOG上的高斯KCF，并在交互式图中显示结果。
%    Press 'Esc' to stop the tracker early. You can navigate the video using the scrollbar at the bottom.
%    您可以使用底部的滚动条浏览视频。

%  RUN_TRACKER VIDEO %    Allows you to select a VIDEO by its name. 'all' will run all videos
%  and show average statistics. 'choose' will select one interactively.选择一个视频。

%  RUN_TRACKER VIDEO KERNEL %    Choose a KERNEL. 'gaussian'/'polynomial' to run KCF, 'linear' for DCF.
%    选择一个核，高斯核或者多项式核去运行KCF。

%  RUN_TRACKER VIDEO KERNEL FEATURE  %    Choose a FEATURE type, either 'hog' or 'gray' (raw pixels).
%    选择使用特征的类型，hog还是gray

%  RUN_TRACKER(VIDEO, KERNEL, FEATURE, SHOW_VISUALIZATION, SHOW_PLOTS)
% Decide whether to show the scrollable figure, and the precision plot.决定是否显示可滚动图形，以及精确图。
%  Useful combinations:%  >> run_tracker choose gaussian hog  %Kernelized Correlation Filter (KCF)
%  >> run_tracker choose linear hog    %Dual Correlation Filter (DCF)
%  >> run_tracker choose gaussian gray %Single-channel KCF (ECCV'12 paper)
%  >> run_tracker choose linear gray   %MOSSE filter (single channel)

function [precision, fps] = run_tracker(video, kernel_type, feature_type, show_visualization, show_plots)
%function [输出变量] = 函数名称(输入变量）
%递归调用
%path to the videos (you'll be able to choose one with the GUI).
switch(1)
    case 1
        base_path = '.\data\Benchmark\';
    case 2
        base_path = 'C:\Users\Administrator\Documents\MATLAB\KCF-FOR-MATLAB-master\data\OTB100\';
end

%default settings  默认设置
if nargin < 1, video = 'choose'; end %all %如果输入变量的个数小于1个，那么给video赋默认值，只执行这一条语句
if nargin < 2, kernel_type = 'gaussian'; end%gaussian nargin的具体定义
if nargin < 3, feature_type = 'hog'; end%hog
if nargin < 4, show_visualization = ~strcmp(video, 'all'); end% strcmp对比字符串 相同返回1 否则返回0  ~strcmp则是取反
if nargin < 5, show_plots = ~strcmp(video, 'all'); end%？？？？？？？？？？？？？？？？？
%parameters according to the paper. at this point we can override parameters based on the chosen kernel or feature type
%参数根据文件。在这一点上，我们可以根据选择的内核或功能类型覆盖参数
kernel.type = kernel_type;

features.gray = false;
features.hog = false;

padding = 1.5;  % 跟踪框的填充倍数
lambda = 1e-4;  %regularization 正规化 模型更新参数
output_sigma_factor = 0.1;  %spatial bandwidth (proportional to target)
%空间带宽（与目标成正比）
switch(feature_type)
    case 'gray'
        interp_factor = 0.075;  %linear interpolation factor for adaptation
        %用于适应的线性内插因子
        kernel.sigma = 0.2;     %gaussian kernel bandwidth
        %高斯内核带宽
        kernel.poly_a = 1;      %polynomial kernel additive term
        %多项式核加法项
        kernel.poly_b = 7;      %polynomial kernel exponent
        %多项式内核指数
        features.gray = true;
        cell_size = 1;
        
    case 'hog'
        interp_factor = 0.02;  %interp_factor是跟踪器的自适应速率，即更新模型的记忆因子
        
        kernel.sigma = 0.5; %高斯内核带宽
        
        kernel.poly_a = 1; %多项式核加法项
        kernel.poly_b = 9; %多项式内核指数
        
        features.hog = true;
        features.hog_orientations = 9;
        cell_size = 4;
        
    otherwise
        error('Unknown feature.')
end

assert(any(strcmp(kernel_type, {'linear', 'polynomial', 'gaussian'})), 'Unknown kernel.')
%在matlab中assert函数用来判断一个expression是否成立
%strcmp是用于做字符串比较的函数

switch video
    case 'choose'
        video = choose_video(base_path);%videoname
        if ~isempty(video)%isempty(msg)判断msg是否为空，如果为空，结果为1，否则为0.
            [precision, fps] = run_tracker(video, kernel_type, feature_type, show_visualization, show_plots);%调用本身，递归调用。
            if nargout == 0  %don't output precision as an argument
                clear precision
            end
        end
        fprintf(' FPS:% 4.2f\n', fps)
        
    case 'all'
        %all videos, call self with each video name.
        dirs = dir(base_path);
        videos = {dirs.name};
        videos(strcmp('.', videos) | strcmp('..', videos) | strcmp('anno', videos) | ~[dirs.isdir]) = [];
        %the 'Jogging' sequence has 2 targets, create one entry for each.
        %“慢跑”序列有2个目标，为每个目标创建一个条目。
        %we could make this more general if multiple targets per video becomes a common occurence.
        %如果每个视频的多个目标成为常见的情况，我们可以使其更为通用。
        videos(strcmpi('Jogging', videos)) = [];
        videos(end+1:end+2) = {'Jogging.1', 'Jogging.2'};
        all_precisions = zeros(numel(videos),1);  %to compute averages
        all_fps = zeros(numel(videos),1);
        
        if ~ exist('parpool', 'file')
            %no parallel toolbox, use a simple 'for' to iterate
            %没有平行的工具箱，使用一个简单的'for'来迭代
            for k = 1:numel(videos)
                [all_precisions(k), all_fps(k)] = run_tracker(videos{k}, ...
                    kernel_type, feature_type, show_visualization, show_plots);
            end
            
            if nargout == 0%don't output precision as an argument
                clear precision
            end
        else
            %evaluate trackers for all videos in parallel
            %并行评估所有视频的跟踪器
            %	if parpool('local') == 0,
            %parpool open;
            %	end
            for k = 1:numel(videos)
                [all_precisions(k), all_fps(k)] = run_tracker(videos{k}, kernel_type, feature_type, show_visualization, show_plots);
            end
        end
        %compute average precision at 20px, and FPS
        %计算20px的平均精度和FPS     px是像素
        mean_precision = mean(all_precisions);
        fps = mean(all_fps);
        fprintf('\nAverage precision (20px):% 1.3f, Average FPS:% 4.2f\n\n', mean_precision, fps)
        if nargout > 0
            precision = mean_precision;
        end
        
    case 'benchmark'
        %running in benchmark mode - this is meant to interface easily with the benchmark's code.
        %以基准模式运行 - 这意味着可以轻松地与基准代码进行连接。
        %get information (image file names, initial position, etc) fromthe benchmark's workspace variables
        %从基准的工作空间变量获取信息（图像文件名，初始位置等）
        seq = evalin('base', 'subS');%base是MATLAB的基本工作空间。将subS的值赋给seq。
        target_sz = seq.init_rect(1,[4,3]);
        pos = seq.init_rect(1,[2,1]) + floor(target_sz/2);
        %初始定位目标的位置
        img_files = seq.s_frames;
        video_path = [];
        %call tracker function with all the relevant parameters
        %具有所有相关参数的呼叫跟踪函数
        positions = tracker(video_path, img_files, pos, target_sz, ...
            padding, kernel, lambda, output_sigma_factor, interp_factor, ...
            cell_size, features, false);
        %return results to benchmark, in a workspace variable
        %将结果返回到基准，在工作空间变量中
        rects = [positions(:,2) - target_sz(2)/2, positions(:,1) - target_sz(1)/2];
        rects(:,3) = target_sz(2);
        rects(:,4) = target_sz(1);
        res.type = 'rect';
        res.res = rects;
        assignin('base', 'res', res);
        
    otherwise
        %we were given the name of a single video to process.
        %我们获得了一个要处理的视频的名称。
        %get image file names, initial state, and ground truth for evaluation
        %获取图像文件名，初始状态和地面实况进行评估
        [img_files, pos, target_sz, ground_truth, video_path] = load_video_info(base_path, video);%路径和视频名字
        %img_files：cell类型，存储所有的图片的名字 pos: =[cy,cx],target:[h,w]
        %groundtruth存储的
        %ground_truth 存储的是
        %call tracker function with all the relevant parameters
        [positions, time] = tracker(video_path, img_files, pos, target_sz, ...
            padding, kernel, lambda, output_sigma_factor, interp_factor, ...
            cell_size, features, show_visualization);
        %calculate and show precision plot, as well as frames-per-second
        %计算和显示精确度图，以及每秒帧数
        %precisions = precision_plot(positions, ground_truth, video, show_plots);
        fps = numel(img_files) / time;
        %fprintf('%12s - Precision (20px):% 1.3f, FPS:% 4.2f\n', video, precisions(20), fps)
        
        if nargout > 0
            %precision = precisions(20);
        end
        precision=0;
 end
end


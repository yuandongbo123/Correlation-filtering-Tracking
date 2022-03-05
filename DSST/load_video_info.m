function [img_files, pos, target_sz, ground_truth, video_path] = load_video_info(base_path, video)
%LOAD_VIDEO_INFO
%   Loads all the relevant information for the video in the given path:
%   在给定路径中加载视频的所有相关信息：
%   the list of image files (cell array of strings), initial position
%   (1x2), target size (1x2), the ground truth information for precision
%   calculations (Nx2, for N frames), and the path where the images are
%   located. The ordering of coordinates and sizes is always [y, x].
%   图像文件列表，初始位置，目标尺寸，精确计算的地面真相信息坐标和大小的顺序总是[y，x]。
%ground_truth是下载视频集的时候就有的，里面存的是每一帧目标框的 左上角坐标 和 宽、高。
%这些数据是事先人工标定好了的。只是我们只用ground_truth里第一帧的数据，其它的评估这个算法精确度时候用。

	%see if there's a suffix, specifying one of multiple targets, for
	%example the dot and number in 'Jogging.1' or 'Jogging.2'.
    %看看是否有一个后缀，指定多个目标之一例如“慢跑1”或“慢跑2”中的点数。

	if numel(video) >= 2 && video(end-1) == '.' && ~isnan(str2double(video(end))),%str2double字符串转换数值
        %tf=isnan(A)：返回一个与A相同维数的数组，若A的元素为非数值，在对应位置上返回逻辑1，否则返回逻辑0（假）。
		suffix = video(end-1:end);  %remember the suffix 
		video = video(1:end-2);  %remove it from the video name 
	else
		suffix = '';
	end

	%full path to the video's files视频文件的完整路径
	if base_path(end) ~= '/' && base_path(end) ~= '\',
		base_path(end+1) = '/';
	end
	video_path = [base_path video '/'];%

	%try to load ground truth from text file (Benchmark's format)
	filename = [video_path 'groundtruth_rect' suffix '.txt'];
	f = fopen(filename);%打开文件
	assert(f ~= -1, ['No initial position or ground truth to load ("' filename '").'])
	
	%the format is [x, y, width, height] 格式为[x，y，width，height]
	try
		ground_truth = textscan(f, '%f,%f,%f,%f', 'ReturnOnError',false);  %**********
	catch  %# try different format (no commas)
		frewind(f);%%位置指针移至文件首部
		ground_truth = textscan(f, '%f %f %f %f');%读取打数据
    end
	ground_truth = cat(2, ground_truth{:});%cat是横向连接，如[A, B];
	fclose(f);%关闭打开的文件
	
	%set initial position and size设置初始位置和大小 target_size=[h,w]
	target_sz = [ground_truth(1,4), ground_truth(1,3)];%就是把跟踪框放在要跟踪的目标上
	pos = [ground_truth(1,2), ground_truth(1,1)] + floor(target_sz/2);%目标的初始位置，即目标框中心坐标
	%pos=[y,x]
	if size(ground_truth,1) == 1
		%we have ground truth for the first frame only (initial position)
        %我们只有第一帧的（初始位置）
		ground_truth = [];
	else
		%store positions instead of boxes
        %这里我进行了注释，为了让得到的结果效果更好
		%ground_truth = ground_truth(:,[2,1]) + ground_truth(:,[4,3]) / 2;
	end
	
	%from now on, work in the subfolder where all the images are
    %从现在起，在所有图像的子文件夹中工作
	video_path = [video_path 'img/'];
	
	%for these sequences, we must limit ourselves to a range of frames.
    %对于这些序列，我们必须将自己限制在一定范围内。
	%for all others, we just load all png/jpg files in the folder.
    %对于其他文件，我们只需加载文件夹中的所有png / jpg文件。
	frames = {'David', 300, 770;
			  'Football1', 1, 74;
			  'Freeman3', 1, 460;
			  'Freeman4', 1, 283};
	
	idx = find(strcmpi(video, frames(:,1)));
	
	if isempty(idx)
		%general case, just list all images一般情况下，只列出所有图像
		img_files = dir([video_path '*.png']);
		if isempty(img_files)
			img_files = dir([video_path '*.jpg']);
			assert(~isempty(img_files), 'No image files to load.')
		end
		img_files = sort({img_files.name});
	else
		%list specified frames. try png first, then jpg.列出指定的帧。 尝试png，然后jpg。
		if exist(sprintf('%s%04i.png', video_path, frames{idx,2}), 'file'),
			img_files = num2str((frames{idx,2} : frames{idx,3})', '%04i.png');
			
		elseif exist(sprintf('%s%04i.jpg', video_path, frames{idx,2}), 'file'),
			img_files = num2str((frames{idx,2} : frames{idx,3})', '%04i.jpg');
			
		else
			error('No image files to load.')
		end
		
		img_files = cellstr(img_files);
	end
	
end


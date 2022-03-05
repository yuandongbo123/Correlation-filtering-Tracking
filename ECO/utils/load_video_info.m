function [seq, ground_truth,video_path] = load_video_info(base_path,video)

if numel(video) >= 2 && video(end-1) == '.' && ~isnan(str2double(video(end))),
    suffix = video(end-1:end);  %remember the suffix                   记住后缀
    video = video(1:end-2);  %remove it from the video name            从视频名称中删除它
else
    suffix = '';
end

if base_path(end) ~= '/' && base_path(end) ~= '\',
    base_path(end+1) = '/';
end
video_path = [base_path video '/'];

ground_truth = dlmread([video_path 'groundtruth_rect' suffix '.txt']);
filename = [video_path 'groundtruth_rect' suffix '.txt']
f = fopen(filename);                                                       %fopen - 打开文件或获得有关打开文件的信息 返回 fileID 
assert(f ~= -1, ['No initial position or ground truth to load ("' filename '").'])
% 
% ground_truth = dlmread([video_path '/groundtruth_rect.txt']);

seq.format = 'otb';
seq.len = size(ground_truth, 1);
seq.init_rect = ground_truth(1,:);

video_path = [video_path 'img/'];

frames = {'David', 300, 770;
          'Football1', 1, 74;
          'Freeman3', 1, 460;
          'Freeman4', 1, 283};
% img_path = [video_path '/img/'];

% if exist([img_path num2str(1, '%08i.png')], 'file'),
%     img_files = num2str((0:seq.len-1)', [img_path '%08i.png']);
% elseif exist([img_path num2str(1, '%0i.jpg')], 'file'),
%     img_files = num2str((0:seq.len-1)', [img_path '%04i.jpg']);
% elseif exist([img_path num2str(1, '%04i.bmp')], 'file'),
%     img_files = num2str((1:seq.len)', [img_path '%04i.bmp']);
% else
%     error('No image files to load.')
% end
% 
% seq.s_frames = cellstr(img_files);
idx = find(strcmpi(video, frames(:,1)));  %find 查找每个非0元素的数组的索引
                                                                       %strcmpi查找元胞数组中不区分大小写的匹配项，并返回0，1数组
	
if isempty(idx),
    %general case, just list all images                                一般情况下，只需列出所有图像
    img_files = dir([video_path '*.png']);
    if isempty(img_files),
        img_files = dir([video_path '*.jpg']);
%         assert(~isempty(img_files), 'No image files to load.')
    end
    img_files = sort({img_files.name});
else
    %list specified frames. try png first, then jpg.                   列出指定的帧。 先尝试PNG，然后JPG
    if exist(sprintf('%s%04i.png', video_path, frames{idx,2}), 'file'),
        img_files = num2str((frames{idx,2} : frames{idx,3})',[img_path '%04i.png']');

    elseif exist(sprintf('%s%04i.jpg', video_path, frames{idx,2}), 'file'),
        img_files = num2str((frames{idx,2} : frames{idx,3})', [img_path '%04i.jpg']');

    else
        error('No image files to load.')
    end
 end
seq.s_frames = cellstr(img_files);    %cellstr - 转换为字符向量元胞数组
	
end


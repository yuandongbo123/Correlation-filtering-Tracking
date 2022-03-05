function [img_files, pos, target_sz, ground_truth, video_path] = load_video_info(video_path)

% [img_files, pos, target_sz, ground_truth, video_path] = load_video_info(video_path)

% text_files = dir([video_path '*_frames.txt']);
% f = fopen([video_path text_files(1).name]);
% frames = textscan(f, '%f,%f');
% fclose(f);

text_files = dir([video_path '*groundtruth_rect.txt']);
assert(~isempty(text_files), 'No initial position and ground truth (*_gt.txt) to load.')

f = fopen([video_path text_files(1).name]);
ground_truth = textscan(f, '%f,%f,%f,%f');  %[x, y, width, height]
ground_truth = cat(2, ground_truth{:});
fclose(f);

%set initial position and size
target_sz = [ground_truth(1,3), ground_truth(1,4)];
pos = [ground_truth(1,1), ground_truth(1,2)];

%see if they are in the 'imgs' subfolder or not
% if exist([video_path num2str(frames{1}, 'imgs/img%05i.png')], 'file'),
%     video_path = [video_path 'imgs/'];
%     img_files = num2str((frames{1} : frames{2})', [video_path 'img%05i.png']);
% elseif exist([video_path num2str(frames{1}, 'imgs/img%05i.jpg')], 'file'),
%     video_path = [video_path 'imgs/'];
%     img_files = num2str((frames{1} : frames{2})', [video_path 'img%05i.jpg']);
% elseif exist([video_path num2str(frames{1}, 'imgs/img%05i.bmp')], 'file'),
%     video_path = [video_path 'imgs/'];
%     img_files = num2str((frames{1} : frames{2})', [video_path 'img%05i.bmp']);
% else
%     error('No image files to load.')
% end
frames = {'David', 300, 770;};
      
idx = find(strcmpi(video_path, ['E:/resarch_dataset/Benchmark/' frames(1)]));
if isempty(idx),
    %general case, just list all images                                一般情况下，只需列出所有图像
    img_files = dir([video_path 'img/' '*.png']);
    if isempty(img_files),
        img_files = dir([video_path 'img/ ''*.jpg']);
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
%list the files
img_files = cellstr(img_files);

end


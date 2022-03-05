function [seq, ground_truth] = load_video_info(video_path)

ground_truth = dlmread([video_path '/groundtruth_rect.txt']);
%dlmread - （不推荐）将 ASCII 分隔的数值数据文件读取到矩阵
seq.len = size(ground_truth, 1);
seq.init_rect = ground_truth(1,:);

frames = {'David', 300, 770;};
      
idx = find(strcmpi(video_path, ['F:/resarch_dataset/Benchmark/' frames(1)]));
if isempty(idx),
    %general case, just list all images                                一般情况下，只需列出所有图像
    a = [video_path '/img/'];
    img_files = dir([a '*.png']);
    if isempty(img_files),
        img_files = dir([a '*.jpg']);
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

seq.s_frames = cellstr(img_files);

end


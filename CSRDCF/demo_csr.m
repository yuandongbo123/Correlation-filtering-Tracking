function demo_csr()

% set this to tracker directory
tracker_path = 'F:\code\CSRDCF';
% add paths
addpath(tracker_path);
addpath(fullfile(tracker_path, 'mex'));
addpath(fullfile(tracker_path, 'utils'));
addpath(fullfile(tracker_path, 'features'));

visualize_tracker    = true;
use_reinitialization = true;

% choose name of the VOT sequence
% sequence_name = 'Football';    
% path to the folder with VOT sequences
base_path = 'F:\resarch_dataset\OTB50/';
sequence_name = choose_video(base_path);
base_path_img = fullfile(base_path, sequence_name,'\img');
base_path = fullfile(base_path, sequence_name);
img_dir = dir(fullfile(base_path_img, '/*.jpg'));

% initialize bounding box - [x,y,width, height]
 gt = read_vot_regions(fullfile(base_path, 'groundtruth_rect.txt'));%获取四个坐标和otb一样
gt8= dlmread(fullfile(base_path, 'groundtruth_rect.txt')); %获取八个点的坐标，这是带旋转的矩形框

start_frame = 1;
n_failures = 0;
time = zeros(numel(img_dir), 1);%记录处理每一帧用的时间

n_tracked = 0;

if visualize_tracker
    figure(1); clf;
end

frame = start_frame;
while frame <= numel(img_dir) % tracking loop
	%
    impath = fullfile(base_path_img,'/', img_dir(frame).name);
    img = imread(impath);%imshow(img);
    
    tic()
	% initialize or track
	if frame == start_frame
        bb = gt8(frame,:) + 1;  % add 1: ground-truth top-left corner is (0,0)
		tracker = create_csr_tracker(img, bb);
        bb = gt(frame,:);  % just to prevent error when plotting
    else
		[tracker, bb] = track_csr_tracker(tracker, img);%输出的是一个矩形框 
    end
    time(frame) = toc();
    
    n_tracked = n_tracked + 1;
    
    % visualization and failure detection
    if visualize_tracker
        
        figure(1); if(size(img,3)<3), colormap gray; end
        imagesc(uint8(img))
        hold on;
        rectangle('Position',bb,'LineWidth',1,'EdgeColor','b');

        text(15, 25, num2str(n_failures), 'Color','r', 'FontSize', ...
            15, 'FontWeight', 'bold');
        
        text(60, 25, [num2str(frame) '/' num2str(numel(img_dir))], 'Color','r', 'FontSize', ...
            15, 'FontWeight', 'bold');
        
        text(15, 45, ['fps:' num2str(1/time(frame))], 'Color','r', 'FontSize', ...
            15, 'FontWeight', 'bold');
        if use_reinitialization  % detect failures and reinit
            area = rectint(bb, gt(frame,:));
            if area < eps && use_reinitialization
                disp('Failure detected. Reinitializing tracker...');
                frame = frame + 4;  % skip 5 frames at reinit (like VOT)
                start_frame = frame + 1;
                n_failures = n_failures + 1;
            end
        end

        hold off;
        if frame == start_frame
            truesize;
        end
        drawnow; 
    end
    
    frame = frame + 1;

end

fps = n_tracked / sum(time);
fprintf('FPS: %.1f\n', fps);

end  % endfunction

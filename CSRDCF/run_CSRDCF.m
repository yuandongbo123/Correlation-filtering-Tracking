function results=run_CSRDCF(seq, res_path, bSaveImage)

% set this to tracker directory
% add paths
% addpath(fullfile(tracker_path, 'mex'));
% addpath(fullfile(tracker_path, 'utils'));
% addpath(fullfile(tracker_path, 'features'));
visualize_tracker = false;
use_reinitialization = true;
%gt8 = dlmread(fullfile(base_path, 'groundtruth_rect.txt'));
gt8=seq.init_rect;
gt=seq.init_rect;
start_frame = 1;
n_failures = 0;
video_path = seq.path;
img_dir=seq.s_frames;

time = zeros(numel(img_dir), 1);
n_tracked = 0;
frame = start_frame;
duration=0;
while frame <= numel(img_dir)  % tracking loop
	% read frame
     tic()
    img = imread(img_dir{frame});
    
	% initialize or track
	if frame == start_frame
        
        bb = gt8(frame,:);  % add 1: ground-truth top-left corner is (0,0)
		tracker = create_csr_tracker(img, bb);        
    else
        
		[tracker, bb] = track_csr_tracker(tracker, img);
    end
    gt(frame,:)=bb;  % j
    
    duration = duration+toc();
    
    n_tracked = n_tracked + 1;
    
    % visualization and failure detection
    frame = frame + 1;

end

fps = frame/duration;
fprintf('FPS: %.1f\n', fps);

results.type = 'rect';
results.res = gt;
results.fps = fps;
end  % endfunction

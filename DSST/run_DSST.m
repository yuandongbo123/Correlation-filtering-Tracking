function results = run_DSST(seq,res_path, bSaveImage)


% img_files = dir([seq.path '*.png']);
% if isempty(img_files)
%     img_files = dir([seq.path '*.jpg']);
%     assert(~isempty(img_files), 'No image files to load.')
% end
% img_files = sort({img_files.name});
% img_files = cellstr(img_files);

% params.img_files=img_files;

s_frames = seq.s_frames;
params.video_path = seq.path;
%parameters according to the paper
params.padding = 1.0;         			% extra area surrounding the target
params.output_sigma_factor = 1/16;		% standard deviation for the desired translation filter output
params.scale_sigma_factor = 1/4;        % standard deviation for the desired scale filter output
params.lambda = 1e-2;					% regularization weight (denoted "lambda" in the paper)
params.learning_rate = 0.025;			% tracking model learning rate (denoted "eta" in the paper)
params.number_of_scales = 33;           % number of scale levels (denoted "S" in the paper)
params.scale_step = 1.02;               % Scale increment factor (denoted "a" in the paper)
params.scale_model_max_area = 512;      % the maximum size of scale examples

params.visualization = 0;

params.wsize    = [seq.init_rect(1,4), seq.init_rect(1,3)]; % Ŀ��ԭʼ�ߴ�
params.init_pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(params.wsize/2);% Ŀ��ԭʼλ��
params.s_frames = s_frames;
% params.no_fram  = seq.en_frame - seq.st_frame + 1; % no_fram֡��
% params.seq_st_frame = seq.st_frame;  % ��ʼ֡
% params.seq_en_frame = seq.en_frame;  % ����֡
results = dsst(params);
fps = results.fps
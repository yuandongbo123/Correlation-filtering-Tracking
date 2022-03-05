
% This demo script runs the ECO tracker with hand-crafted features on the 该演示脚本运行带有手工制作功能的ECO跟踪器
% included "Crossing" video.

% Add paths
setup_paths();

% Load video information
% base_path = 'C:\Users\Administrator\Documents\MATLAB\ECO-master\sequences';
base_path = 'C:\Users\Administrator\Documents\MATLAB\KCF-FOR-MATLAB-master\data\Benchmark';
video = choose_video(base_path);
[seq, ground_truth,video_path] = load_video_info(base_path,video);

% Run ECO
results = testing_ECO_HC(seq,video_path);
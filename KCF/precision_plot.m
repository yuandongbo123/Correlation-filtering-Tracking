function precisions = precision_plot(positions, ground_truth, title, show)
%PRECISION_PLOT
%   Calculates precision for a series of distance thresholds (percentage of
%   frames where the distance to the ground truth is within the threshold).
%   计算一系列距离阈值（距离地面真值的距离在阈值内的帧的百分比）的精度。
%   The results are shown in a new figure if SHOW is true.
%   如果SHOW为真，结果将显示在新图中。
%   Accepts positions and ground truth as Nx2 matrices (for N frames), and a title string.
%   接受Nx2矩阵（N帧）和标题字符串的位置和地面真值。
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/
	max_threshold = 50;  %used for graphs in the paper 用于本文中的图形

	precisions = zeros(max_threshold, 1);
	
	if size(positions,1) ~= size(ground_truth,1),
% 		fprintf('%12s - Number of ground truth frames does not match number of tracked frames.\n', title)
		
        %just ignore any extra frames, in either results or ground truth
		n = min(size(positions,1), size(ground_truth,1));
		positions(n+1:end,:) = [];
		ground_truth(n+1:end,:) = [];
	end
	
	%calculate distances to ground truth over all frames计算所有帧的距离到地面真相
	distances = sqrt((positions(:,1) - ground_truth(:,1)).^2 + ...
				 	 (positions(:,2) - ground_truth(:,2)).^2);
	distances(isnan(distances)) = [];

	%compute precisions
	for p = 1:max_threshold,
		precisions(p) = nnz(distances <= p) / numel(distances);%=nnz(X)返回矩阵X中的非零元素的数目。
	end
	
	%plot the precisions
	if show == 1,
		figure( 'Name',['Precisions - ' title])
		plot(precisions, 'k-', 'LineWidth',2)
		xlabel('Threshold'), ylabel('Precision')
	end
	
end


function [positions, time] = tracker(video_path, img_files, pos, target_sz, ...
	padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, ...
	features, show_visualization)
%TRACKER Kernelized/Dual Correlation Filter (KCF/DCF) tracking.
%   This function implements the pipeline for tracking with the KCF (by choosing a non-linear kernel) and DCF (by choosing a linear kernel). 
%   �ù���ʵ����ʹ��KCF����;����ʹ�÷����Ժˣ��� DCF ��ѡ��һ�����Ժˣ�
%   It is meant to be called by the interface function RUN_TRACKER, which sets up the parameters and loads the video information.  
%   �����ɽӿں���RUN_TRACKER���õģ����ò�����������Ƶ��Ϣ   
%   Parameters:VIDEO_PATH is the location of the image files (must end with a slash'/' or '\').    
        
%     IMG_FILES is a cell array of image file names.  % img_files��ͼ���ļ����ĵ�Ԫ�����顣

%     POS and TARGET_SZ are the initial position and size of the target  (both in format [rows, columns]).  
%     pos, target_sz��Ŀ��ĳ�ʼλ�úʹ�С

%     PADDING is the additional tracked region, for context, relative to the target size.
%     padding�������Ŀ���С�ĸ��Ӹ����������������ġ�

%     KERNEL is a struct describing the kernel.  kernel��һ�������ں˵Ľṹ��

%     The field TYPE must be one of 'gaussian', 'polynomial' or 'linear'.
%     �ֶ�TYPE�����ǡ���˹����������ʽ�������ԡ�֮һ�� The optional fields SIGMA, POLY_A and POLY_B are the parameters 
%     for the Gaussian and Polynomial kernels.   SIGMA, POLY_A and POLY_B�Ǹ�˹�Ͷ���ʽ�ں˵Ĳ��� 
    
%     OUTPUT_SIGMA_FACTOR is the spatial bandwidth of the regression target, relative to the target size
%     output_sigma_factor �ع�Ŀ�������Ŀ���С�Ŀռ����**************************
        


	resize_image = (sqrt(prod(target_sz)) >= 200);%prod��A����ͬά��Ԫ�صĳ˻����ص�����B�� 
    
    if resize_image
		pos = floor(pos / 2);
		target_sz = floor(target_sz / 2);
    end %y = floor(x) ������x��Ԫ��ȡ����ֵyΪ�����ڱ������С����
    %resize_imageָʾ�Ƿ���Ҫ��ԭͼ�������ţ���Ϊ�����������ʱ���㷨��ͼ��������²�����
    %�п��ܸ��������õ�ͼ���ԭͼҪС�������Ҫ�˲�������ָʾ��

	%window size, taking padding into account ���ڴ�С����padding���ǽ�ȥ
	window_sz = floor(target_sz * ( 1+padding));%���ٿ�Ĵ�С
	
% 	we could choose a size that is a power of two, for better FFT performance
% 	���ǿ���ѡ��һ����СΪ2���ݣ����ڸ��õ�FFT���֡� 
%   in practice it is slower, due to the larger window size.
%   ��ʵ���У����ڴ��ڳߴ�ϴ��ٶȽ�����% 	window_sz = 2 .^ nextpow2(window_sz);
%   create regression labels, gaussian shaped, with a bandwidth proportional to target size
%   �����ع��ǩ����˹�Σ�������Ŀ��ߴ�ɱ���
	
	output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;%����������
    % output_sigma_factor �� �ع�Ŀ�������Ŀ���С�Ŀռ����
    % B = prod(A)��A����ͬά��Ԫ�صĳ˻����ص�����B��
    
	yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));%floor����ȡ��
    %cell_size
    %imshow(abs(yf));
    yf=single(yf);
	cos_window = hann(size(yf,1)) * hann(size(yf,2))';
    % 
	if show_visualization  %create video interface ������Ƶ����
		update_visualization = show_video(img_files, video_path, resize_image);
    end
	
	%note: variables ending with 'f' are in the Fourier domain.   %�����ڸ���Ҷ������'f'��β��
	time = 0;  %to calculate FPS
	positions = zeros(numel(img_files), 2);  %to calculate precision
    % POSITIONS��Ŀ��λ����ʱ�����Ƶ�Nx2����
    
	for frame = 1:numel(img_files),
		%load image ����ͼƬ
		im = imread([video_path img_files{frame}]);
        
        patch_rgb= get_subwindow(im, pos, window_sz);%ͼ���
        
		if size(im,3) > 1
			im = rgb2gray(im);%����3ͨ���ģ�����ɫ��ƬתΪ�Ҷ���Ƭ
		end
		if resize_image
			im = imresize(im, 0.5);%�ú������ڶ�ͼ�������Ŵ���
		end

		tic()

		if frame > 1
            %obtain a subwindow for detection at the position from last
            %frame, and convert to Fourier domain (its size is unchanged)
            %�����һ֡��λ�û�ȡ����Ӵ��ڣ���ת��Ϊ����Ҷ�����С���䣩
			patch = get_subwindow(im, pos, window_sz);%ͼ���
			zf = fft2(get_features(patch, features, cell_size, cos_window));
			%����ȡ�����������и���Ҷ�任���õ�zf
			%calculate response of the classifier at all shifts %������ת���м������������Ӧ
			switch kernel.type
                case 'gaussian',
                    kzf = gaussian_correlation(zf, model_xf, kernel.sigma);%����zf��ģ�͵ĺ���ؾ���	
                case 'polynomial',%zf���������ĸ���Ҷ�任��
                    kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);%model_xf
                case 'linear',
                    kzf = linear_correlation(zf, model_xf);
            end
            %������Ӧֵ  %equation for fast detection ���ټ�ⷽ��
           
			response = real(ifft2(model_alphaf .* kzf)); 
%             figure(11)
%             shift= floor(window_sz/cell_size);
%             haha = circshift(response, -floor(shift(1:2) / 2) +1);
%             imshow(haha);
%             response=haha;
			%target location is at the maximum response. we must take into account the fact that,
			% if the target doesn't move, the peak will appear at the top-left corner,
			% not at the center (this is discussed in the paper).the responses wrap around cyclically.
			% Ŀ��λ�ô��������Ӧ�� ���Ǳ��뿼�ǵ������Ŀ�겻�ƶ�����ֵ�����������Ͻǣ�
            %�����������ģ����ڱ��������ۣ�����Щ��Ӧ�����Ե��ƹ���
            
			[vert_delta, horiz_delta] = find(response == max(response(:)), 1);
            %���ݼ���ֵ�ھ����е�λ�ã������ǰ֡��Ŀ�����ĵ�Ԥ��ֵ
             %find(X,k)������X�е�k������Ԫ�ص�����λ�ã�[vert_delta, horiz_delta]����λ��
             
            %�ں�����˲���Ŀ������У������ӦֵԽ������Χ����ɢ��ԽС���������Ӧ
            %ֵ��Ӧ��λ��ΪĿ������λ�õ����Ŷ�Խ�ߡ�
           
% 			if vert_delta > size(zf,1) / 2, 
%                 %wrap around to negative half-space of vertical axis
%                 %���Ƶ���ֱ��ĸ���ռ�
% 				vert_delta = vert_delta - size(zf,1);
% 			end
% 			if horiz_delta > size(zf,2) / 2,  %same for horizontal axis ˮƽ����ͬ
% 				horiz_delta = horiz_delta - size(zf,2);
% 			end
% 			pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];%posλ��
%��������floor����ȡ�����Ա�ceil����ȡ��Ч�����ã�
            pos=pos+cell_size*[vert_delta-floor(size(response,1)/2), horiz_delta-floor(size(response,2)/2)];
        end %ͨ�����ټ�ⷽ�̵õ�Ŀ�꣬Ȼ��ͨ��Ŀ��õ�pos������λ��
        
%******************************%ģ�͸��¹���
		%obtain a subwindow for training at newly estimated target position
        %���¹��Ƶ�Ŀ��λ�û��ѵ���Ӵ���
		patch = get_subwindow(im, pos, window_sz);
		xf = fft2(get_features(patch, features, cell_size, cos_window));%��ͼ������ȡ�ܼ�����ת������Ҷ��
        % get_features��ͼ������ȡ�ܼ������������и���Ҷ�任
		%Kernel Ridge Regression, calculate alphas (in Fourier domain)
        
        %����ع飬�����������Ҷ��
		switch kernel.type%ѡ����ʲô��
            case 'gaussian',%�ø�˹��
                kf = gaussian_correlation(xf, xf, kernel.sigma);%kernel.sigma��˹�ں˴���	
            case 'polynomial',
                kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
            case 'linear',
                kf = linear_correlation(xf, xf);
        end
        
		alphaf = yf ./ (kf + lambda);   %equation for fast training����ѵ������ʽ
        %lambda���滯
        
		if frame == 1,  %first frame, train with a single image��һ֡���õ�һͼ��ѵ��
			model_alphaf = alphaf;
			model_xf = xf;
		else
			%subsequent frames, interpolate model����֡����ֵģ��
			model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
            %interp_factor�Ǹ�����������Ӧ����
			model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
		end

		%save position and timing����λ�ú�ʱ��
		positions(frame,:) = pos;%ÿ֡��λ�ã���������������
		time = time + toc();

		%visualization���ӻ�
        show_visualization=1;
		if show_visualization
           vis_response;
% 			box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];%ʲô��˼
% 			stop = update_visualization(frame, box);
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             %����ͼƬ
%             savePath = sprintf('./image/%d.jpg',frame);
%             imwrite(frame2im(getframe(gcf)),savePath); 
%             hold off;
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 			if stop, break, end  %user pressed Esc, stop early
% 			drawnow
% 			pause(0.01)  %uncomment to run slower
		end
		
	end

	if resize_image,
		positions = positions * 2;
	end
end


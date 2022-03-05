function [results] = dsst(params)
% parameters
padding = params.padding;                         	%extra area surrounding the target
output_sigma_factor = params.output_sigma_factor;	%spatial bandwidth (proportional to target)
lambda = params.lambda;
learning_rate = params.learning_rate;
nScales = params.number_of_scales;
scale_step = params.scale_step;
scale_sigma_factor = params.scale_sigma_factor;
scale_model_max_area = params.scale_model_max_area;

video_path  = params.video_path; 
s_frames = params.s_frames; % ȫ��ͼ���ļ�
pos = floor(params.init_pos);
target_sz = floor(params.wsize);
visualization = params.visualization;
num_frames = numel(s_frames); % ֡��

init_target_sz = target_sz;
base_target_sz = target_sz; % target size att scale = 1
sz = floor(base_target_sz * (1 + padding));% window size, taking padding into account

% desired translation filter output (gaussian shaped), bandwidth proportional to target size
output_sigma = sqrt(prod(base_target_sz)) * output_sigma_factor;
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
y = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
yf = single(fft2(y));

% desired scale filter output (gaussian shaped), bandwidth proportional to number of scales
scale_sigma = nScales/sqrt(33) * scale_sigma_factor;
ss = (1:nScales) - ceil(nScales/2);
ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
ysf = single(fft(ys));

cos_window = single(hann(sz(1)) * hann(sz(2))'); % store pre-computed translation filter cosine window

% store pre-computed scale filter cosine window
if mod(nScales,2) == 0
    scale_window = single(hann(nScales+1));
    scale_window = scale_window(2:end);
else
    scale_window = single(hann(nScales));
end;

ss = 1:nScales; % scale factors
scaleFactors = scale_step.^(ceil(nScales/2) - ss);

% compute the resize dimensions used for feature extraction in the scale estimation
scale_model_factor = 1;
if prod(init_target_sz) > scale_model_max_area
    scale_model_factor = sqrt(scale_model_max_area/prod(init_target_sz));
end
scale_model_sz = floor(init_target_sz * scale_model_factor);

currentScaleFactor = 1;

rect_position = zeros(num_frames, 4); % to calculate precision

time = 0; % to calculate FPS

% find maximum and minimum scales
try
    im = imread([video_path '/img/' s_frames{1}]);
catch
    try
        im = imread(s_frames{1});
    catch
        im = imread([video_path '/' s_frames{1}]);
    end
end

min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));
max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));

for frame = 1:num_frames,
    try
        im = imread([video_path '/img/' s_frames{frame}]);
    catch
        try
            im = imread([s_frames{frame}]);
        catch
            im = imread([video_path '/' s_frames{frame}]);
        end
    end
    tic;
    
    if frame > 1
        
        % extract the test sample feature map for the translation filter
        xt = get_translation_sample(im, pos, sz, currentScaleFactor, cos_window);
        
        % calculate the correlation response of the translation filter
        xtf = fft2(xt);
        response = real(ifft2(sum(hf_num .* xtf, 3) ./ (hf_den + lambda)));
%         disp(num2str(size(xtf)))
%         kxtf=sum(xtf .* conj(model_xlf), 3) / numel(xtf);
%         response = real(ifft2(model_alphaf .* kxtf)); 
 
        [row, col] = find(response == max(response(:)), 1);% find the maximum translation response
        
        pos = pos + round((-sz/2 + [row, col]) * currentScaleFactor);  % update the position
        
        % extract the test sample feature map for the scale filter
        xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
        
        % calculate the correlation response of the scale filter
        xsf = fft(xs,[],2);
        scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + lambda)));
%         kxsf=sum(xsf .* conj(model_xsf), 1) / numel(xsf);
%         scale_response = real(ifft2(model_salphaf .* kxsf)); 
        
        % find the maximum scale response
        recovered_scale = find(scale_response == max(scale_response(:)), 1);
        
        % update the scale
        currentScaleFactor = currentScaleFactor * scaleFactors(recovered_scale);
        if currentScaleFactor < min_scale_factor
            currentScaleFactor = min_scale_factor;
        elseif currentScaleFactor > max_scale_factor
            currentScaleFactor = max_scale_factor;
        end
    end
    
    % extract the training sample feature map for the translation filter
    xl = get_translation_sample(im, pos, sz, currentScaleFactor, cos_window);
    
    % calculate the translation filter update
    xlf = fft2(xl);
%     kxlf=sum(xlf .* conj(xlf), 3) / numel(xlf);
%     alphaf = yf ./ (kxlf + lambda);
    new_hf_num = bsxfun(@times, yf, conj(xlf));
    new_hf_den = sum(xlf .* conj(xlf), 3);
    % extract the training sample feature map for the scale filter
    xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
    
    % calculate the scale filter update
    xsf = fft(xs,[],2);
    new_sf_num = bsxfun(@times, ysf, conj(xsf));
    new_sf_den = sum(xsf .* conj(xsf), 1);
%     kxsf=sum(xsf .* conj(xsf), 1) / numel(xsf);
%     salphaf = ysf ./ (kxsf + lambda);
    
    if frame == 1
        % first frame, train with a single image
%         model_alphaf = alphaf;
% 		model_xlf = xlf;
%         model_salphaf = salphaf;
% 		model_xsf = xsf;
        hf_den = new_hf_den;
        hf_num = new_hf_num;
        sf_den = new_sf_den;
        sf_num = new_sf_num;
    else
        % subsequent frames, update the model
%         model_alphaf = (1 - learning_rate) * model_alphaf + learning_rate * alphaf;
% 	    model_xlf = (1 - learning_rate) * model_xlf + learning_rate * xlf;
%         model_salphaf = (1 - learning_rate) * model_salphaf + learning_rate * salphaf;
% 	    model_xsf = (1 - learning_rate) * model_xsf + learning_rate * xsf;
        hf_den = (1 - learning_rate) * hf_den + learning_rate * new_hf_den;
        hf_num = (1 - learning_rate) * hf_num + learning_rate * new_hf_num;
        sf_den = (1 - learning_rate) * sf_den + learning_rate * new_sf_den;
        sf_num = (1 - learning_rate) * sf_num + learning_rate * new_sf_num;
    end
    
    target_sz = floor(base_target_sz * currentScaleFactor);% calculate the new target size
    rect_position(frame,:) = [pos([2,1]) - floor(target_sz([2,1])/2), target_sz([2,1])];
    
    time = time + toc;
    
    %visualization
    if visualization == 1
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        if frame == 1,  %first frame, create GUI
            figure;
            im_handle = imshow(uint8(im), 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
            rect_handle = rectangle('Position',rect_position_vis, 'EdgeColor','g');
            text_handle = text(10, 10, int2str(frame));
            set(text_handle, 'color', [0 1 1]);
        else
            try  %subsequent frames, update GUI
                set(im_handle, 'CData', im)
                set(rect_handle, 'Position', rect_position_vis)
                set(text_handle, 'string', int2str(frame));
            catch
                return
            end
        end   
        drawnow
%         pause
    end
end

fps = numel(s_frames) / time;
results.type = 'rect';
results.res = rect_position;
results.fps = fps;
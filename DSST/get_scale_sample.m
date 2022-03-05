function out = get_scale_sample(im, pos, base_target_sz, scaleFactors, scale_window, scale_model_sz)

% out = get_scale_sample(im, pos, base_target_sz, scaleFactors, scale_window, scale_model_sz)
% 
% Extracts a sample for the scale filter at the current
% location and scale.

nScales = length(scaleFactors);
%ע�������patch_sz��û�п���padding������ֱ�Ӷ�target���任
for s = 1:nScales
    patch_sz = floor(base_target_sz * scaleFactors(s));    
    %scaleFactors�������ǲ�ͬ�ķŴ������33ά�������1.5--0.5֮��ģ����������Ե�
    %
    xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
    ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);
    %�������pos��patch�õ�xs��ys��������Χ��
    
    % check for out-of-bounds coordinates, and set them to the values at
    % the borders
    %����߽�
    xs(xs < 1) = 1;
    ys(ys < 1) = 1;
    xs(xs > size(im,2)) = size(im,2);
    ys(ys > size(im,1)) = size(im,1);
    
    % extract image�õ���ǰ�ߴ��µ�ͼ��
    im_patch = im(ys, xs, :);
    
    % resize image to model size
    % �ع�ͳһ�Ĵ�С�������scale_model_sz�Ǹ���һ��ʼ��Ŀ���С���ģ���dsst������������19*26
    im_patch_resized = mexResize(im_patch, scale_model_sz, 'auto');
    
    % extract scale features
    % �õ�31ά��hog������19/4*26/4*31=4*6*31=744ά�������������fhog�Ǵ��������ˣ���û�в���
    temp_hog = fhog(single(im_patch_resized), 4);  %cell��4*4��
    temp = temp_hog(:,:,1:31);      %ֻȡ��ǰ31ά��hog�����һά�ǻҶȣ���û��Ҫ    
    
    if s == 1                 %����ǵ�һ���߶ȣ�����һ��out�����������
        out = zeros(numel(temp), nScales, 'single');
    end
    %  window 744*3
    out(:,s) = temp(:) * scale_window(s);%scale_window���ڳ߶���
end
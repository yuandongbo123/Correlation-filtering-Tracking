function out = get_feature_map(im_patch)

% allocate space
out = zeros(size(im_patch, 1), size(im_patch, 2), 28, 'single');

% if grayscale image
if size(im_patch, 3) == 1
    out(:,:,1) = single(im_patch)/255 - 0.5;
    temp = fhog(single(im_patch), 1);%cell大小是1？
    out(:,:,2:28) = temp(:,:,1:27);
else
    out(:,:,1) = single(rgb2gray(im_patch))/255 - 0.5;%归一化-0.5-0.5  而不是0-1
    temp = fhog(single(im_patch), 1);
    out(:,:,2:28) = temp(:,:,1:27);
end

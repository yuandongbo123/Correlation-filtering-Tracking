function x = get_features(im, features, cell_size, cos_window)
%GET_FEATURES
%   Extracts dense features from image.  从图像中提取密集特征。
%   Extracts features specified in struct FEATURES, from image IM. The features should be densely
%   sampled,in cells or intervals of CELL_SIZE.从图像中密集采样，在CELL_SIZE的单元格或间隔中
%   The output has size[height in cells, width in cells, features].输出具有大小[单元格中的高度，单元格中的宽度，特征]  

%   To specify HOG features, set field 'hog' to true, and'hog_orientations' to the number of bins.  
%   要指定HOG功能，请将字段“hog”设置为true，将“hog_orientations”设置为bin数。

%   To experiment with other features simply add them to this function
%   and include any needed parameters in the FEATURES struct.
%   要实验其他功能，只需将其添加到此功能并在FEATURES结构中包含所需的任何参数。

%   To allow combinations of features, stack them with x = cat(3, x, new_feat).
	if features.hog
		%HOG features, from Piotr's Toolbox来自Piotr的工具箱的HOG功能
		x = double(fhog(single(im) / 255, cell_size, features.hog_orientations));
		x(:,:,end) = [];  
        %remove all-zeros channel ("truncation feature")
        %删除全零通道（“截断功能”）
	end
	
	if features.gray
		%gray-level (scalar feature)灰度级（标量特征）
		x = double(im) / 255;
		
		x = x - mean(x(:));%归一化到   -0.5-0.5
	end
	
	%process with cosine window if needed如果需要，使用余弦窗口进行处理
	if ~isempty(cos_window)
		x = bsxfun(@times, x, cos_window);%在每一个通道上进行点乘
	end
	
end

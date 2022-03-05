function update_visualization_func = show_video(img_files, video_path, resize_image)
%SHOW_VIDEO
%   Visualizes a tracker in an interactive figure, given a cell array of image file names,
%   their path, and whether to resize the images to half size or not.
%   ����ͼ���ļ����Ƶĵ�Ԫ�����飬���ӻ�����ʽͼ���еĸ������� ���ǵ�·�����Լ��Ƿ�ͼ���С������һ�롣
%   This function returns an UPDATE_VISUALIZATION function handle, that
%   can be called with a frame number and a bounding box [x, y, width,
%   height], ������֡�źͱ߿�[x��y��width��height]����
%   as soon as the results for a new frame have been calculated.һ���������¿�ܵĽ����

%   This way, your results are shown in real-time, but they are also remembered so you can navigate
%   and inspect the video afterwards.���������Ľ����ʱ��ʾ����Ҳ����ס���������������ͼ�����Ƶ
%�����ٵõ���Ŀ��λ�û���ԭͼ���ϣ���������ʾ��

	%store one instance per frame ÿ֡�洢һ��ʵ��
	num_frames = numel(img_files);
	boxes = cell(num_frames,1);

	%create window
	[fig_h, axes_h, unused, scroll] = videofig(num_frames, @redraw, [], [], @on_key_press);  %#ok, unused outputs
	set(fig_h, 'Name', ['Tracker - ' video_path])
	axis off;
	
	%image and rectangle handles start empty, they are initialized later
    %ͼ��;��ξ����ʼΪ�գ������Ժ󱻳�ʼ��
	im_h = [];
	rect_h = [];
	tt_h=[];
    
	update_visualization_func = @update_visualization;
	stop_tracker = false;
	

	function stop = update_visualization(frame, box)
		%store the tracker instance for one frame, and show it. returns
		%true if processing should stop (user pressed 'Esc').
        %��������ʵ���洢һ֡������ʾ���� �������ֹͣ���û����¡�Esc�������򷵻�true��
		boxes{frame} = box;
         
		scroll(frame);
              
		stop = stop_tracker;
        
	end

	function redraw(frame)
		%render main image
		im = imread([video_path img_files{frame}]);
		if size(im,3) > 1,
			im = rgb2gray(im);
		end
		if resize_image,
			im = imresize(im, 0.5);
		end
		
		if isempty(im_h),  %create image
			im_h = imshow(im, 'Border','tight', 'InitialMag',200, 'Parent',axes_h);
		else  %just update it
			set(im_h, 'CData', im)
            
          end
		
		%render target bounding box for this frame
		if isempty(rect_h),  %create it for the first time
			rect_h = rectangle('Position',[0,0,1,1], 'EdgeColor','g', 'Parent',axes_h);
            numStr   = sprintf('#%03d',frame);
            tt_h=text(10,20,numStr,'Color','r', 'FontWeight','bold', 'FontSize',10);
		end
		if ~isempty(boxes{frame}),
			set(rect_h, 'Visible', 'on', 'Position', boxes{frame});
            numStr   = sprintf('#%03d',frame);
            set(tt_h, 'Visible', 'on','String',numStr);
      	else
			set(rect_h, 'Visible', 'off');
            set(tt_h,'Visible','off');
		end
	end

	function on_key_press(key)
		if strcmp(key, 'escape'),  %stop on 'Esc'
			stop_tracker = true;
		end
	end

end


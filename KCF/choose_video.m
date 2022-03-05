function video_name = choose_video(base_path)
%CHOOSE_VIDEO
%   Allows the user to choose a video (sub-folder in the given path).
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/

	%process path to make sure it's uniform
	if ispc(), base_path = strrep(base_path, '\', '/'); end%ispc用来判断当前电脑是不是Windows系统，是返回1，不是返回0
	%strrep()
    if base_path(end) ~= '/', base_path(end+1) = '/'; end
	
	%list all sub-folders  列出所有子文件夹
	contents = dir(base_path);%dir('G:\Matlab')列出指定目录下所有子文件夹和文件
	names = {};
	for k = 1:numel(contents),%numel()：返回数组或者向量中所含元素的总数。
		name = contents(k).name;
		if isdir([base_path name]) && ~any(strcmp(name, {'.', '..'})),%isdir用于判断输入是否表示一个文件夹
			%any函数作用：判断元素是否为非零元素any(v),如果v是非零元素返回true(即1)否则返回flase(即0)
            names{end+1} = name;  %#ok
		end
	end
	
	%no sub-folders found  没有找到子文件夹
	if isempty(names), video_name = []; return; end
	%isempty(names)????判断names是否为空，如果为空，结果为1，否则为0.
    
	%choice GUI
	choice = listdlg('ListString',names, 'Name','Choose video', 'SelectionMode','single');
    %'Name','Choose video',代表的是选择框的名字是Choose video
	%listdlg列表选择对话框
	if isempty(choice),  %user cancelled
		video_name = [];
	else
		video_name = names{choice};
	end
	
end


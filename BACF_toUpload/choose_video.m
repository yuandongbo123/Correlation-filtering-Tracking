function video_name = choose_video(base_path)
%CHOOSE_VIDEO
%   Allows the user to choose a video (sub-folder in the given path).      允许用户选择视频（给定路径中的子文件夹）。
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/

	%process path to make sure it's uniform                                处理路径以确保它是统一的
	if ispc(), base_path = strrep(base_path, '\', '/'); end    %ispc - 确定版本是否适用于 Windows (PC) 平台。strrep - 查找并替换子字符串
	if base_path(end) ~= '/', base_path(end+1) = '/'; end
	
	%list all sub-folders                                                      %列出所有子文件夹
	contents = dir(base_path);
	names = {};
	for k = 1:numel(contents),                                        %numel - 数组元素的数目  == prod(size(A))
		name = contents(k).name;
		if isfolder([base_path name]) && ~any(strcmp(name, {'.', '..'})),
			names{end+1} = name;  %#ok
		end
	end
	
	%no sub-folders found                                                  未找到子文件夹
	if isempty(names), video_name = []; return; end
	
	%choice GUI                                                            选择GUI
	choice = listdlg('ListString',names, 'Name','Choose video', 'SelectionMode','single');  %listdlg - 创建列表选择对话框
	                                                                                                                                         %'Name','Choose video'  Name(对话框标题） Choose video
                                                                                                                                             %'SelectionMode','single' SelectionMode（选择模式） single
                                                                                                                                             %'ListString',names  ListString（传入名称） {names}
                                                                                                                                             %choice(返回的是索引）
	if isempty(choice),  %user cancelled                                   用户取消
		video_name = [];
	else
		video_name = names{choice};
	end
	
end


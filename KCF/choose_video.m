function video_name = choose_video(base_path)
%CHOOSE_VIDEO
%   Allows the user to choose a video (sub-folder in the given path).
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/

	%process path to make sure it's uniform
	if ispc(), base_path = strrep(base_path, '\', '/'); end%ispc�����жϵ�ǰ�����ǲ���Windowsϵͳ���Ƿ���1�����Ƿ���0
	%strrep()
    if base_path(end) ~= '/', base_path(end+1) = '/'; end
	
	%list all sub-folders  �г��������ļ���
	contents = dir(base_path);%dir('G:\Matlab')�г�ָ��Ŀ¼���������ļ��к��ļ�
	names = {};
	for k = 1:numel(contents),%numel()�����������������������Ԫ�ص�������
		name = contents(k).name;
		if isdir([base_path name]) && ~any(strcmp(name, {'.', '..'})),%isdir�����ж������Ƿ��ʾһ���ļ���
			%any�������ã��ж�Ԫ���Ƿ�Ϊ����Ԫ��any(v),���v�Ƿ���Ԫ�ط���true(��1)���򷵻�flase(��0)
            names{end+1} = name;  %#ok
		end
	end
	
	%no sub-folders found  û���ҵ����ļ���
	if isempty(names), video_name = []; return; end
	%isempty(names)????�ж�names�Ƿ�Ϊ�գ����Ϊ�գ����Ϊ1������Ϊ0.
    
	%choice GUI
	choice = listdlg('ListString',names, 'Name','Choose video', 'SelectionMode','single');
    %'Name','Choose video',�������ѡ����������Choose video
	%listdlg�б�ѡ��Ի���
	if isempty(choice),  %user cancelled
		video_name = [];
	else
		video_name = names{choice};
	end
	
end


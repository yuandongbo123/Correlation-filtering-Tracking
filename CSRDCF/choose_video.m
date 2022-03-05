function video_name = choose_video(base_path)
%CHOOSE_VIDEO
%   Allows the user to choose a video (sub-folder in the given path).      �����û�ѡ����Ƶ������·���е����ļ��У���
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/

	%process path to make sure it's uniform                                ����·����ȷ������ͳһ��
	if ispc(), base_path = strrep(base_path, '\', '/'); end    %ispc - ȷ���汾�Ƿ������� Windows (PC) ƽ̨��strrep - ���Ҳ��滻���ַ���
	if base_path(end) ~= '/', base_path(end+1) = '/'; end
	
	%list all sub-folders                                                      %�г��������ļ���
	contents = dir(base_path);
	names = {};
	for k = 1:numel(contents),                                        %numel - ����Ԫ�ص���Ŀ  == prod(size(A))
		name = contents(k).name;
		if isfolder([base_path name]) && ~any(strcmp(name, {'.', '..'})),
			names{end+1} = name;  %#ok
		end
	end
	
	%no sub-folders found                                                  δ�ҵ����ļ���
	if isempty(names), video_name = []; return; end
	
	%choice GUI                                                            ѡ��GUI
	choice = listdlg('ListString',names, 'Name','Choose video', 'SelectionMode','single');  %listdlg - �����б�ѡ��Ի���
	                                                                                                                                         %'Name','Choose video'  Name(�Ի�����⣩ Choose video
                                                                                                                                             %'SelectionMode','single' SelectionMode��ѡ��ģʽ�� single
                                                                                                                                             %'ListString',names  ListString���������ƣ� {names}
                                                                                                                                             %choice(���ص���������
	if isempty(choice),  %user cancelled                                   �û�ȡ��
		video_name = [];
	else
		video_name = names{choice};
	end
	
end


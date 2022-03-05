% compile hog and segmetation mex files
% before compiling set the following variables to the correct paths:
% opencv_include and opencv_libpath

current_folder = pwd;
% mkdir('mex');
% %filesep代表'/'
% cd(['mex_src' filesep 'hog']);
% mex gradientMex.cpp
% movefile('*.mex*', [current_folder filesep 'mex'])
cd(current_folder);

if ispc     % Windows machine
    % set opencv include path
    %  opencv_include = 'E:\development\opencv-2.4.12\opencv\build\include\';
    % opencv_include = 'F:\opencv\opencv3.4.1_contrib_install\install\include\';
    % set opencv lib path
    %opencv_libpath = 'E:\development\opencv-2.4.12\opencv\build\x64\vc11\lib\';
    %opencv_libpath = 'F:\opencv\opencv3.4.1_contrib_install\install\x64\vc15\lib\';

    files = dir([opencv_libpath '*opencv*.lib']);
    lib = [];
    for i = 1:length(files),
        lib = [lib ' -l' files(i).name(1:end-4)];
    end
    cd(['mex_src' filesep 'segmentation']);
    eval(['mex mex_extractforeground.cpp src\segment.cpp -Isrc\ -I' opencv_include ' -L' opencv_libpath ' ' lib]);
    eval(['mex mex_extractbackground.cpp src\segment.cpp -Isrc\ -I' opencv_include ' -L' opencv_libpath ' ' lib]);
    eval(['mex mex_segment.cpp src\segment.cpp -Isrc\ -I' opencv_include ' -L' opencv_libpath ' ' lib]);
    movefile('*.mex*', [current_folder filesep 'mex'])
    cd(current_folder);

elseif isunix   % Unix machine
    % set opencv include path
    opencv_include = '/usr/local/include/';
    % set opencv lib path
    opencv_libpath = '/usr/local/lib/';

    lib = [];
    files = dir([opencv_libpath '*opencv*.so.3.4.1']);
    for i = 1:length(files)
        lib = [lib ' -l' files(i).name(4:end-9)];
    end

    cd(['mex_src' filesep 'segmentation']);
    eval(['mex mex_extractforeground.cpp src/segment.cpp -Isrc/ -I' opencv_include ' -L' opencv_libpath ' ' lib]);
    eval(['mex mex_extractbackground.cpp src/segment.cpp -Isrc/ -I' opencv_include ' -L' opencv_libpath ' ' lib]);
    eval(['mex mex_segment.cpp src/segment.cpp -Isrc/ -I' opencv_include ' -L' opencv_libpath ' ' lib]);
    movefile('*.mex*', [current_folder filesep 'mex'])
    cd(current_folder);

end

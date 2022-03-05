function [seq, im] = get_sequence_frame(seq,video_path)

seq.frame = seq.frame + 1;

if strcmpi(seq.format, 'otb')
    if seq.frame > seq.num_frames
        im = [];
    else
        im = imread([video_path seq.image_files{seq.frame}]);
    end
elseif strcmpi(seq.format, 'vot')
    [seq.handle, image_file] = seq.handle.frame(seq.handle);
    if isempty(image_file)
        im = [];
    else
        im = imread(image_file);
    end
else
    error('Uknown sequence format');
end
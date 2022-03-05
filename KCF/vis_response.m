% Do a reflection in the fourier domain for a 2-dimensional signal

%res
% function xf_reflected = reflect_spectrum2(xf)
% 
% xf_reflected = circshift(flip(flip(xf, 1), 2), [1 1 0]);

if frame == 1
    res_vis_sz = [500, 500];
    fig_handle_detail = figure('Name', 'Details','Position',[500, 500, res_vis_sz]);
    anno_handle = annotation('textbox',[0.3,0.92,0.7,0.08],'LineStyle','none',...
        'String',['Tracking Frame #' num2str(frame)]);
else
    
    figure(fig_handle_detail);
    
    set(anno_handle, 'String',['Tracking Frame #' num2str(frame)]);
    
    set(gca, 'position', [0 0 1 1 ]);
    
    patch_to_show = imresize(patch_rgb(:,:,:),res_vis_sz);
    
    imagesc(patch_to_show);
    
    f=getframe(gcf);
    imwrite(f.cdata,['./pic/raw/',num2str(frame),'.jpg']);
    %saveas(gcf,'./pic/raw/',num2str(seq.frame),'jpg')%
    hold on;
    
%     %响应图可视化
%     sampled_scores_display = circshift(imresize(response(:,:,sind),res_vis_sz),floor(0.5*res_vis_sz));
%     %sampled_scores_display=imresize(abs(real(ifft2(g_f(:,:,1)))), res_vis_sz);
%     resp_handle = imagesc(abs(sampled_scores_display));
%     alpha(resp_handle, 1);
%     hold off;
%     axis off;
%     title('Response map');
%     f=getframe(gcf);
%     imwrite(f.cdata,['./pic/response/',num2str(seq.frame),'.jpg']);
    % 滤波器可视化
    % imagesc(ifft2(conj(hf(:,:,1)), 'symmetric'));
  
    sampled_scores_display=imresize(abs(real(ifft2(model_xf(:,:,1)))), res_vis_sz);
    resp_handle = imagesc(abs(sampled_scores_display));
    alpha(resp_handle, 0.6);
    hold off;
    axis off;
    title('filter map');
    f=getframe(gcf);
    imwrite(f.cdata,['./pic/filter/',num2str(frame),'.jpg']);
    
    %saveas(gcf,'./pic/response/',num2str(seq.frame),'jpg')
end

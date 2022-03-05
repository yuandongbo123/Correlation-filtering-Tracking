
clc;
close all;
run_tracker();

% %%CF滤波循环移位样本可视化
% x = imread('.\image\1.jpg');
% [h,w]=size(x);
% for i = 1:20:size(x,1)  % 50为步长，可自行设置
%    for j = 1:20:size(x,2)
%          %如果同时对矩阵进行行和列的移位则令K= [col，row]，其中col表示列位移，row表示行位移。
%           img = circshift(x,[i j]);%circshift(x,K,m);当K是数字，m=1列移位，m=2行移位，k>0向下，k<0向上
%          % figure(101),imshow(img,[]);
%           figure(1),imshow(img,'border','tight','initialmagnification','fit');
%           % 去掉边框
%           pause;
%     end
% end






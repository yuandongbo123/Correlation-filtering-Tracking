function H = fhog( I, binSize, nOrients, clip, crop )
% Efficiently compute Felzenszwalb's HOG (FHOG) features.
%
% A fast implementation of the HOG variant used by Felzenszwalb et al.
% in their work on discriminatively trained deformable part models.形变模型
%  http://www.cs.berkeley.edu/~rbg/latent/index.html
% Gives nearly identical results to features.cc in code release version 5
% but runs 4x faster (over 125 fps on VGA color images).
%
% The computed HOG features are 3*nOrients+5 dimensional. There are
% 2*nOrients contrast sensitive orientation channels, nOrients contrast
% insensitive orientation channels, 4 texture channels and 1 all zeros
% channel (used as a 'truncation' feature). Using the standard value of
% nOrients=9 gives a 32 dimensional feature vector at each cell. This
% variant of HOG, refered to as FHOG, has been shown to achieve superior
% performance to the original HOG features. For details please refer to
% work by Felzenszwalb et al. (see link above).
% 所以论文中Hog特征的提取是将sample区域划分成若干的区域，
%然后再每个区域提取特征，代码中是在每个区域提取了32维特征，即,
%其中就是梯度方向划分的bin个数，每个方向提取了3个特征，2个是对方向bin敏感的，
%1个是不敏感的，另外4个特征是关于表观纹理的特征还有一个是零，表示阶段特征，
%具体参见fhog。提取了31个特征(最后一个0不考虑)之后，不是串联起来，
%而是将每个cell的特征并起来，那么一幅图像得到的结果就是一个立体块，
%假设划分cell的结果是,那么fhog提取结果就是,我们成31这个方向为通道。
%那么就可以通过cell的位移来获得样本，这样对应的就是每一通道对应位置的移位，             
%所有样本的第i通道都是有生成图像的第i通道移位获得的，
%，所以分开在每一个通道上计算，就可以利用循环矩阵的性质了。


% This function is essentially a wrapper for calls to gradientMag()and gradientHist().
% 此函数本质上是一个用于调用gradientMag（）和gradientHist（）。

%Specifically, it is equivalent to the following:具体来说，它等同于以下内容：
%  [M,O] = gradientMag( I,0,0,0,1 ); softBin = -1; useHog = 2;
%  H = gradientHist(M,O,binSize,nOrients,softBin,useHog,clip);
% See gradientHist() for more general usage.一般用法
%
% This code requires SSE2 to compile and run (most modern Intel and AMD processors support SSE2).
% 此代码需要SSE2进行编译并运行
% Please see: http://en.wikipedia.org/wiki/SSE2 .
% USAGE
%  H = fhog( I, [binSize], [nOrients], [clip], [crop] )
%
% INPUTS
%  I        - [hxw] color or grayscale input image (must have type single)
%    +++++++++  I是彩色或者灰度图像，必须是单通道类型
%  binSize  - [8] spatial bin size 
%    +++++++++    binSize是空间尺寸
%  nOrients - [9] number of orientation bins
%   +++++++++   nOrients    180度分成9个方向块
%  clip     - [.2] value at which to clip histogram bins 
%               clip 看看值在哪个块
%  crop     - [0] if true crop boundaries
%
% OUTPUTS
%  H        - [h/binSize w/binSize nOrients*3+5] computed hog features
%   ++++++ h是计算HOG特征的
%
% EXAMPLE
%  I=imResample(single(imread('peppers.png'))/255,[480 640]);
%  tic, for i=1:100, H=fhog(I,8,9); end; disp(100/toc) % >125 fps
%  figure(1); im(I); V=hogDraw(H,25,1); figure(2); im(V)
%
% EXAMPLE
%  % comparison to features.cc (requires DPM code release version 5)
%  I=imResample(single(imread('peppers.png'))/255,[480 640]); Id=double(I);
%  tic, for i=1:100, H1=features(Id,8); end; disp(100/toc)
%  tic, for i=1:100, H2=fhog(I,8,9,.2,1); end; disp(100/toc)
%  figure(1); montage2(H1); figure(2); montage2(H2);
%  D=abs(H1-H2); mean(D(:))
%
% See also hog, hogDraw, gradientHist
%
% Piotr's Image&Video Toolbox      Version 3.23
% Copyright 2013 Piotr Dollar.  [pdollar-at-caltech.edu]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see external/bsd.txt]

%Note: modified to be more self-contained
% H = fhog( I, binSize, nOrients, clip, crop )
%I是彩色或者灰度图像, binSize是空间尺寸,nOrients180度分成9个方向块,clip 看看值在哪个块

if( nargin<2 ), binSize=8;  end    
if( nargin<3 ), nOrients=9; end    
if( nargin<4 ), clip=.2;    end     
if( nargin<5 ), crop=0;     end   
%nargin是什么意思

softBin = -1; useHog = 2; b = binSize;   

[M,O]=gradientMex('gradientMag',I,0,1);%？？？？？？？？？？？？

H = gradientMex('gradientHist',M,O,binSize,nOrients,softBin,useHog,clip);

if( crop ), e=mod(size(I),b)<b/2; H=H(2:end-e(1),2:end-e(2),:); end
            %mod 是求模函数
end

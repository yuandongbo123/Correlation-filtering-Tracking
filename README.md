# Correlation-filtering-Tracking
 这是一个基于相关滤波跟踪算法的合集

## 1.KCF 
- 近年来相关滤波跟踪领域最具影响力的算法之一，主要思想就是利用了循环矩阵对角化等性质，使得核化后的岭回归计算变得简单。他的原理很简单但是论文中的公式很是令繁多，但细细推来却能让人眼前一亮。他的代码也少的令人感动！
[kcf原文链接](https://ieeexplore.ieee.org/abstract/document/6870486) [参考博客](https://blog.csdn.net/shenxiaolu1984/article/details/50905283?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164697489416780269823659%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=164697489416780269823659&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-2-50905283.pc_search_result_cache&utm_term=KCF&spm=1018.2226.3001.4187)

## 2.SAMF

- 在当年kcf横扫跟踪界之后，随后引发了相关滤波跟踪器的热潮，samf就是其中之一。samf有主要贡献两点，(1) 将单一的特征扩展为多个特征；(2) 利用尺度池的方法实现目标自适应跟踪。（这两点改进提升了准确性能，却降低了他的速度，可以说是有得有失）[samf原文](https://link.springer.com/chapter/10.1007%2F978-3-319-16181-5_18)  [参考博客](https://blog.csdn.net/weixin_38128100/article/details/80557315?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164697598016781685362342%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=164697598016781685362342&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-80557315.pc_search_result_cache&utm_term=SAMF&spm=1018.2226.3001.4187)

# Comparison of factor number estimators
This code provides a means of comparing performance of factor number estimators by a theoretical Monte Carlo study and an empirical study based on portfolio optimization.

I focus on 7 different estimators from 4 papers: $\text{PC}_1$, $\text{IC}_1$ and $\text{BIC}_3$ by Bai and Ng (2002), $\text{ED}$ by Onatski (2010), $\text{ER}$ and $\text{GR}$ by Ahn and Horenstein (2013), and $\text{TKCV}$ by Wei and Chen (2020).

The code presented here is an edited version of the code I have submitted to obtain the Master's Thesis from M.Sc. in Advanced Studies in Economics at KU Leuven in September 2024.

## Theoretical comparison

For theoretical comparison I employ four different Monte Carlo simulation designs. For each, a synthetic data is generated according to a data generating process (DGP) creating user determined temporal and cross sectional correlations. Then factor number are estimated from this synthetic data.

Running the following script produces factor numbers and their summary statistics, when the estimation is done on DGP with the parameters `[1]` to `[7]`.

```python
monte_carlo_simul_nt [1] [2] [3] [4] [5] [6] [7]
```

where 
```
[1]: true number of factors (int)
[2]: upper limit on factor estimation (int)
[3]: SNR - signal to noise ratio (float)
[4]: rho - strength of temporal correlations (float)
[5]: beta - strength of cross sectional correlations (float)
[6]: type of cross sectional correlation growth (string: "static"/"dynamic")
[7]: number of Monte Carlo iterations (int)
```


## Empirical comparison

To judge the performance of the studied factor number estimators on real data, I have done a portfolio optimization study. Specifically, I have explored the performance of global minimal variance (GMV) portfolios formed with POET covariance matrix estimator, developed by Fan et al. (2013). This covariance matrix estimator itself requires a factor number estimator for its functioning.

For data, I have used the daily returns of S&P500 index constituents, obtained from Refinitiv Eikon Datastream at FEB, KU Leuven. The data itself is not published on this repository.

Running the following script produces the results of portfolio optimization, as well as statistics about POET performance across different factor number estimators:

```Python
portfolio_optimization [1] [2] [3] [4]
```
where 
```
[1]: N - number of stocks (int)
[2]: years - length of data sample (int)
[3]: k_max - upper limit for factor estimation (int)
[4]: k_min - lower limit for factor estimation (int)
```



## References
Ahn, S. C., & Horenstein, A. R. (2013). Eigenvalue ratio test for the number of factors. Econometrica, 81 (3), 1203–1227. https://doi.org/10.3982/ECTA8968

Bai, J., & Ng, S. (2002). Determining the number of factors in approximate factor models. Econometrica, 70 (1), 191–221. https://doi.org/10.1111/1468-0262.00273

Fan, J., Liao, Y., & Mincheva, M. (2013). Large Covariance Estimation by Thresholding Principal Orthogonal Complements. Journal of the Royal Statistical Society Series B: Statistical Methodology, 75 (4), 603–680. https://doi.org/10.1111/rssb.12016

Onatski, A. (2010). Determining the number of factors from empirical distribution of eigenvalues. The Review of Economics and Statistics, 92 (4),
1004–1016. https://doi.org/10.1162/REST_a_00043

Wei, J., & Chen, H. (2020). Determining the number of factors in approximate factor models by twice k-fold cross validation. Economics Letters, 191, 109149. https://doi.org/10.1016/j.econlet.2020.109149



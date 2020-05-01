import pandas as pd
import numpy as np
import scipy.stats

def annualize_rets(r,periods_per_year):
    """
    Computes Annualized set of returns
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def semideviation(r):
    """
    Returns the semideviations aka negative deviation of those returns whic are negativ.
    """
    is_negative = r < 0
    return r[is_negative].std(ddof = 0)

def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns 
    Compute and return a dataframe that contains:
    the wealth index
    the previous peaks
    percentage drawdowns
    """
    wealth_index = 1000*(1 + return_series).cumprod()
    previous_peaks =  wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth" : wealth_index,
        "Peaks" : previous_peaks,
        "Drawdown" : drawdowns
    })

def skewness(r):
    """
    Alternatively use scipy.stats.skew()
    Compute the skewness of the Supplied Series or DataFrame
    Returs a float or a Series
    """
    demeaned_r = r - r.mean()
    # use population std so set ddof = 0
    # ddof = degrees of freedom
    sigma_r = r.std(ddof = 0)
    exp = (demeaned_r**3).mean()
    return exp/(sigma_r**3)


def kurtosis(r):
    """
    Alternatively use scipy.stats.kurtosis()
    Compute the kurtosis of the Supplied Series or DataFrame
    Returs a float or a Series
    """
    demeaned_r = r - r.mean()
    # use population std so set ddof = 0
    # ddof = degrees of freedom
    sigma_r = r.std(ddof = 0)
    exp = (demeaned_r**4).mean()
    return exp/(sigma_r**4)


# Jarque bera test using scipy
def JB_Test(r, level=0.01):
    """
    Applies Jarque Bera Test to determine if the series is normally distributed or not
    Test is applied at level of 1%
    Returns True if hypothesis of normality is accepted
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level

def var_historic(r, level = 5):
    """
    Returns historic value at Risk at a specified level
    ie; returns the number such that "level" percent of the returns fall below that number, and the (100-level) percent are above that.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")

        
# Value at Risk using parametric Gausssian
from scipy.stats import norm
def var_gaussian(r, level=5, modified = False):
    """
    Returns the parametric Gaussina VaR
    if modified is true then modiefied Var as per Cornish-Fisher is returned
    """
    # to norm.ppf gives CDF of standar normal at input value
    z = norm.ppf(level/100)
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z+
             (z**2 - 1)*s/6+
             (z**3 - 3*z)*(k-3)/24-
             (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof = 0))


# CVaR
def cvar(r, level = 5, gaussian = False, modified = False):
    """
    Computes the Conditional VaR of series.
    """
    #r <= -var_historic(r, level = level)
    #gives all returns less than VaR
    #negative sign as i report VaR as positive
    if isinstance(r, pd.Series):
        if gaussian:
            is_beyond = r <= -var_gaussian(r, level = level, modified = modified)
            return -r[is_beyond].mean()
        else:
            is_beyond = r <= -var_historic(r, level = level)
            return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar, level = level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")
        
        
        
        
## Portfolio Optimisation

# Markowitz
def portfolio_return(weights, returns):
    """
    weights --> corresponding returns
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Weights --> Volatility
    """
    return (weights.T @ covmat @ weights)**0.5

# for multiple assets
from scipy.optimize import minimize
# def target_is_met(w, er):
#     return target_return - rk.portfolio_return(w, er)
# lambda function is replacement of call of target_is_met 
# it takes inputs weighst and er and after : comes the value it returns
# and it stops as soon as it return 0
# return_is_target is of type equality constraint 
def minimize_vol(target_return, er, cov):
    """
    gives the weights corresponding to minimum volatility for the target_return. 
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # means every weight in output weight vector is between 0 an 1
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights_sum = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    result = minimize(portfolio_vol, init_guess, args=(cov,), method = 'SLSQP',
                     options = {'disp': False},
                     constraints=(weights_sum, return_is_target), bounds = bounds)
    return result.x


def optimal_weights(n_points, er, cov):
    """
    generates list of weights satisfying optimality condition.
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

# code for Maximum Sharpe ratio portfolio
# we want to maximize the sharpe ratio so we will minimize the negative of sharpe ratio.
from scipy.optimize import minimize
def msr(riskfree_rate, er, cov):
    """
    Returns the maximum sharpe ratio portfolio weights for a given risk free rate, expected return and covariance matrix.
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n 
    weights_sum = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Returns teh negative of the sharpe ratio for the given weights.
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r-riskfree_rate)/vol
    
    result = minimize(neg_sharpe_ratio, init_guess, args=(riskfree_rate, er, cov,), method = 'SLSQP',
                     options = {'disp': False},
                     constraints=(weights_sum), bounds = bounds)
    return result.x


def gmv(cov):
    """
    Returns weights of the Global minimum vol portfolio given covariance matrix.
    """
    # if all return are same then max sharpe ratio is minimizing the volatility
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)

# cml = capital market line
# ew = equally weighted portfolio
def plot_ef(n_points, er, cov, style = ".-", show_cml = False, riskfree_rate = 0, show_ew = False, show_gmv = False):
    """
    Plots the N-asset frontier 
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    ax = ef.plot.line(x = "Volatility", y = "Returns", style = style, figsize = (10,6))
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # display equally weighted portfolio
        ax.plot([vol_ew], [r_ew], color = "goldenrod", marker = "o", markersize = 10)
    
    if show_gmv:
        ax.set_xlim(left = 0)
        w_gmv = gmv(cov)
        r_ew = portfolio_return(w_gmv, er)
        vol_ew = portfolio_vol(w_gmv, cov)
        # display equally weighted portfolio
        ax.plot([vol_ew], [r_ew], color = "midnightblue", marker ="o", markersize = 10)
        
    if show_cml:
        ax.set_xlim(left = 0) # anything that mathplotlib plots in the are is called axes.
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # add capital market line
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color = "red", marker = "o", linestyle = "dashed", markersize = 12, linewidth = 2)
        # here I am saying takethe axes drawn in first step and plt on that
        return ax
    
    
# convinience function
def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher Corrected VaR (2%)": cf_var5,
        "Historic CVaR (2%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })
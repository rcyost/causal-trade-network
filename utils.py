
import numpy as np
from numpy.core.fromnumeric import sort
from numpy import arange

import pandas as pd
from pandas.plotting import lag_plot

from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, legend, title
import matplotlib.ticker as ticker

from pymnet import *

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from scipy import stats
from scipy.signal import detrend

from kneed import KneeLocator

def load_data():
    mcl = pd.read_csv('countryList.csv',
                    dtype={'Name':'str',
                            'ISO':'int',
                            'IC':'str',
                            'Region':'str'})


    imports = pd.read_csv(r'C:\Users\yosty\Desktop\Desktop_Folder\14 - git\trade\wto\ITS_MTV_M\ITS_MTV_MM.csv')
    # change dates to a proper pandas datetime format
    imports['Year'] = pd.to_datetime(imports['Year'].astype('str') + '-' + imports['PeriodCode'].str.strip('M'))


    exports = pd.read_csv(r'C:\Users\yosty\Desktop\Desktop_Folder\14 - git\trade\wto\ITS_MTV_M\ITS_MTV_MX.csv')
    exports['Year'] = pd.to_datetime(exports['Year'].astype('str') + '-' + exports['PeriodCode'].str.strip('M'))

    # create time series datasets
    imports = pd.DataFrame(imports
                .query('PartnerEconomy != ["European Union", "Extra EU Trade"]')
                .query('ReportingEconomy != "European Union"')
                .groupby(['Year', 'ReportingEconomy', 'ProductOrSector', 'PartnerEconomy'])['Value']
                .agg('sum')
    ).reset_index()


    exports = pd.DataFrame(exports
                .query('PartnerEconomy != ["European Union", "Extra EU Trade"]')
                .query('ReportingEconomy != "European Union"')
                .groupby(['Year', 'ReportingEconomy', 'ProductOrSector', 'PartnerEconomy'])['Value']
                .agg('sum')
    ).reset_index()



    return(mcl, imports, exports)

#mcl, exports, imports = load_data()




def wide_data(imports, exports):
    importsWide = (pd.pivot_table(imports,
                            index= 'Year',
                            columns=['ReportingEconomy','ProductOrSector'],
                            values='Value',
                            aggfunc= np.sum)
                    # several series have 2006 missing so remove and dropna, we keep more series
                    .query('Year != [2006]')
                    # the aggfunc sum can create 0's, replace with NaN to drop
                    # future log diff transform with produce errors if there is a 0
                    .replace(0, np.NaN)
                    .dropna(axis=1)
    ).pct_change().iloc[1:]

    # flatten multiindex column names
    importsWide.columns = ['-'.join(col) for col in importsWide.columns]


    exportsWide = (pd.pivot_table(exports,
                            index= 'Year',
                            columns=['ReportingEconomy','ProductOrSector'],
                            values='Value',
                            aggfunc= np.sum)
                    .query('Year != [2006]')
                    .replace(0, np.NaN)
                    .dropna(axis=1)
    ).pct_change().iloc[1:]

    exportsWide.columns = ['-'.join(col) for col in exportsWide.columns]

    importsWide.columns = [col + '-import' for col in importsWide.columns]
    exportsWide.columns = [col + '-export' for col in exportsWide.columns]

    importExports = pd.merge(left=importsWide, right=exportsWide, on='Year')

    importExports.columns = importExports.columns.str.replace('-Total merchandise', '')

    return(importExports)

#importsWide, exportsWide = wide_data(imports, exports)

def printSeries(inputData, ncols, width, length):

    from math import ceil

    nrows = ceil(len(inputData.columns) / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=120, figsize=(width,length))
    for i, ax in enumerate(axes.flatten()):
        if i > len(inputData.columns):
            pass
        else:
            data = inputData[inputData.columns[i]]
            ax.plot(data, color='red', linewidth=1)
            # Decorations
            ax.set_title(inputData.columns[i])
            ax.xaxis.set_ticks_position('none')
            ax.get_xaxis().set_visible(False)
            ax.yaxis.set_ticks_position('none')
            ax.spines["top"].set_alpha(0)
            ax.tick_params(labelsize=6)

    plt.tight_layout()

#printSeries(importsWide, ncols=3, width=10, length=80)


# https://github.com/leosmigel/analyzingalpha/blob/master/2019-10-06-time-series-analysis-with-python/time_series_analysis_with_python.ipynb

# https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/

# https://www.machinelearningplus.com/time-series/kpss-test-for-stationarity/


#  ADF and KPSS -> not stationary
#. !ADF and !KPSS -> stationary
#. !ADF and KPSS -> trend stationary, remove trend
#  ADF and !KPSS -> difference stationary, use differencing to make stationary
def checkADF(data, p_thresh):

    tempStats = []

    for i, (colname, series) in enumerate(data.iteritems()):
        adfP = adfuller(series, autolag = 'AIC', regression = 'ct')[1]
        aPass = adfP < p_thresh

        tempStats.append(pd.DataFrame({
        'country': data.columns[i],
        'seriesLen': len(series),
        'adfPass': aPass,
        'adfPvalue': adfP}, index= [0]))

    return(pd.concat(tempStats))


def diffSeries(data:pd.DataFrame, p_thresh:float, n:int) -> list:
    """[takes in a wide dataset of series. checks for stationarity.
    if any of the series are unstationary,
    a log diff transformation is applied to all, to improve this,
    a series by series check-transform process should be used]

    Args:
        data (pd.DataFrame): [wide df of series, rows are years each col is a series]
        p_thresh (float): [.01, .05, .1]
        n (int): [number of diff's]

    Returns:
        list: [description]
    """
    # check for stationarity
    results = checkADF(data=data, p_thresh=p_thresh)

    # if there a non-stationary time series transform
    if False in results['adfPass']:
        tempData = []
        # transform each series
        for i, (colname, series) in enumerate(data.iteritems()):
            # could write helper functions are try different transform techniques here
            tempData.append(np.log(series).diff(periods=n).dropna())
            #tempData.append(pd.Series(detrend(series)))
        transformedData = pd.concat(tempData, axis = 1)
        adfResults = checkADF(data=transformedData, p_thresh=p_thresh)
        # include the 1 value to indicate a transformation took place
        return([transformedData, adfResults, 1])

    # if all stationary
    else:
        return [data, results, 0]

# # https://stefan-jansen.github.io/machine-learning-for-trading/09_time_series_models/

# if we use pct change series diffed isn't needed
# importStat, importAdfResults, importdiffedCheck = diffSeries(data=importsWide, p_thresh=.1, n=1)
# exportStat, exportAdfResults, exportdiffedCheck = diffSeries(data=exportsWide, p_thresh=.1, n=1)


# printSeries(importsWide, ncols=3, width=10, length=120)
# printSeries(importStat, ncols=3, width=10, length=120)



def printLagPlots(raw:pd.DataFrame, transformed:pd.DataFrame, width:int, length:int):

    nrows = len(raw.columns)
    ncols = 2

    fig, axes = plt.subplots(nrows,ncols, dpi=120, figsize=(width,length))

    for i in range(nrows):
        lag_plot(raw[raw.columns[i]], ax=axes[i, 0]).set(title=f'raw {raw.columns[i]}')
        if raw.columns[i] in transformed.columns:
            lag_plot(transformed[raw.columns[i]], ax=axes[i, 1]).set(title=f'log diff {raw.columns[i]}')

    plt.tight_layout()

# printLagPlots(importsWide, importStat, 10, 200)
# printLagPlots(exportsWide, exportStat, 10, 200)


# https://stats.stackexchange.com/questions/107954/lag-order-for-granger-causality-test
# https://davegiles.blogspot.com/2011/04/testing-for-granger-causality.html
# https://stats.stackexchange.com/questions/24753/interpreting-granger-causality-tests-results

def granger(inputData, maxlags, p_thresh):

    from statsmodels.tsa.stattools import grangercausalitytests
    from itertools import combinations

    # test granger causality of all combinations
    grangerPList = []
    for col in combinations(inputData.columns, 2):
        if col[0] != col[1]:
            tempData = inputData[[col[0], col[1]]]
            results = grangercausalitytests(tempData, maxlag= maxlags, verbose=False)

            stats = {'pValueLag-'+ str(i+1) : results[i+1][0]['ssr_chi2test'][1] for i in range(maxlags)}
            stats['var1'] = tempData.columns[0]
            stats['var2'] = tempData.columns[1]

            grangerPList.append(pd.DataFrame(stats, index=[0]))

    grangerPData = pd.concat(grangerPList)

    grangerPData = (pd.concat(grangerPList)
    .pivot_table(
        columns=['var1', 'var2'],
        values = [col for col in grangerPData.columns if col not in ['var1', 'var2']])
    )

    onlyGrangCause = grangerPData.copy()

    # return copy that only has granger causality
    for col, series in onlyGrangCause.iteritems():
        if (series < p_thresh).all(False):
            onlyGrangCause.drop(col, axis=1, inplace=True)

    return(grangerPData, onlyGrangCause)


#grangerPData = granger(importsWide, exportsWide, 25)

# https://www.quantrocket.com/codeload/pairs-pipeline/pairs_pipeline/Part3-Pairs-Selection-Pipeline.ipynb.html
# https://nbviewer.org/github/mapsa/seminario-doc-2014/blob/master/cointegration-example.ipynb
# https://stats.stackexchange.com/questions/21539/what-is-the-correct-procedure-to-choose-the-lag-when-performing-johansen-cointeg
# https://quantdare.com/cointegration-in-economy/
# https://quant.stackexchange.com/questions/3270/what-are-the-applications-of-cointegration
# https://corporatefinanceinstitute.com/resources/knowledge/other/cointegration/
# https://quant.stackexchange.com/questions/1027/how-are-correlation-and-cointegration-related
# https://quant.stackexchange.com/a/1038


def coint(data, threshhold):

    # unique combination of all series

    tempList = []
    for col in combinations(data.columns, 2):
        if col[0] != col[1]:
            tempData = data[[col[0], col[1]]]

            results = coint_johansen(tempData, -1, 12)
            # threshhold 0-90%, 1-95%, 2-99%
            trace_crit_value = results.cvt[:, threshhold]
            eigen_crit_value = results.cvm[:, threshhold]
            if np.all(results.lr1 >= trace_crit_value) and np.all(results.lr2 >= eigen_crit_value):
                tempList.append(pd.DataFrame({
                'series1' : col[0],
                'series2' : col[1]}, index=[0]))

    return(pd.concat(tempList))

#cointResults = coint(importsWide, exportsWide, 2)

# https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial/#two
# https://hastie.su.domains/TALKS/enet_talk.pdf
# https://machinelearningmastery.com/elastic-net-regression-in-python/
# https://machinelearningmastery.com/elastic-net-regression-in-python/



def elasticSelect(data, onlyGrangeCause, y):

    # y = 'Kazakhstan-Total merchandise-export'

    # get series that pass the previous tests for our series of interest
    X = data[[col[1] for col in onlyGrangeCause.columns if y in col[0]]]
    y_data = data[[y]]

    # define model
    model = ElasticNet()
    # define model evaluation method
    # define grid
    ratios = arange(0, 1, 0.01)
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
    # define search
    # search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', n_jobs=-1)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1)
    # perform the search
    results = model.fit(X, y_data)
    # summarize
    # print('alpha: %f' % model.alpha_)
    # print('l1_ratio_: %f' % model.l1_ratio_)

    resultCoef = [i for i in zip(X.columns, model.coef_)]
    nonZeroCoef = [i for i in resultCoef if i[1] != 0]

    return(resultCoef, nonZeroCoef)

# resultCoef, nonZeroCoef = elasticSelect(data, onlyGrangeCause, 'Kazakhstan-Total merchandise-export')



# https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
# http://www.phdeconomics.sssup.it/documents/Lesson18.pdf
#  linear combination of past values of itself and the past values of other variables in the system
# http://www.princeton.edu/~mwatson/papers/Stock_Watson_HOM_Vol2
# decent reference page for the one below
# http://www.ams.sunysb.edu/~zhu/ams586/VAR_Lecture2.pdf
# https://www.youtube.com/watch?v=TpQtD7ONfxQ&list=PLh8md8UCjx2ZHgbBzgT_6ya3L3Z_BaeuD&index=2


"""
The knee method for finding optimal VAR lag length with AIC didn't work as well as I'd hope... using lag of 12 for now
"""
def varFunc(data, series, lag=12):
    varData = data[series]

    # VAR
    model = VAR(varData)
    # varList = []
    # for i in range(search):
    #     #print(i)
    #     varList.append(pd.DataFrame({
    #         'lag':i,
    #         'aic': model.fit(i).aic,
    #         'bic': model.fit(i).bic}, index=[i]))

    # searchResults = pd.concat(varList)

    # # sensitivity = 1 seems to stop around 3-5 lags
    # # S= 10 seems to get us to the ~12 lags
    # knee = KneeLocator(
    #         y=searchResults['aic'],
    #         x=searchResults.index,
    #         curve="convex",
    #         direction="decreasing",
    #         S=100)


    return(varData, model.fit(lag))

#results, modelData = varFunc(importsWide, exportsWide, 25)



def varResidProbPlot(model,width=5):

    nrows = model.resid.shape[1]
    fig, axes = plt.subplots(nrows, 1, dpi=120, figsize=(width,nrows*2.5))
    for i in range(nrows):
        stats.probplot(x=model.resid.iloc[:, i], dist="norm", plot=axes[i])
        axes[i].set(title=model.resid.columns[i] + ' residuals')

    plt.tight_layout()


def varP(model, y, pThresh):
    # get p values
    # remove const in first row [1:]
    varP = model.pvalues[1:].unstack()
    # filter
    varP = pd.DataFrame(varP[varP < pThresh]).reset_index()
    # clean up and get lag data
    varP[['lag', 'series2']] = varP['level_1'].str.split('.', expand=True)
    varP['lag'] = varP['lag'].str.replace('L', '')
    varP.drop('level_1', axis=1, inplace=True)
    varP.rename({'level_0':'series1', 0:'pValue'},axis=1, inplace=True)
    varP.query('series1 == @y', inplace=True)
    return(varP[['series1', 'series2', 'lag', 'pValue']])


def varCoef(model, y, varPValues):
    # get p values
    # remove const in first row [1:]
    varP = model.params[[y]][1:].unstack().reset_index()
    # clean up and get lag data
    varP[['lag', 'series2']] = varP['level_1'].str.split('.', expand=True)
    varP['lag'] = varP['lag'].str.replace('L', '')
    varP.drop('level_1', axis=1, inplace=True)
    varP.rename({'level_0':'series1', 0:'coeff'},axis=1, inplace=True)

    varPValues['key'] = varPValues['series1']+varPValues['series2']+varPValues['lag']
    varP['key'] = varP['series1']+varP['series2']+varP['lag']
    temp = pd.merge(left=varPValues, right=varP, on='key', how='left', suffixes=('', '_y'))
    temp.drop(temp.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)


    temp['lag'] = pd.to_numeric(temp['lag'])
    temp.sort_values('lag')

    return(temp[['series1', 'series2', 'lag', 'coeff']])




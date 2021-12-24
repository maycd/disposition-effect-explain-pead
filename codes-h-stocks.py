import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import time
from pandas.tseries.offsets import *
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from itertools import product
from statsmodels.stats.outliers_influence import variance_inflation_factor
time_start = time.time()

# Import HSI index data
idx = pd.read_csv('D:/MyDownloads/History/disposition_effect_data/data_input/h_idx_hsi.csv',
                  usecols=['Indexcd', 'Trddt', 'Idxret'],
                  dtype={'Indexcd': 'str', 'Idxret': 'float'})
idx['Trddt'] = pd.to_datetime(idx['Trddt'], format='%d/%m/%Y')
#idx.info()







# Read announcement date file
announce = pd.read_csv('D:/MyDownloads/History/disposition_effect_data/data_input/h_stocks_ad.csv',
                       usecols=['Stkcd', 'Annodt'], dtype={'Stkcd': 'str'})
# Convert stock trading time column to DateTime
announce['Annodt'] = pd.to_datetime(announce['Annodt'], format='%d/%m/%Y')
# Fill missing dates
announce.fillna(method='ffill', inplace=True)
# Convert to the nearest next business day
announce['Annodt_b'] = announce['Annodt'] - BusinessDay() + BusinessDay()
#announce.info()

# Read price volume file for Jinyu Group
data = pd.read_csv('D:/MyDownloads/History/disposition_effect_data/data_input/h_stocks_daily.csv',
                   usecols=['Stkcd', 'Trddt', 'Clsprc', 'Dnshrtrd', 'Dnvaltrd', 'Dsmvosd'],
                   dtype={'Stkcd': 'str', 'Clsprc': 'float', 'Dnshrtrd': 'float', 'Dsmvosd': 'float'})
# Convert stock trading time column to DateTime
data['Trddt'] = pd.to_datetime(data['Trddt'], format='%d/%m/%Y')
# Assert that conversion happened
assert data['Trddt'].dtype == 'datetime64[ns]'
# Check missing values (no missing values)
print(data.isna().sum())

# Import cross-listed company data
crosslist = pd.read_csv('D:/MyDownloads/History/disposition_effect_data/data_input/crosslist.csv',
                        usecols=['Conme_en', 'Stkcd_a', 'Stkcd_h'], dtype='str')
# Find duplicated values (no duplicates)
print(data[data.duplicated()])
# Remove irrelevant stocks from data
data = data[data['Stkcd'].isin(crosslist['Stkcd_h'])]
#print('data:\n', data.head(3))

# Calculate number of outstanding shares
data['shares'] = data.Dsmvosd.mul(1000)/data.Clsprc
# Calculate daily turnover
data['turnover'] = data['Dnshrtrd']/data['shares']

# Create empty DataFrame
dfmdl = pd.DataFrame(columns=['Stock code', 'SSR orig', 'AIC orig',
                              'Alpha orig', 'Beta orig',
                              'SSR', 'AIC', 'R-squared', 'Adj R-squared',
                              'Alpha', 'Beta', 'Beta t', 'Gamma', 'Gamma t',
                              'Pearson', 'Spearman', 'Kendall'])
dfcgo = pd.DataFrame(columns=['cgo'])
dfretabs = pd.DataFrame(columns=['ret_abs'])
dfar = pd.DataFrame(columns=['ar'])
dfcar = pd.DataFrame(columns=['car'])

# List unique stock codes
codeslist = data['Stkcd'].unique()

for ind, code in enumerate(codeslist):
    stock = data[data['Stkcd'].isin([code])].copy()
    # Calculate daily return
    stock['ret'] = stock['Clsprc'].pct_change()







    # Calculate abnormal return
    idx = pd.merge(stock[['Trddt', 'ret']], idx[['Trddt', 'Idxret']], how='left', on='Trddt')
    idx['ar'] = idx['ret'] - idx['Idxret']
    dfar = pd.concat([dfar, idx[['ar']]], ignore_index=True)

    # Calculate cumulative abnormal return over 30 business days
    idx['car'] = idx['ar'].rolling(window=30).sum()
    dfcar = pd.concat([dfcar, idx[['car']]], ignore_index=True)

    # Initialize reference price
    rp_list = [0]
    rp_i = 0
    # Calculate reference prices
    for i in range(len(stock)-1):
        rp_i = stock.iloc[i, 7] * stock.iloc[i, 2] + \
               (1 - stock.iloc[i, 7]) * rp_i
        rp_list.append(rp_i)
    # Attach to the DataFrame
    stock.loc[:, 'rp'] = rp_list

    # Calculate capital gain overhang
    stock['cgo'] = (stock['Clsprc'].shift(1) - rp_list) / stock['Clsprc'].shift(1)
    # Fill DataFrame with new rows for trading days
    dfcgo = pd.concat([dfcgo, stock[['cgo']]], ignore_index=True)

    # Calculate absolute return
    stock.loc[:, 'ret_abs'] = abs(stock['ret'])
    # Fill DataFrame with new rows for trading days
    dfretabs = pd.concat([dfretabs, stock[['ret_abs']]], ignore_index=True)

    # Drop the first row for analysis
    stock.dropna(axis=0, how='any', inplace=True)

    # Correlation coefficients
    corr_p = stock['ret_abs'].corr(stock['cgo'], method='pearson')
    corr_s = stock['ret_abs'].corr(stock['cgo'], method='spearman')
    corr_k = stock['ret_abs'].corr(stock['cgo'], method='kendall')

    # Linear regression on V vs |r|
    X = sm.add_constant(stock['ret_abs'])
    y = stock['turnover']
    reso = sm.OLS(y, X).fit()
    ssro = reso.ssr
    aico = reso.aic
    alphao, betao = reso.params

    # Linear regression on V vs |r|, CGO
    X = sm.add_constant(stock[['ret_abs', 'cgo']])
    resr = sm.OLS(y, X).fit()
    alphar, betar, gamma = resr.params
    b_t = resr.tvalues['ret_abs']
    g_t = resr.tvalues['cgo']
    g_p = resr.pvalues['cgo']

    # Fill Dataframe with a new row for a stock
    dfmdl = dfmdl.append({'Stock code': code, 'SSR orig': ssro, 'AIC orig': aico,
                          'Alpha orig': alphao, 'Beta orig': betao,
                          'SSR': resr.ssr, 'AIC': resr.aic,
                          'Alpha': alphar, 'Beta': betar, 'Beta t': b_t,
                          'Gamma': gamma, 'Gamma t': g_t, 'Gamma p': g_p,
                          'R-squared': resr.rsquared, 'Adj R-squared': resr.rsquared_adj,
                          'Pearson': corr_p, 'Spearman': corr_s, 'Kendall': corr_k}, ignore_index=True)

# Fill zeros in string
dfmdl['Stock code'] = dfmdl['Stock code'].astype(str).str.zfill(5)
dfmdl.to_excel('D:/MyDownloads/History/disposition_effect_data/data_output/h_dispmdl.xlsx', index=False)

# Concatenate data with absolute return, CAR and CGO
data = pd.concat([data, dfretabs, dfar, dfcar, dfcgo], axis=1)
#data.info()

# Drop initial values
data_nona = data[['Stkcd', 'Trddt', 'ret_abs', 'turnover', 'cgo', 'ar', 'car']].dropna(axis=0, how='any')
disp = data_nona[['Stkcd', 'ret_abs', 'turnover', 'cgo']]
# Set stock code as index for disp
disp.set_index('Stkcd', inplace=True)
# Apply min, max, mean, sd, skewness, kurtosis, adfuller test
disp_1171 = disp.loc['1171'].agg(['min', 'max', np.median, np.mean, np.std, 'skew', 'kurt', adfuller])
# Reset index
disp.reset_index('Stkcd', inplace=True)
# Export to excel
disp_1171.to_excel('D:/MyDownloads/History/disposition_effect_data/data_output/disp_1171.xlsx',
                   sheet_name='disposition statistics')
print('data_nona:\n', data_nona)
# Left join announce with data_nona
data_ad = pd.merge(announce, data_nona, how='left', left_on=['Stkcd', 'Annodt_b'], right_on=['Stkcd', 'Trddt'])
print('data_ad:\n', data_ad)
# Import eps data
eps = pd.read_csv('D:/MyDownloads/History/disposition_effect_data/data_input/h_stocks_eps.csv',
                  usecols=['Stkcd', 'StateTypeCode', 'Accper', 'eps'], dtype={'Stkcd': 'str', 'eps': 'float'})
eps = eps[eps['StateTypeCode'] == 'A']
eps.pop('StateTypeCode')
eps['Accper'] = pd.to_datetime(eps['Accper'], format='%d/%m/%Y')
# Match to find total share capital
sc = pd.read_csv('D:/MyDownloads/History/disposition_effect_data/data_input/a_stocks_eps_sc.csv',
                 usecols=['Stkcd', 'Accper', 'ttl_shr_cap'], dtype={'Stkcd': 'str', 'ttl_shr_cap': 'float'})
sc['Accper'] = pd.to_datetime(sc['Accper'], format='%d/%m/%Y')
sc = pd.merge(sc, crosslist, how='left', left_on='Stkcd', right_on='Stkcd_a')
sc.pop('Stkcd')
sc.rename(columns={'Accper': 'Accper_sc'}, inplace=True)
#print('sc:\n', sc)
eps = pd.merge(eps, sc, how='left', left_on=['Stkcd', 'Accper'], right_on=['Stkcd_h', 'Accper_sc'])
eps.pop('Stkcd_h')
eps.pop('Accper_sc')
#print('eps:\n', eps)

# Map to find month
eps['mth'] = eps['Accper'].map(lambda x: x.month)
eps_b = pd.DataFrame()



for ind, code in enumerate(eps['Stkcd'].unique()):
    stockeps = eps[eps['Stkcd'] == code].copy()

    # Relative total share capital
    stockeps['rel_sc'] = stockeps['ttl_shr_cap'] / stockeps['ttl_shr_cap'].shift()
    # Calculate seasonal earnings per share for December
    stockeps['eps_s'] = stockeps['eps'] - stockeps['eps'].shift()
    # Calculate adjusted seasonal earnings per share for June
    for eps_idx in stockeps[stockeps['mth'] == 6].index.tolist():
        stockeps.loc[eps_idx, 'eps_s'] = stockeps.loc[eps_idx, 'eps'] / stockeps.loc[eps_idx, 'rel_sc']

    # Calculate unexpected earnings method 1
    stockeps['ue1'] = stockeps['eps_s'] - stockeps['eps_s'].shift(2)
    # Calculate unexpected earnings method 2
    stockeps['ue2'] = stockeps['ue1'] / abs(stockeps['eps_s'].shift(2))
    # Calculate standardized unexpected earnings method 1
    stockeps['ue1_std'] = stockeps['ue1'].rolling(window=5).std()
    stockeps['sue_1'] = stockeps['ue1'] / stockeps['ue1_std']
    # Calculate standardized unexpected earnings method 2
    stockeps['ue2_std'] = stockeps['ue2'].rolling(window=5).std()
    stockeps['sue_2'] = stockeps['ue2'] / stockeps['ue2_std']

    # Forward fill eps as frequency of business days
    stockeps.set_index('Accper', inplace=True)
    stockeps = stockeps.asfreq('B').ffill()
    stockeps.reset_index('Accper', inplace=True)
    eps_b = pd.concat([eps_b, stockeps], ignore_index=True)
    #print('stockeps:\n', stockeps)

# Drop missing values
eps_b.dropna(axis=0, how='any', inplace=True)
data_ad.dropna(axis=0, how='any', inplace=True)

# Left join data_nona with eps to match nearest previous Accper for ad
data_ad = pd.merge(data_ad, eps_b[['Stkcd', 'Accper', 'sue_1', 'sue_2']], how='left',
                   left_on=['Stkcd', 'Annodt_b'], right_on=['Stkcd', 'Accper'])

# Show whether SUE1 is positive
data_ad['sue_po'] = np.sign(data_ad['sue_1'])
# Create an interaction variable
data_ad['interact'] = (np.sign(data_ad.cgo) + np.sign(data_ad.sue_1)) / 2 \
                      * abs(data_ad.cgo * data_ad.sue_1)
#print('data_ad:\n', data_ad)

data_ad.pop('Trddt')
data_ad.pop('Accper')
# Export data to excel
data_ad.to_csv('D:/MyDownloads/History/disposition_effect_data/data_output/h_data_ad.csv',
               index=False, date_format='%d/%m/%Y')
data_ad.describe().to_excel('D:/MyDownloads/History/disposition_effect_data/data_output/h_data_ad_describe.xlsx')
data_ad.dropna(how='any', inplace=True)

# Pearson correlation matrix and heat map
corr_p = data_ad[['cgo', 'car', 'sue_1', 'turnover']].corr(method='pearson')
sns.heatmap(corr_p, annot=True)
plt.savefig('D:/MyDownloads/History/disposition_effect_data/data_output/h_heatmap.png', dpi=300)
plt.show()
plt.clf()

# Pair plot
sns.pairplot(data_ad, vars=['cgo', 'sue_1', 'car', 'turnover'], hue='Stkcd', palette='husl')
plt.savefig('D:/MyDownloads/History/disposition_effect_data/data_output/h_pairplot.png', dpi=300)
plt.show()
plt.clf()


def cal_vif(df):
    vif = pd.DataFrame()
    vif['variables'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif


# Calculate VIF for testing multicollinearity
cal_vif(data_ad[['cgo', 'sue_1', 'car']]).\
    to_excel('D:/MyDownloads/History/disposition_effect_data/data_output/h_vif.xlsx')

# Select positive cgo and positive sue
data_ad = data_ad[(data_ad.cgo > 0) & (data_ad.sue_1 > 0)]

# Fit OLS models
repre_list = ['car ~ cgo', 'car ~ sue_1', 'car ~ sue_2',
              'turnover ~ cgo * sue_1', 'turnover ~ cgo * sue_2',
              'turnover ~ cgo + sue_1 + cgo:sue_1 + car', 'turnover ~ cgo + sue_2 + cgo:sue_2 + car']
resdf = pd.DataFrame()

for repre in repre_list:
    res_repre = ols(repre, data=data_ad).fit()
    mdl_names = pd.DataFrame(np.repeat(repre, len(res_repre.params) - 1))
    mdl_names.reset_index(drop=True, inplace=True)
    resdf_repre = pd.concat([res_repre.params[1:],
                             res_repre.tvalues[1:], res_repre.pvalues[1:]], axis=1)
    resdf_repre.reset_index(drop=True, inplace=True)
    resdf_repre = pd.concat([mdl_names, resdf_repre], axis=1)
    resdf = pd.concat([resdf, resdf_repre], ignore_index=True)

resdf.columns = ['Model', 'Coefficient', 't-value', 'p-value']
resdf.to_excel('D:/MyDownloads/History/disposition_effect_data/data_output/h_resdf.xlsx')
res5 = ols('turnover ~ cgo * sue_2', data=data_ad).fit()
print('res5.params:\n', res5.params)

# Create scatter plot for cgo, car, & sue_1
sns.scatterplot(x='cgo', y='car', data=data_ad, hue='sue_1')
# Create combination of values of arrays of numbers
xmin, xmax = plt.xlim()
p = product(np.arange(xmin, xmax, 0.05), np.arange(-1.5, 4.6, 0.5))
# Transform to DataFrame
explanatory_data = pd.DataFrame(p, columns=['cgo', 'sue_1'])
# Fit OLS model with interaction
res3_1 = ols('car ~ cgo * sue_1', data=data_ad).fit()
# Add column of predictions
prediction_data = explanatory_data.assign(car=res3_1.predict(explanatory_data))
sns.scatterplot(x='cgo', y='car', data=prediction_data,
                hue='sue_1', marker='s', legend=False)
plt.title('Multiple linear regression on CAR vs CGO and SUE with interaction effect')
plt.savefig('D:/MyDownloads/History/disposition_effect_data/data_output/h_car_vs_cgo_sue_1.png', dpi=300)
plt.show()
print('time cost: ', time.time() - time_start, 's')

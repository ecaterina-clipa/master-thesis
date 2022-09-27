#!/usr/bin/env python
# coding: utf-8

# # Zero Curve Benchmark for Temperature | Emissions

# In[57]:


import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import tikzplotlib
import pylab
pylab.rcParams['figure.figsize'] = [14, 10]


# In[2]:


from QuantLib import *


# In[3]:


#We set the evaluation date
today = Date(8, 7, 2022)
Settings.instance().evaluationDate = today


# In[4]:


#import data for inputs in the yield curves and scenarios' variations (personal path, to be changed)
df =  pd.read_excel(r'C:\Users\Cathy\Documents\DOC\QF-UniBo\TESI\3_Data\Data_to_use.xlsx', sheet_name='ESTR_ON')
df6m = pd.read_excel (r'C:\Users\Cathy\Documents\DOC\QF-UniBo\TESI\3_Data\Data_to_use.xlsx', sheet_name='EURIBOR6M')
df_scenarios = pd.read_excel (r'C:\Users\Cathy\Documents\DOC\QF-UniBo\TESI\3_Data\Data_to_use.xlsx', sheet_name='Em_scenarios')
df_tscenarios = pd.read_excel (r'C:\Users\Cathy\Documents\DOC\QF-UniBo\TESI\3_Data\Data_to_use.xlsx', sheet_name='Temp_scenarios')


# ## €STR DISCOUNTING CURVE

# #Construction of the OIS discounting curve

# In[5]:


estr = Eonia()
#Deposits
helpers = [ DepositRateHelper(QuoteHandle(SimpleQuote(rate/100)),
Period(1,Days), fixingDays, TARGET(), Following, False, Actual360())
for rate, fixingDays in [(df.RATE[0], 0), (df.RATE[1], 1), (df.RATE[2], 2)] ]
#OIS rates
helpers += [ OISRateHelper(2, Period(*tenor), QuoteHandle(SimpleQuote(rate/100)), estr)
for rate, tenor in [(df.RATE[3], (1,Weeks)), (df.RATE[4], (2,Weeks))] ]
#ECB DATES
helpers += [ DatedOISRateHelper(start_date, end_date, QuoteHandle(SimpleQuote(rate/100)), estr)
for rate, start_date, end_date in [ 
(-0.289, Date(28,July,2022), Date(15,September,2022)),
(0.229, Date(15,September,2022), Date(3,November,2022)),
(0.625, Date(3,November,2022), Date(22,December,2022)),
(0.9, Date(22,December,2022), Date(9,February,2023))] ]
#OIS rates
helpers += [ OISRateHelper(2, Period(tenor),
QuoteHandle(SimpleQuote(rate/100)), estr)
for rate, tenor in zip(df.RATE[9:].to_numpy().tolist(), df.TENOR[9:].to_numpy().tolist()) ] 
#Construction of the curve
estr_curve_c = PiecewiseLogCubicDiscount(0, TARGET(), helpers, Actual365Fixed())
estr_curve_c.enableExtrapolation()
discount_estr_curve = RelinkableYieldTermStructureHandle()
discount_estr_curve.linkTo(estr_curve_c)


# In[6]:


#Create a semiannual schedule for the floating leg of IRS rates
begin = Date(12,7,2022) #starting from the spot date
end = Date(12,7,2082)
tenor = Period(Semiannual)
cal = TARGET()
busconv= ModifiedFollowing
busend = ModifiedFollowing
rule = DateGeneration.Forward
endofmonth = False
schedule6 = Schedule(begin, end, tenor, cal, busconv, busend, rule, endofmonth)

semiannual = []
for i, d in enumerate(schedule6):
    semiannual.append(d)


# In[7]:


#Year fraction of the floating leg of IRS rates tauL
tauL = [Actual360().yearFraction(semiannual[i-1],semiannual[i]) for i in range(1,len(semiannual)) ]


# In[8]:


#Discount factors every 6 months for the IRS rates
Pc6 = []
for i in semiannual:
    Pc6.append(estr_curve_c.discount(i))
Pc_spot = Pc6[0] #Discount factor at spot date
Pc6 = Pc6[1:]


# ## Zero Curve 6 months

# In[ ]:


#Construction of the zero curve with 6 months underlying


# In[9]:


df6m


# In[10]:


euribor6m = Euribor6M()
#Deposit
helpers6m = [ DepositRateHelper(QuoteHandle(SimpleQuote(df6m.RATE[0]/100)), Period(6,Months), 2,
TARGET(), ModifiedFollowing, False, Actual360()) ]
#FRA rates
starts = [1, 2, 3, 4, 5, 6, 12]
helpers6m += [ FraRateHelper(QuoteHandle(SimpleQuote(rate/100)), start, euribor6m)
for rate, start in zip(df6m.RATE[1:8],starts) ]
#IRS rates
tenors = [x for x in range(2,21)] + [25,30,35,40,50,60]
helpers6m += [ SwapRateHelper(QuoteHandle(SimpleQuote(rate/100)),
Period(tenor, Years), TARGET(), Annual, Unadjusted,
Thirty360(Thirty360.BondBasis), euribor6m, QuoteHandle(), Period(0, Days),
discount_estr_curve)
for rate, tenor in zip(df6m.RATE[8:],tenors)]
#Curve construction
euribor6m_curve = PiecewiseLogCubicDiscount(2, TARGET(), helpers6m, Actual365Fixed())
euribor6m_curve.enableExtrapolation()


# In[11]:


#to see start and end date
i= 0
for h in helpers6m:
    df6m.at[i, 'START_DATE'] = h.earliestDate()
    df6m.at[i, 'PILLAR_DATE'] = h.pillarDate()
    i += 1
#with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
df6m


# In[12]:


#Create annual schedule for the fixed leg of IRS, always from spot
begin = Date(12,7,2022)
end = Date(12,7,2082)
tenor = Period(Annual)
cal = TARGET()
busconv= ModifiedFollowing
busend = ModifiedFollowing
rule = DateGeneration.Forward
endofmonth = False
schedule = Schedule(begin, end, tenor, cal, busconv, busend, rule, endofmonth)

annual = []
for d in schedule:
    annual.append(d)


# In[13]:


#Semiannual ZCB 6m
P6m_semiannual = []
for i in semiannual:
    P6m_semiannual.append(euribor6m_curve.discount(i))
P6m_semiannual = P6m_semiannual[1:]


# In[14]:


#Annual schedule corresponding to IRS quotes
t2 = [euribor6m_curve.nodes()[i][0] for i in range(1,len(euribor6m_curve.nodes())) ]
t2 = [t2[6]] + t2[8:]


# In[15]:


#Annual year fractions for IRS quotes
tau = [Thirty360(Thirty360.BondBasis).yearFraction(annual[i-1],annual[i]) for i in range(1,len(annual)) ]
tauK = tau[:20] + [tau[25]] + [tau[29]] + [tau[34]] + [tau[39]] + [tau[49]] + [tau[59]] #schedule of IRS


# In[16]:


#Pc annual for Ac
Pc2 = []
for i in annual:
    Pc2.append(estr_curve_c.discount(i))
Pc2 = Pc2[1:]


# In[17]:


#AcK annual
Ac = []
summation = 0
for i in range(len(Pc2)):
    summation += Pc2[i]*tau[i]
    Ac.append(summation)
Ac = Ac[:20] + [Ac[25]] + [Ac[29]] + [Ac[34]] + [Ac[39]] + [Ac[49]] + [Ac[59]]


# In[18]:


#DataFrame for the floating leg of IRS
df6m_floating = pd.DataFrame()
df6m_floating['Time'] = np.arange(0.5, 60.5, 0.5).tolist()
df6m_floating['Maturity'] = [d.to_date() for d in semiannual[1:]]
df6m_floating['Pc'] = Pc6
df6m_floating['P6m'] = P6m_semiannual
df6m_floating['Fwd'] = [euribor6m_curve.forwardRate(d, euribor6m.maturityDate(d), Actual360(), Simple).rate() for d in semiannual[:-1]]
df6m_floating['TauL'] = tauL


# In[19]:


#DataFrame for the fixed leg of IRS
df6m_fix = pd.DataFrame()
df6m_fix['Maturity'] = [d.to_date() for d in t2]
df6m_fix['IRS rate'] = [df6m.RATE[6]] + [df6m.RATE[i] for i in range(8,33)]
df6m_fix['TauK'] = tauK
df6m_fix['AcK'] = Ac


# In[20]:


df6m_complete = pd.merge(df6m_floating, df6m_fix, how = 'outer', left_on = 'Maturity', right_on = 'Maturity')
df6m_complete


# In[21]:


#Creates an excel file with all the data to be able to perfom a sort of reverse bootstrapping
#df6m_complete.to_excel('em_temp_benchmark.xlsm')


# In[22]:


spot = euribor6m_curve.referenceDate()


# ## Shifts for the zero rates for emissions

# #These shifts must be used in the excel file built above

# In[23]:


zero_rate20y = euribor6m_curve.zeroRate(Actual365Fixed().yearFraction(spot,  Date(14,7,2042)), 
                                           Continuous, Annual).rate()
zero_rate40y = euribor6m_curve.zeroRate(Actual365Fixed().yearFraction(spot, Date(12,7,2062)), 
                                           Continuous, Annual).rate()
zero_rate60y = euribor6m_curve.zeroRate(Actual365Fixed().yearFraction(spot, Date(13,7,2082)), 
                                           Continuous, Annual).rate()
print(zero_rate20y, zero_rate40y, zero_rate60y)


# In[24]:


#Correlations between historical rates and European daily emissions - from Correlation.xlsx, Sheet CO2_Corr
corr20 = 0.524685407
corr40 = 0.53415279
corr60 = 0.533030985


# In[25]:


#Function for calculating the correspondent ZCB distorted
def delta_P(zr, scenario, corr, spot, maturity):
    delta_zr = zr + scenario*corr*zr
    deltaP = np.exp(-delta_zr*Actual365Fixed().yearFraction(spot, maturity))
    return delta_zr, deltaP   #returns the value of the biased zr and biased ZCB 


# In[26]:


#SCENARIO SSP1 - 1.9
SSP119_20 = delta_P(zero_rate20y, df_scenarios.iloc[0,1], corr20, spot, Date(14,7,2042))
SSP119_40 = delta_P(zero_rate40y, df_scenarios.iloc[1,1], corr40, spot, Date(12,7,2062))
SSP119_60 = delta_P(zero_rate60y, df_scenarios.iloc[2,1], corr60, spot, Date(13,7,2082))
print('SSP119_20 : ', SSP119_20)
print('SSP119_40 : ', SSP119_40)
print('SSP119_60 : ', SSP119_60)


# In[27]:


#SCENARIO SSP1 - 2.6
SSP126_20 = delta_P(zero_rate20y, df_scenarios.iloc[0,2], corr20, spot, Date(14,7,2042))
SSP126_40 = delta_P(zero_rate40y, df_scenarios.iloc[1,2], corr40, spot, Date(12,7,2062))
SSP126_60 = delta_P(zero_rate60y, df_scenarios.iloc[2,2], corr60, spot, Date(13,7,2082))
print('SSP126_20 : ', SSP126_20)
print('SSP126_40 : ', SSP126_40)
print('SSP126_60 : ', SSP126_60)


# In[28]:


#SCENARIO SSP2 - 4.5
SSP245_20 = delta_P(zero_rate20y, df_scenarios.iloc[0,3], corr20, spot, Date(14,7,2042))
SSP245_40 = delta_P(zero_rate40y, df_scenarios.iloc[1,3], corr40, spot, Date(12,7,2062))
SSP245_60 = delta_P(zero_rate60y, df_scenarios.iloc[2,3], corr60, spot, Date(13,7,2082))
print('SSP245_20 : ', SSP245_20)
print('SSP245_40 : ', SSP245_40)
print('SSP245_60 : ', SSP245_60)


# In[29]:


#SCENARIO SSP3 -7.0
SSP370_20 = delta_P(zero_rate20y, df_scenarios.iloc[0,4], corr20, spot, Date(14,7,2042))
SSP370_40 = delta_P(zero_rate40y, df_scenarios.iloc[1,4], corr40, spot, Date(12,7,2062))
SSP370_60 = delta_P(zero_rate60y, df_scenarios.iloc[2,4], corr60, spot, Date(13,7,2082))
print('SSP370_20 : ', SSP370_20)
print('SSP370_40 : ', SSP370_40)
print('SSP370_60 : ', SSP370_60)


# In[30]:


#SCENARIO SSP5 - 8.5
SSP585_20 = delta_P(zero_rate20y, df_scenarios.iloc[0,5], corr20, spot, Date(14,7,2042))
SSP585_40 = delta_P(zero_rate40y, df_scenarios.iloc[1,5], corr40, spot, Date(12,7,2062))
SSP585_60 = delta_P(zero_rate60y, df_scenarios.iloc[2,5], corr60, spot, Date(13,7,2082))
print('SSP585_20 : ', SSP585_20)
print('SSP585_40 : ', SSP585_40)
print('SSP585_60 : ', SSP585_60)


# # Interpolating delta zero rates for emissions

# In[ ]:


#Upload the file with the computed zero coupon bonds after the introduction of the shifts. We now construct the discouting curve and the relative zero coupon curve through interploation using QuantLib


# In[31]:


df_Pmodified = pd.read_excel(r'C:\Users\Cathy\Documents\DOC\QF-UniBo\TESI\3_Data\em_benchmark.xlsm', sheet_name='Scenarios')


# In[32]:


#IRS schedule
dates = [Date(12,7,2022)] + semiannual[1:5] + annual[3:21] + [annual[25]] + [annual[30]] + [annual[35]] + [annual[40]] + [annual[50]] + [annual[60]]
#Construct the delta curve of P6m object
ORIGINAL = [df_Pmodified.Soriginal[i] for i in range(1,len(df_Pmodified))]
PSSP119 = [df_Pmodified.SSP119[i] for i in range(1,len(df_Pmodified))]
PSSP126 = [df_Pmodified.SSP126[i] for i in range(1,len(df_Pmodified))]
PSSP245 = [df_Pmodified.SSP245[i] for i in range(1,len(df_Pmodified))]
PSSP370 = [df_Pmodified.SSP370[i] for i in range(1,len(df_Pmodified))]
PSSP585 = [df_Pmodified.SSP585[i] for i in range(1,len(df_Pmodified))]
DORIGINAL = DiscountCurve(dates,ORIGINAL, Actual365Fixed())
delta_zcbSSP119= DiscountCurve(dates, PSSP119, Actual365Fixed())
delta_zcbSSP126= DiscountCurve(dates, PSSP126, Actual365Fixed())
delta_zcbSSP245= DiscountCurve(dates, PSSP245, Actual365Fixed())
delta_zcbSSP370= DiscountCurve(dates, PSSP370, Actual365Fixed())
delta_zcbSSP585= DiscountCurve(dates, PSSP585, Actual365Fixed())


# In[33]:


delta_zero_rates = [DORIGINAL.zeroRate(Actual365Fixed().yearFraction(spot, d), Continuous, Annual).rate()
             for d in dates]
delta_zero_ratesSSP119 = [delta_zcbSSP119.zeroRate(Actual365Fixed().yearFraction(spot, d), Continuous, Annual).rate()
             for d in dates]
delta_zero_ratesSSP126 = [delta_zcbSSP126.zeroRate(Actual365Fixed().yearFraction(spot, d), Continuous, Annual).rate()
             for d in dates]
delta_zero_ratesSSP245 = [delta_zcbSSP245.zeroRate(Actual365Fixed().yearFraction(spot, d), Continuous, Annual).rate()
             for d in dates]
delta_zero_ratesSSP370 = [delta_zcbSSP370.zeroRate(Actual365Fixed().yearFraction(spot, d), Continuous, Annual).rate()
             for d in dates]
delta_zero_ratesSSP585 = [delta_zcbSSP585.zeroRate(Actual365Fixed().yearFraction(spot, d), Continuous, Annual).rate()
             for d in dates]


# In[34]:


#Construct Zero Curve object
all_original = MonotonicCubicZeroCurve(dates, delta_zero_rates, Actual365Fixed())
all_zrSSP119 = MonotonicCubicZeroCurve(dates, delta_zero_ratesSSP119, Actual365Fixed())
all_zrSSP126 = MonotonicCubicZeroCurve(dates, delta_zero_ratesSSP126, Actual365Fixed())
all_zrSSP245 = MonotonicCubicZeroCurve(dates, delta_zero_ratesSSP245, Actual365Fixed())
all_zrSSP370 = MonotonicCubicZeroCurve(dates, delta_zero_ratesSSP370, Actual365Fixed())
all_zrSSP585 = MonotonicCubicZeroCurve(dates, delta_zero_ratesSSP585, Actual365Fixed())


# In[35]:


dates = [ spot+Period(i,Days) for i in range(0,21916+1) ]
#dates = [ spot+Period(i,Months) for i in range(0,12*60+1) ]
#dates = [ spot+Period(i,Years) for i in range(0,60+1,2) ]
#dates = [Date(12,7,2022)] + semiannual[1:5] + annual[3:21] + [annual[25]] + [annual[30]] + [annual[35]] + [annual[40]] + [annual[50]] + [annual[60]]
zrSSP119 = [all_zrSSP119.zeroRate(Actual365Fixed().yearFraction(spot, d), Continuous, Annual).rate()
             for d in dates]
zrSSP126 = [all_zrSSP126.zeroRate(Actual365Fixed().yearFraction(spot, d), Continuous, Annual).rate()
             for d in dates]
zrSSP245 = [all_zrSSP245.zeroRate(Actual365Fixed().yearFraction(spot, d), Continuous, Annual).rate()
             for d in dates]
zrSSP370 = [all_zrSSP370.zeroRate(Actual365Fixed().yearFraction(spot, d), Continuous, Annual).rate()
              for d in dates]
zrSSP585 = [all_zrSSP585.zeroRate(Actual365Fixed().yearFraction(spot, d), Continuous, Annual).rate()
              for d in dates]
zero_rate = [all_original.zeroRate(Actual365Fixed().yearFraction(spot, d), Continuous, Annual).rate()
             for d in dates ]
dates1 = [d.to_date() for d in dates]
fig, ax = plt.subplots() 
ax.plot(dates1,zero_rate, label = 'Original')
ax.plot(dates1,zrSSP119,'--', label = 'SSP119')
ax.plot(dates1,zrSSP126 ,'--', label = 'SSP126')
ax.plot(dates1,zrSSP245,'--', label = 'SSP245')
ax.plot(dates1,zrSSP370,'--', label = 'SSP370')
ax.plot(dates1,zrSSP585,'--', label = 'SSP585')
ax.legend()
ax.set_title('Emissions scenarios')


# # Shifts for zero rates for temperature

# In[ ]:


#We recover the shifts in the emissions benchmark curves for the temperature


# In[36]:


#Correlations between historical rates and European daily mean average - from Correlation.xlsx, Sheet TAVG_Corr
tcorr20 = -0.068404104
tcorr40 = -0.067624392
tcorr60 = -0.066291365


# In[37]:


#We recover the zero rates that we wish to bias in SSP119
zrSSP119_20 = all_zrSSP119.zeroRate(Actual365Fixed().yearFraction(spot, Date(14,7,2042)), 
                                           Continuous, Annual).rate()
zrSSP119_40 = all_zrSSP119.zeroRate(Actual365Fixed().yearFraction(spot, Date(12,7,2062)), 
                                           Continuous, Annual).rate()
zrSSP119_60 = all_zrSSP119.zeroRate(Actual365Fixed().yearFraction(spot, Date(13,7,2082)), 
                                           Continuous, Annual).rate()
print(zrSSP119_20, zrSSP119_40, zrSSP119_60)


# In[38]:


#SCENARIO SSP1 - 1.9
tSSP119_20 = delta_P(zrSSP119_20, df_tscenarios.iloc[0,1], tcorr20, spot, Date(14,7,2042))
tSSP119_40 = delta_P(zrSSP119_40, df_tscenarios.iloc[1,1], tcorr40, spot, Date(12,7,2062))
tSSP119_60 = delta_P(zrSSP119_60, df_tscenarios.iloc[2,1], tcorr60, spot, Date(13,7,2082))
print('tSSP119_20 : ', tSSP119_20)
print('tSSP119_40 : ', tSSP119_40)
print('tSSP119_60 : ', tSSP119_60)


# In[39]:


#We recover the zero rates that we wish to bias in SSP126
zrSSP126_20 = all_zrSSP126.zeroRate(Actual365Fixed().yearFraction(spot, Date(14,7,2042)), 
                                           Continuous, Annual).rate()
zrSSP126_40 = all_zrSSP126.zeroRate(Actual365Fixed().yearFraction(spot, Date(12,7,2062)), 
                                           Continuous, Annual).rate()
zrSSP126_60 = all_zrSSP126.zeroRate(Actual365Fixed().yearFraction(spot, Date(13,7,2082)), 
                                           Continuous, Annual).rate()
print(zrSSP126_20, zrSSP126_40, zrSSP126_60)


# In[40]:


#SCENARIO SSP1 - 2.6
tSSP126_20 = delta_P(zrSSP126_20, df_tscenarios.iloc[0,2], tcorr20, spot, Date(14,7,2042))
tSSP126_40 = delta_P(zrSSP126_40, df_tscenarios.iloc[1,2], tcorr40, spot, Date(12,7,2062))
tSSP126_60 = delta_P(zrSSP126_60, df_tscenarios.iloc[2,2], tcorr60, spot, Date(13,7,2082))
print('tSSP126_20 : ', tSSP126_20)
print('tSSP126_40 : ', tSSP126_40)
print('tSSP126_60 : ', tSSP126_60)


# In[41]:


#We recover the zero rates that we wish to bias in SSP245
zrSSP245_20 = all_zrSSP245.zeroRate(Actual365Fixed().yearFraction(spot, Date(14,7,2042)), 
                                           Continuous, Annual).rate()
zrSSP245_40 = all_zrSSP245.zeroRate(Actual365Fixed().yearFraction(spot, Date(12,7,2062)), 
                                           Continuous, Annual).rate()
zrSSP245_60 = all_zrSSP245.zeroRate(Actual365Fixed().yearFraction(spot, Date(13,7,2082)), 
                                           Continuous, Annual).rate()
print(zrSSP245_20, zrSSP245_40, zrSSP245_60)


# In[42]:


#SCENARIO SSP - 4.5
tSSP245_20 = delta_P(zrSSP245_20, df_tscenarios.iloc[0,3], tcorr20, spot, Date(14,7,2042))
tSSP245_40 = delta_P(zrSSP245_40, df_tscenarios.iloc[1,3], tcorr40, spot, Date(12,7,2062))
tSSP245_60 = delta_P(zrSSP245_60, df_tscenarios.iloc[2,3], tcorr60, spot, Date(13,7,2082))
print('tSSP245_20 : ', tSSP245_20)
print('tSSP245_40 : ', tSSP245_40)
print('tSSP245_60 : ', tSSP245_60)


# In[43]:


#We recover the zero rates that we wish to bias in SSP370
zrSSP370_20 = all_zrSSP370.zeroRate(Actual365Fixed().yearFraction(spot, Date(14,7,2042)), 
                                           Continuous, Annual).rate()
zrSSP370_40 = all_zrSSP370.zeroRate(Actual365Fixed().yearFraction(spot, Date(12,7,2062)), 
                                           Continuous, Annual).rate()
zrSSP370_60 = all_zrSSP370.zeroRate(Actual365Fixed().yearFraction(spot, Date(13,7,2082)), 
                                           Continuous, Annual).rate()
print(zrSSP370_20, zrSSP370_40, zrSSP370_60)


# In[44]:


#SCENARIO SSP - 7.0
tSSP370_20 = delta_P(zrSSP370_20, df_tscenarios.iloc[0,4], tcorr20, spot, Date(14,7,2042))
tSSP370_40 = delta_P(zrSSP370_40, df_tscenarios.iloc[1,4], tcorr40, spot, Date(12,7,2062))
tSSP370_60 = delta_P(zrSSP370_60, df_tscenarios.iloc[2,4], tcorr60, spot, Date(13,7,2082))
print('tSSP370_20 : ', tSSP370_20)
print('tSSP370_40 : ', tSSP370_40)
print('tSSP370_60 : ', tSSP370_60)


# In[45]:


#We recover the zero rates that we wish to bias in SSP585
zrSSP585_20 = all_zrSSP585.zeroRate(Actual365Fixed().yearFraction(spot, Date(14,7,2042)), 
                                           Continuous, Annual).rate()
zrSSP585_40 = all_zrSSP585.zeroRate(Actual365Fixed().yearFraction(spot, Date(12,7,2062)), 
                                           Continuous, Annual).rate()
zrSSP585_60 = all_zrSSP585.zeroRate(Actual365Fixed().yearFraction(spot, Date(13,7,2082)), 
                                           Continuous, Annual).rate()
print(zrSSP585_20, zrSSP585_40, zrSSP585_60)


# In[46]:


#SCENARIO SSP5 - 8.5
tSSP585_20 = delta_P(zrSSP585_20, df_tscenarios.iloc[0,5], tcorr20, spot, Date(14,7,2042))
tSSP585_40 = delta_P(zrSSP585_40, df_tscenarios.iloc[1,5], tcorr40, spot, Date(12,7,2062))
tSSP585_60 = delta_P(zrSSP585_60, df_tscenarios.iloc[2,5], tcorr60, spot, Date(13,7,2082))
print('tSSP585_20 : ', tSSP585_20)
print('tSSP585_40 : ', tSSP585_40)
print('tSSP585_60 : ', tSSP585_60)


# #  Interpolating delta zero rates for temperature

# In[47]:


df_Pmodifiedt = pd.read_excel(r'C:\Users\Cathy\Documents\DOC\QF-UniBo\TESI\3_Data\em_temp_benchmark.xlsm', sheet_name='Scenarios')


# In[48]:


dates = [Date(8,7,2022)] + [Date(12,7,2022)]  + semiannual[1:5] + annual[3:21] + [annual[25]] + [annual[30]] + [annual[35]] + [annual[40]] + [annual[50]] + [annual[60]]
#Construct the delta curve of P6m object
ORIGINAL = [df_Pmodifiedt.Original[i] for i in range(len(df_Pmodifiedt))]
tPSSP119 = [df_Pmodifiedt.PSSP119[i] for i in range(len(df_Pmodifiedt))]
tPSSP126 = [df_Pmodifiedt.PSSP126[i] for i in range(len(df_Pmodifiedt))]
tPSSP245 = [df_Pmodifiedt.PSSP245[i] for i in range(len(df_Pmodifiedt))]
tPSSP370 = [df_Pmodifiedt.PSSP370[i] for i in range(len(df_Pmodifiedt))]
tPSSP585 = [df_Pmodifiedt.PSSP585[i] for i in range(len(df_Pmodifiedt))]
DORIGINAL = DiscountCurve(dates,ORIGINAL, Actual365Fixed())
tdelta_zcbSSP119= DiscountCurve(dates, tPSSP119, Actual365Fixed())
tdelta_zcbSSP126= DiscountCurve(dates, tPSSP126, Actual365Fixed())
tdelta_zcbSSP245= DiscountCurve(dates, tPSSP245, Actual365Fixed())
tdelta_zcbSSP370= DiscountCurve(dates, tPSSP370, Actual365Fixed())
tdelta_zcbSSP585= DiscountCurve(dates, tPSSP585, Actual365Fixed())


# In[49]:


delta_zero_rates = [DORIGINAL.zeroRate(Actual365Fixed().yearFraction(today, d), Continuous, Annual).rate()
             for d in dates]
tdelta_zrSSP119 = [tdelta_zcbSSP119.zeroRate(Actual365Fixed().yearFraction(today, d), Continuous, Annual).rate()
             for d in dates]
tdelta_zrSSP126 = [tdelta_zcbSSP126.zeroRate(Actual365Fixed().yearFraction(today, d), Continuous, Annual).rate()
             for d in dates]
tdelta_zrSSP245 = [tdelta_zcbSSP245.zeroRate(Actual365Fixed().yearFraction(today, d), Continuous, Annual).rate()
             for d in dates]
tdelta_zrSSP370 = [tdelta_zcbSSP370.zeroRate(Actual365Fixed().yearFraction(today, d), Continuous, Annual).rate()
             for d in dates]
tdelta_zrSSP585 = [tdelta_zcbSSP585.zeroRate(Actual365Fixed().yearFraction(today, d), Continuous, Annual).rate()
             for d in dates]


# In[50]:


#Construct Zero Curve object
all_original = MonotonicCubicZeroCurve(dates, delta_zero_rates, Actual365Fixed())
tall_zrSSP119 = MonotonicCubicZeroCurve(dates, tdelta_zrSSP119, Actual365Fixed())
tall_zrSSP126 = MonotonicCubicZeroCurve(dates, tdelta_zrSSP126, Actual365Fixed())
tall_zrSSP245 = MonotonicCubicZeroCurve(dates, tdelta_zrSSP245, Actual365Fixed())
tall_zrSSP370 = MonotonicCubicZeroCurve(dates, tdelta_zrSSP370, Actual365Fixed())
tall_zrSSP585 = MonotonicCubicZeroCurve(dates, tdelta_zrSSP585, Actual365Fixed())


# In[65]:


dates = [ today+Period(i,Days) for i in range(0,21916+1) ]
#dates = [ spot+Period(i,Months) for i in range(0,12*60+1) ]
#dates = [ spot+Period(i,Years) for i in range(0,60+1,2) ]
#dates = [Date(12,7,2022)] + semiannual[1:5] + annual[3:21] + [annual[25]] + [annual[30]] + [annual[35]] + [annual[40]] + [annual[50]] + [annual[60]]
#dates = [Date(8,7,2022)] + [annual[i] for i in range(2,len(annual),2)]
#dates = [Date(12,7,2022)] + [annual[i] for i in range(2,len(annual),2)]
zrSSP119t = [tall_zrSSP119.zeroRate(Actual365Fixed().yearFraction(today, d), Continuous, Annual).rate()
             for d in dates]
zrSSP126t = [tall_zrSSP126.zeroRate(Actual365Fixed().yearFraction(today, d), Continuous, Annual).rate()
             for d in dates]
zrSSP245t = [tall_zrSSP245.zeroRate(Actual365Fixed().yearFraction(today, d), Continuous, Annual).rate()
             for d in dates]
zrSSP370t = [tall_zrSSP370.zeroRate(Actual365Fixed().yearFraction(today, d), Continuous, Annual).rate()
              for d in dates]
zrSSP585t = [tall_zrSSP585.zeroRate(Actual365Fixed().yearFraction(today, d), Continuous, Annual).rate()
              for d in dates]
zero_rate = [all_original.zeroRate(Actual365Fixed().yearFraction(today, d), Continuous, Annual).rate()
             for d in dates ]
dates1 = [d.to_date() for d in dates]
fig, ax = plt.subplots() 
ax.plot(dates1,zero_rate, '--', label = 'Original')
ax.plot(dates1,zrSSP119t, '#107d00', label = 'SSP1-1.9')
ax.plot(dates1,zrSSP126t , '#16D92C', label = 'SSP1-2.6')
ax.plot(dates1,zrSSP245t, '#FECE05', label = 'SSP2-4.5')
ax.plot(dates1,zrSSP370t, '#E33630',label = 'SSP3-7.0')
ax.plot(dates1,zrSSP585t, '#8F1E36', label = 'SSP5-8.5')
ax.grid(axis = 'y', linewidth = 0.5)
ax.legend()
ax.set_title('Temperature | Emissions Benchmark Curves')


# In[ ]:


#Create a file as input of the data for the two-facto Vasicek model


# In[52]:


em_temp = pd.DataFrame()


# In[53]:


em_temp['Tau'] = [Actual365Fixed().yearFraction(today, d) for d in dates]
em_temp['Original'] = zero_rate
em_temp['SSP119'] = zrSSP119t
em_temp['SSP126'] = zrSSP126t
em_temp['SSP245'] = zrSSP245t
em_temp['SSP370'] = zrSSP370t
em_temp['SSP585'] = zrSSP585t
em_temp


# In[54]:


#em_temp.to_excel('em_temp.xlsx')


# In[ ]:





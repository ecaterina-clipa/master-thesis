#!/usr/bin/env python
# coding: utf-8

# # Two factor Vasicek model implementation

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import pandas as pd
import random
import time 
from multiprocessing import Pool


# Functions for the calibration of the two-factor Vasicek model

# In[7]:


#Zero coupon bond P(t,Ti) under the two-factor Vasicek model
def zcb(r0, c0, tau, corr, kappa1, kappa2, theta1, theta2, lambda1, lambda2, sigma1, sigma2):
    ''' INPUTS
    tau: year fractions of the zcb considered
    r0 and c0: vectors of random values
    kappa1, kappa2, theta1, theta2, lambda1, lambda2, sigma1, sigma2: parameters to estimate
    corr: correlation between the two factors for that period
    sigma12 = corr*sigma1*sigma2 
    OUTPUT
    p: model zero coupon bond price'''
    
    B1 = (1-np.exp(-kappa1*tau)) / kappa1
    B2 = (1-np.exp(-kappa2*tau)) / kappa2
    
    gamma1= kappa1**2*(theta1-(sigma1*lambda1)/kappa1) - sigma1**2/2
    gamma2= kappa2**2*(theta2-(sigma2*lambda2)/kappa2) - sigma2**2/2
    
    A = (gamma1*(B1-tau))/kappa1**2 - (sigma1**2 * B1**2)/(4*kappa1) +    (gamma2*(B2-tau))/kappa2**2 - (sigma2**2 * B2**2)/(4*kappa2) +     ((corr*sigma1*sigma2)/(2*kappa1*kappa2))*((tau - B1 - B2) + (1- np.exp(- tau*(kappa1+kappa2)))/(kappa1+kappa2)) +     ((corr*sigma1*sigma2)/(2*kappa1*kappa2))*((tau - B2 - B1 ) + (1- np.exp(- tau*(kappa1+kappa2)))/(kappa1+kappa2))
    
    A_ = A
    B_1 = B1
    B_2 = B2
    r_1 = r0
    c_2 = c0
    
    p = np.exp(A_ - B_1 * r_1 - B_2 * c_2)
    return p

#Function for the recovering of the value of the zero rate from the zero-coupon bond
def zero_rates(tau, p):
    zr = -np.log(p)/tau
    return zr

#Objective Function corresponds to the residuals between the model zero rates and the distored zero rates 
def objFunc(par, r0, c0, tau, corr, original_zr):
    #unpack parameters list
    kappa1 = par[0]
    kappa2 = par[1]
    theta1 = par[2]
    theta2 = par[3]
    lambda1 = par[4]
    lambda2 = par[5]
    sigma1 = par[6]
    sigma2 = par[7]
    
    #calculate the zcb prices P(t,T) for a r0 and a c0
    p = zcb(r0, c0, tau, corr, kappa1, kappa2, theta1, theta2, lambda1, lambda2, sigma1, sigma2)
    
    #Calculate model zero rates
    ZR = zero_rates(tau, p)
    
    #the goal is to minimize the distance between model rates and market rates
    diff = (original_zr - ZR)
      
    return diff

def calibration(func, param, r0, c0, tau, corr, original_zr):
    '''INPUTS:
    func: objective function that will be minimized through least squares
    param: vectors of guessed parameters
    r0 and c0: vectors of random values 
    tau: year fractions of the zcb considered
    corr: correlation between the two factors for that period
    original_zr: zero-coupon rates from the benchmark curve
    OUTPUTS
    par: vector of parameters estimated
    p: zero-coupon bonds of the model
    zr: zero-coupon rates of the model'''
    #start = time.time()
    sol = least_squares(func, param, args= (r0, c0, tau, corr, original_zr), method='trf')
    par = sol.x
    kappa1_star = par[0]
    kappa2_star = par[1]
    theta1_star = par[2]
    theta2_star = par[3]
    lambda1_star = par[4]
    lambda2_star = par[5]
    sigma1_star = par[6]
    sigma2_star = par[7]
    
    #calculate the zcb and zero rates with calibrated parameters (star values)
    p = zcb(r0, c0, tau, corr, kappa1_star, kappa2_star, theta1_star, theta2_star, lambda1_star, lambda2_star,
            sigma1_star, sigma2_star)
    zr = zero_rates(tau,p)
    #end = time.time()
    #elapsed = end - start
    #print('It takes', elapsed, 'seconds')
    return par, p, zr 


# In[3]:


def model_curve(Nt, seed, guess_par, tau, original_zr):
    '''INPUT
    Nt: number of simulations
    seed: seed for the random number generator
    guess_par: initial guess for the parameters of the model (kappa1, kappa2, theta1, theta2, lambda1, lambda2, sigma1, sigma2)
    tau: year fraction
    original_zr: zero-coupon rates from the benchmark curve
    OUTPUT
    mean_par: average of the parameters found
    mean_P: average of the zero-coupon bonds found
    zr_mean: average of the zero-coupon rates found'''

    #Selection of the correlation for that part of the curve (corresponding to the three shifts at 20, 40, 60 years of maturity)
    if tau.iloc[-1] < 21:
        corr = 0.514746784
    elif tau.iloc[-1]< 41:
        corr = 0.531856011
    else:
        corr = 0.533238941
    
    parameters = []
    P = []
    zr = []
    
    for i in range(Nt):        
        random.seed(seed+i)
        r0 = [random.uniform(0.00,0.03) for i in range(len(tau))]
        c0 = [random.uniform(0.00,0.03) for i in range(len(tau))]

        par, p11, zr11  = calibration(objFunc, guess_par, r0, c0, tau, corr, curve)
        parameters.append(par)
        P.append(p11)
        zr.append(zr11)
    
    mean_par = np.mean(parameters, axis = 0).tolist()
    mean_P = np.mean(P, axis = 0).tolist()
    zr_mean = np.mean(zr, axis = 0).tolist()
    return mean_par, mean_P, zr_mean


# # # Benchmark curves

# In[2]:


emissions = pd.read_excel(r'C:\Users\Cathy\Documents\DOC\QF-UniBo\TESI\3_Data\emissions.xlsx') #change path
#Year fractions
original_tau = emissions['Tau']
#Zero rates from all the scenarios
original = emissions['Original']
zrSSP119 = emissions['SSP119']
zrSSP126 = emissions['SSP126']
zrSSP245 = emissions['SSP245']
zrSSP370 = emissions['SSP370']
zrSSP585 = emissions['SSP585']


# In[4]:


emissions


# Sampling random values for the stochastic components r0 and c0

# In[9]:


Nt = 1000
params = [3,4, 1, 3,0.6,0.5,0.3, 0.3]


# In[5]:


#calibration every 10 years
parts = [[i for i in range(0,6)], [i for i in range(6,11)], [i for i in range(11,16)], [i for i in range(16,21)], [i for i in range(21,26)], [i for i in range(26,31)]]


# In[ ]:


with Pool(processes=1) as pool:
    print("Starting pool of processes...")
    SSP119 = [pool.apply_async(model_curve, args = (Nt, 1234, params, original_tau[i], zrSSP119[i])) for i in parts]
    SSP126 = [pool.apply_async(model_curve, args = (Nt, 1234, params, original_tau[i], zrSSP126[i])) for i in parts]
    SSP245 = [pool.apply_async(model_curve, args = (Nt, 1234, params, original_tau[i], zrSSP245[i])) for i in parts]
    SSP370= [pool.apply_async(model_curve, args = (Nt, 1234, params, original_tau[i], zrSSP370[i])) for i in parts]
    SSP585 = [pool.apply_async(model_curve, args = (Nt, 1234, params, original_tau[i], zrSSP585[i])) for i in parts]
    final_resultsSSP119 = [res.get() for res in SSP119]
    final_resultsSSP126 = [res.get() for res in SSP126]
    final_resultsSSP245 = [res.get() for res in SSP245]
    final_resultsSSP370 = [res.get() for res in SSP370]
    final_resultsSSP585 = [res.get() for res in SSP585]

print("All results calculated!")


# In[ ]:


parameters = []
P = []
zr = []
for i in range(len(final_resultsSSP119)):
        parameters += final_resultsSSP119[i][0]
        P += final_resultsSSP119[i][1]
        zr += final_resultsSSP119[i][2]
        
df = pd.DataFrame(list(zip(zr, P)), columns=['Zr_SSP119', 'Zcb_SSP119'])
df_par = pd.DataFrame(parameters, columns = ['SSP119'])

parameters = []
P = []
zr = []
for i in range(len(final_resultsSSP126)):
        parameters += final_resultsSSP126[i][0]
        P += final_resultsSSP126[i][1]
        zr += final_resultsSSP126[i][2]
        
df['Zr_SSP126'] = zr
df['Zcb_SSP126'] = P
df_par['SSP126'] = parameters

parameters = []
P = []
zr = []
for i in range(len(final_resultsSSP245)):
        parameters += final_resultsSSP245[i][0]
        P += final_resultsSSP245[i][1]
        zr += final_resultsSSP245[i][2]
        
df['Zr_SSP245'] = zr
df['Zcb_SSP245'] = P
df_par['SSP245'] = parameters

parameters = []
P = []
zr = []
for i in range(len(final_resultsSSP370)):
        parameters += final_resultsSSP370[i][0]
        P += final_resultsSSP370[i][1]
        zr += final_resultsSSP370[i][2]
        
df['Zr_SSP370'] = zr
df['Zcb_SSP370'] = P
df_par['SSP370'] = parameters

parameters = []
P = []
zr = []
for i in range(len(final_resultsSSP585)):
        parameters += final_resultsSSP585[i][0]
        P += final_resultsSSP585[i][1]
        zr += final_resultsSSP585[i][2]
        
df['Zr_SSP585'] = zr
df['Zcb_SSP585'] = P
df_par['SSP585'] = parameters

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('emissions_results.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df.to_excel(writer, sheet_name='Rates')
df_par.to_excel(writer, sheet_name='Parameters')

# Close the Pandas Excel writer and output the Excel file.
writer.save()


# In[3]:


SSP = pd.read_excel(r'C:\Users\Cathy\Documents\DOC\QF-UniBo\TESI\3_Data\emissions_results.xlsx')


# In[6]:


fig, ax = plt.subplots() 
ax.plot(original_tau, zrSSP119, '#107d00', label = 'Original')
ax.plot(original_tau, SSP['Zr_SSP119'], '.--', color = 'b', label = 'Model')
ax.grid(axis = 'y', linewidth = 0.5)
ax.legend()
ax.set_title('SSP1-1.9 Benchmark Curve Simulation')


# In[11]:


fig, ax = plt.subplots() 
ax.plot(original_tau, zrSSP126, '#16D92C', label = 'Original')
ax.plot(original_tau, SSP['Zr_SSP126'], '.--', color = 'b', label = 'Model')
ax.grid(axis = 'y', linewidth = 0.5)
ax.legend()
ax.set_title('SSP1-2.6 Benchmark Curve Simulation')


# In[12]:


fig, ax = plt.subplots() 
ax.plot(original_tau, zrSSP245, '#FECE05', label = 'Original')
ax.plot(original_tau, SSP['Zr_SSP245'], '.-', color = 'b', label = 'Model')
ax.grid(axis = 'y', linewidth = 0.5)
ax.legend()
ax.set_title('SSP2-4.5 Benchmark Curve Simulation')


# In[13]:


fig, ax = plt.subplots() 
ax.plot(original_tau, zrSSP370, '#E33630', label = 'Original')
ax.plot(original_tau, SSP['Zr_SSP370'], '.-', color = 'b', label = 'Model')
ax.grid(axis = 'y', linewidth = 0.5)
ax.legend()
ax.set_title('SSP3-7.0 Benchmark Curve Simulation')


# In[14]:


fig, ax = plt.subplots() 
ax.plot(original_tau, zrSSP585, '#8F1E36', label = 'Original')
ax.plot(original_tau, SSP['Zr_SSP585'], '.-', color = 'b', label = 'Model')
ax.grid(axis = 'y', linewidth = 0.5)
ax.legend()
ax.set_title('SSP5-8.5 Benchmark Curve Simulation')


# In[15]:


plt.suptitle('Emission Benchmark Curves')

ax1 = plt.subplot2grid(shape=(3,6), loc=(0,1), colspan=2)
ax2 = plt.subplot2grid((3,6), (0,3), colspan=2)
ax3 = plt.subplot2grid((3,6), (1,1), colspan=2)
ax4 = plt.subplot2grid((3,6), (1,3), colspan=2)
ax5 = plt.subplot2grid((3,6), (2,2), colspan=2)
plt.subplots_adjust(hspace=0.4, wspace=1)

ax1.plot(original_tau, zrSSP119, '#107d00', label = 'Original')
ax1.plot(original_tau, SSP['Zr_SSP119'], '.', color = 'b', label = 'Model')
ax1.grid(axis = 'y', linewidth = 0.5)
ax1.legend()
ax1.set_title('SSP1-1.9')

ax2.plot(original_tau, zrSSP126, '#16D92C', label = 'Original')
ax2.plot(original_tau, SSP['Zr_SSP126'], '.', color = 'b', label = 'Model')
ax2.grid(axis = 'y', linewidth = 0.5)
ax2.legend()
ax2.set_title('SSP1-2.6')

ax3.plot(original_tau, zrSSP245, '#FECE05', label = 'Original')
ax3.plot(original_tau, SSP['Zr_SSP245'], '.', color = 'b', label = 'Model')
ax3.grid(axis = 'y', linewidth = 0.5)
ax3.legend()
ax3.set_title('SSP2-4.5')

ax4.plot(original_tau, zrSSP370, '#E33630', label = 'Original')
ax4.plot(original_tau, SSP['Zr_SSP370'], '.', color = 'b', label = 'Model')
ax4.grid(axis = 'y', linewidth = 0.5)
ax4.legend()
ax4.set_title('SSP3-7.0')

ax5.plot(original_tau, zrSSP585, '#8F1E36', label = 'Original')
ax5.plot(original_tau, SSP['Zr_SSP585'], '.', color = 'b', label = 'Model')
ax5.grid(axis = 'y', linewidth = 0.5)
ax5.legend()
ax5.set_title('SSP5-8.5')


# In[ ]:





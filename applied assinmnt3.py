# -*- coding: utf-8 -*-
"""
Created on Thu May 11 20:45:52 2023

@author: joels
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cluster_tools as ct
import sklearn.metrics as skmet
import sklearn.cluster as cluster
from scipy.optimize import curve_fit


def reading_af(fname, Countries, Years):
    """a function called reading_funcn is defined to read excel file and will
    return 2 dataframes one with countries as coloumns and one with years as
    coloumns
    parameters:
        fname : file name
        countries : contries which i need to plot
        years : period of time
        """
    # reading the excel file
    af0 = pd.read_excel(fname, skiprows=3)
    # dropping the country code from dataframe
    af0.drop(columns=["Country Code"], axis=1, inplace=True)
    # making  Country name as index
    af0.set_index(['Country Name'], inplace=True)
    # specifically to select countries and years from data frame af0
    # saving to new dataframe af1
    af1 = af0.iloc[Countries, Years]
    # sorting index of pandas dataframe and renaming the axis
    # filling the missing values to 0 by fillna(0)
    af1 = af1.sort_index().rename_axis("Years", axis=1).fillna(0)
    af2 = af1.T
    return af1, af2


def exp_growth(t, n0, g):
    """ Computes exponential function with scale and growth as free parameters
"""
    t = t - 1990
    f = n0 * np.exp(g * t)
    return f


def logistics(t, a, k, t0):
    """ Computes logistics function with scale and incr as free parameters
"""
    f = a / (1.0 + np.exp(-k * (t - t0)))
    return f


def err_ranges(years, exp_growth, popt, sigma):
    import itertools as iter
    lower = exp_growth(years, *popt)
    upper = lower
    uplow = []
    for p, s in zip(popt, sigma):
        pmin = p-s
        pmax = p+s
        uplow.append((pmin, pmax))
        pmix = list(iter.product(*uplow))
        for p in pmix:
            y = exp_growth(years, *popt)
            lower = np.minimum(lower, y)
            upper = np.maximum(upper, y)
    return lower, upper


Countries = [35, 55, 67, 78, 81, 99, 109, 129, 148, 151, 159, 165, 169, 172,
             176, 180, 189, 199, 219, 220, 225, 230, 238, 241, 250, 251]
Years = [32, 33, 34, 35, 53, 55, 57, 59, 60, 61]

energy_f1, energy_f2 = reading_af("renauble energy consumption.xls", Countries,
                                  Years)
methaneemission_f1, methaneemission_f2 = reading_af("methane.xls", Countries,
                                                    Years)
# 1st scatter plot
# extract the two columns for clustering
df_clust = energy_f1[["1990", "2015"]]
# entries with one nan are useless
df_clust = df_clust.dropna()
df_clust = df_clust.reset_index()
print(df_clust.iloc[0:15])
df_clust.set_index(['Country Name'], inplace=True)
df_norm, df_min, df_max = ct.scaler(df_clust)
print()
print("n score")
# loop over number of clusters
for ncluster in range(2, 10):
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)
    # Fit the data, results are stored in the kmeans object
    # fit done on x,y pairs
    kmeans.fit(df_norm)
    labels = kmeans.labels_
# extract the estimated cluster centres
    cen = kmeans.cluster_centers_
# calculate the silhoutte score
print(ncluster, skmet.silhouette_score(df_clust, labels))

# best number of clusters
ncluster = 3
# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=ncluster)
# Fit the data, results are stored in the kmeans object
# fit done on x,y pairs
kmeans.fit(df_norm)
labels = kmeans.labels_
# extract the estimated cluster centres
cen = kmeans.cluster_centers_
cen = np.array(cen)
xcen = cen[:, 0]
ycen = cen[:, 1]
# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(df_norm["1990"], df_norm["2015"], 30, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 55, "k", marker="d")
plt.xlabel("Energy produced(1990)")
plt.ylabel("Energy produced(2015)")
plt.title("Energy production in 1990 and 2015")
plt.show()

print(cen)
# Applying the backscale function to convert the cluster centre
scen = ct.backscale(cen, df_min, df_max)
print()
print(scen)
xcen = scen[:, 0]
ycen = scen[:, 1]
# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')

plt.scatter(df_clust["1990"], df_clust["2015"], 30, labels,
            marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.xlabel("Energy produced(1990)")
plt.ylabel("Energy produced(2015)")
plt.title("Energy production in 1990 and 2015")
plt.show()

# 2nd scatter

df_clust = methaneemission_f1[["1990", "2015"]]
# entries with one nan are useless
df_clust = df_clust.dropna()
df_clust = df_clust.reset_index()
print(df_clust.iloc[0:15])
df_clust.set_index(['Country Name'], inplace=True)
df_norm, df_min, df_max = ct.scaler(df_clust)
print()
print("n score")
# loop over number of clusters
for ncluster in range(2, 10):
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)
    # Fit the data, results are stored in the kmeans object
    # fit done on x,y pairs
    kmeans.fit(df_norm)
    labels = kmeans.labels_
# extract the estimated cluster centres
    cen = kmeans.cluster_centers_
# calculate the silhoutte score
print(ncluster, skmet.silhouette_score(df_clust, labels))

# best number of clusters
ncluster = 3
# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=ncluster)
# Fit the data, results are stored in the kmeans object
# fit done on x,y pairs
kmeans.fit(df_norm)
labels = kmeans.labels_
# extract the estimated cluster centres
cen = kmeans.cluster_centers_
cen = np.array(cen)
xcen = cen[:, 0]
ycen = cen[:, 1]
# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(df_norm["1990"], df_norm["2015"], 30, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 55, "k", marker="d")
plt.xlabel("methane emisssion(1990)")
plt.ylabel("methane emisssion(2015)")
plt.title("Methane Emission in 1990 and 2015")
plt.show()

print(cen)
# Applying the backscale function to convert the cluster centre
scen = ct.backscale(cen, df_min, df_max)
print()
print(scen)
xcen = scen[:, 0]
ycen = scen[:, 1]
# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')

plt.scatter(df_clust["1990"], df_clust["2015"], 30, labels,
            marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.xlabel("methane emisssion(1990)")
plt.ylabel("methane emisssion(2015)")
plt.title("Methane Emission in 1990 and 2015")
plt.show()

# plotting line graph
date = [32, 37, 42, 47, 52, 57, 61]
County = [35]
totpopulation_f1, totpopulation_f2 = reading_af("assinmnt2totalpopulation.xls",
                                                County, date)
totpopulation_f2 = totpopulation_f2.tail(-1)
totpopulation_f2.reset_index(inplace=True)
# totpopulation_f2.columns = ['Year', 'population']
totpopulation_f2["Years"] = totpopulation_f2["Years"].astype(int)
totpopulation_f2["Canada"] = totpopulation_f2["Canada"].astype(float)
print(totpopulation_f2)

# totpopulation_f2 = totpopulation_f2.tail(-1)
totpopulation_f2.reset_index(inplace=True)

totpopulation_f2 = totpopulation_f2.rename(columns={'Canada': 'population'})
print("totpopulation_f2")
# plotting
plt.plot(totpopulation_f2["Years"], totpopulation_f2["population"],
         label="total population")
plt.xlabel("YEARS")
plt.ylabel("Population, Total")
plt.title("Total population of Canada")
# curve fitting
popt, pcov = curve_fit(exp_growth, totpopulation_f2["Years"],
                       totpopulation_f2["population"], p0=(0.27e7, 0.056))
print("TOTAL POPULATION 1990", popt[0]/1e9)
plt.plot(totpopulation_f2["Years"], exp_growth(totpopulation_f2["Years"],
                                               *popt), label="Curved Fit",
         color="navy")
# years we ned to plot
YEAR = np.linspace(1995, 2030, 5)
pop = exp_growth(YEAR, *popt)
plt.plot(YEAR, pop, label="TREND", linewidth=1, alpha=1, color="cyan")
sigma = np.sqrt(np.diag(pcov))
# to find error
low, up = err_ranges(YEAR, exp_growth, popt, sigma)
plt.fill_between(YEAR, low, up, color="crimson", alpha=0.7, label="ERROR")
plt.legend()
plt.grid()
plt.show()

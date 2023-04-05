# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 16:11:10 2023

@author: joels
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


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
    # Transposing the dataframe af1 and save as af2
    af2 = af1.T

    return af1, af2


def stats_af(af):
    """a fuction called stats_af is used to do some basic statistic on
    the data frame. Takes the dataframe with countries as
    columns as the argument.
    parameter:
        af = file name
        """
    # To explore dataset by describe()
    print(af.describe())
    # to find skewness
    print("\nSkewness:\n", skew(af))
    # to find kurtosis
    print("\nKurtosis:\n", kurtosis(af))

    return


def heatmapped(filename, Country, Indicators):
    """ a function called heatmapped is used here to plot a heatmap which
    will show the correlation between indicators which causes
    climate changes.
    parameters:
        filename: the file name we need to plot
        country:specific country to show the correlation
        indicators:specific indicators resulting in climatic changes
        """
    # reading the excel file
    af0 = pd.read_excel(filename, skiprows=3)
    # dropping the country code from dataframe
    af0.drop(columns=["Country Code", "Indicator Code"], axis=1, inplace=True)
    # making  Country name as index
    af0.set_index(["Country Name", "Indicator Name"], inplace=True)
    # specifically to select countries and years from data frame af0
    # saving to new dataframe af1
    af1 = af0.loc[Country].fillna(0).T
    # slicing the dataframe
    af = af1.loc["1970":"2015", Indicators]
    # plotting the heatmap
    plt.figure(figsize=(10, 5))
    plt.title('Correlation Heatmap of india', fontsize='x-large')
    sns.heatmap(af.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')

    return


def plotting_af(af, kind, title):
    """ a function called plotting_af is used to plot graph.
    parameters:
        af : file name
        kind : kind of the graph to plot like bar, line.
        title : to give title for the graph.
        """
    af = af.plot(kind=kind, figsize=(9, 6), rot=45)
    # to set legend in best location and defining the fontsize of legend.
    af.legend(loc='best', fontsize=10)
    # to set title
    af.set_title(title, fontweight='bold', fontsize='x-large',
                 fontname="Times New Roman")
    return


# To choose the countries and years need to use for dataframes


Countries = [35, 40, 55, 81, 109, 119, 202, 205, 251]
Years = [37, 42, 47, 52, 57, 61]

# creating 4 dataframes and its transpose
forest_f1, forest_f2 = reading_af("assinmnt2forestarea.xls",
                                  Countries, Years)
co2emission_f1, co2emission_f2 = reading_af("assinmnt2co2emission.xls",
                                            Countries, Years)
totpopulation_f1, totpopulation_f2 = reading_af("assinmnt2totalpopulation.xls",
                                                Countries, Years)
energy_f1, energy_f2 = reading_af("assinmnt2energyuse.xls", Countries, Years)


# plotting 1 bar graphs
plotting_af(forest_f1, 'bar',
            'Percentage of Forest land in 9 different countries')
# initialising the x label
plt.xlabel("Countries", fontweight='bold')
# initialising the y label
plt.ylabel("% of land area", fontweight='bold')
# saving the figure
plt.savefig("figure1ass2.png")

# plotting 2 line graphs
plotting_af(totpopulation_f2, 'line',
            'Total Population in 9 different countries')
# initialising the x label
plt.xlabel("Years", fontweight='bold')
# initialising the y label
plt.ylabel("Population", fontweight='bold')
# setting the grid
plt.grid(zorder=0)
# saving the figure
plt.savefig("figure2ass2.png")

plotting_af(co2emission_f2, 'line', 'Carbon Dioxide emissions(kt)')
# initialising the x label
plt.xlabel("Years", fontweight='bold')
# initialising the y label
plt.ylabel("CO2 emissions (kt)", fontweight='bold')
# setting the grid
plt.grid(zorder=0)
# saving the figure
plt.savefig("figure3ass2.png")

Indicators = ["Population, total", "CO2 emissions (kt)",
              "Arable land (% of land area)",
              "Forest area (% of land area)"]
# calling the function
heatmap = heatmapped("API_19_DS2_en_excel_v2_5360124.xls", "India", Indicators)
# saving the figure
plt.savefig("figure3ass2.png")

plt.show()


# some basic statistic function using stats_af function
stats_af(forest_f2)
stats_af(totpopulation_f2)
stats_af(co2emission_f2)

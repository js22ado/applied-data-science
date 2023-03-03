# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as plt

# Reading csv files for plotting 3 Graphs

energy_frame1 = pd.read_excel('joelmain.xlsx', index_col=0)
energy_frame2 = pd.read_excel('joel1.xlsx', index_col=0)
energy_frame3 = pd.read_excel('joel2.xlsx', index_col=0)

# LINE GRAPH


def energy_function1(df):
    """ plotting line graph inside a function energy_function with an
    attribute df.while calling the function values will be readed """

# figure size
    plt.figure(figsize=(15, 10))
    """initialising for loop"""
    for i in range(4):
        plt.plot(df.columns, df.iloc[i], label=str(df.index[i]))

# plotting x and y labels

    plt.xlabel("Years", fontweight='bold', size=30)
    plt.ylabel("Electricity produced(in kwh)", fontweight='bold', fontsize=30)

# implement xticks, y ticks, xlim, grid

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlim(df.columns.min(), df.columns.max())
    plt.grid(zorder=0)

# implementing title, legend

    plt.title("Electricity production", fontweight='bold', fontsize=40)
    plt.legend(loc='best', fontsize=20)

# for saving figure
    plt.savefig("plot1_main.png")

    plt.show()
    return

# PIE CHART


def energy_function2(df1):
    """ Plotting a pie chart using a function energy_function with
    an attribute df1.while calling the function values will be readed"""

# figure size

    plt.figure(figsize=[20, 20])

# main function

    plt.pie(energy_frame2[2017], labels=df1.index,
            autopct='%1.1f%%', textprops={'fontsize': 30})

# title

    plt.title('Electricity production in 2017', fontweight='bold',
              fontsize=50)

# saving figure

    plt.savefig("plot2(1)_main.png")

    plt.show()
    return


def energy_function3(df2):
    """Plotting bar graph using a function energy_function with
    an attribute df2.while calling function values will be readed"""

# implementing x ticks

    plt.xticks(energy_frame3.index)

# implementing x and y labels

    plt.xlabel("Years", fontweight='bold', fontsize=10)
    plt.ylabel("Electricity produced(in %)", fontweight='bold', fontsize=10)

# main function

    plt.bar(df2.index, df2['Bahamas'])

# title

    plt.title("Electricity produced in Bahamas",
              fontweight='bold', fontsize=15)

# plotting figure

    plt.figure(figsize=[15, 6])

# saving figure

    plt.savefig("plot3(1)_main.png")

    plt.show()
    return

# function calling


energy_function1(energy_frame1)
energy_function2(energy_frame2)
energy_function3(energy_frame3)

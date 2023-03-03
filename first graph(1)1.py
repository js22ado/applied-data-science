# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as plt


def my_function(df):
    
    plt.figure(figsize=(15,10))
    plt.plot(df.columns, df.iloc[0],label="Burundi")
    plt.plot(df.columns, df.iloc[1],label="Belgium")
    plt.plot(df.columns, df.iloc[2],label="Benin")
    plt.plot(df.columns, df.iloc[3],label="Burkina faso")
    plt.plot(df.columns, df.iloc[4],label="Bangladesh")
    plt.plot(df.columns, df.iloc[5],label="Bulgaria")
    plt.plot(df.columns, df.iloc[6],label="Bahrain")
    plt.plot(df.columns, df.iloc[7],label="Bahamas")
    plt.plot(df.columns, df.iloc[8],label="Bosnia and Herzegovina")
    plt.plot(df.columns, df.iloc[9],label="Belarus")
    plt.legend(loc='best')
    plt.xlabel("Years", size=25)
    plt.ylabel("Electricity produced(in %)",size=25)
    plt.title("Electricity produced between 2007 and 2017 (%)",size=30)
    plt.savefig("plot1_main.png")
    plt.show()
   
     
data_frame1 = pd.read_excel('joel1.xlsx', index_col=0)
my_function(data_frame1)

#Pie chart
plt.pie(data_frame1[2017],labels=data_frame1.index,autopct='%1.1f%%')
plt.title('Electricity production in 2017')
plt.show()
plt.pie(data_frame1[2007],labels=data_frame1.index,autopct='%1.1f%%')
plt.title('Electricity production in 2007')
plt.savefig("plot2_main.png")
plt.show()

#bar chart
energy_frame=pd.read_excel('joel2.xlsx',index_col=0)
plt.Figure(figsize=[15,5])
plt.bar(energy_frame.index,energy_frame['Bahamas'])
plt.xticks(energy_frame.index)
plt.xlabel("Years", size=13)
plt.ylabel("Electricity produced(in %)",size=13)
plt.title("Electricity produced in Bahamas in given years",size=15)
plt.show()
plt.bar(energy_frame.index,energy_frame['Belarus'])
plt.xticks(energy_frame.index)
plt.xlabel("Years", size=13)
plt.ylabel("Electricity produced(in %)",size=13)
plt.title("Electricity produced in Belarus in given years",size=15)
plt.savefig("plot3_main.png")
plt.show()


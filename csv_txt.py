'''
Oct 25th, 2018

convert the csv to text format

'''
import pandas as pd

############################################################
# function

def csv2txt(input,output):
    data=pd.read_csv(filename)
    with open(output,'w') as out:
        data.to_string(out)

csv2txt('resolution.csv','resolution.txt')

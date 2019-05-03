

import pandas as pd
import matplotlib.pyplot as plt


# checking correlation for processed data for full correlation information due to
# the fact correlation works only for numerical data
diabetic_data = pd.read_csv("../data/processed_data/latest_for_research_modified_data.csv")

"""
    plotting a graphical correlation matrix for each pair of columns in the data frame "diabetic_data.csv".
    
    Input:
        df: pandas DataFrame representing the data set "diabetic_data.csv"
        size: vertical and horizontal size of the plot
    
    Displays:
        matrix of correlation between columns. Blue-Cyan-Yellow => less to more correlated 
                                               0 -----------> 1 => ""
"""

corr = diabetic_data.corr()  # pandas data frame correlation function
size = len(diabetic_data.columns)
fig, ax = plt.subplots(figsize=(size, size))
ax.matshow(corr)  # color code the rectangles by correlation value
plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks
plt.show()

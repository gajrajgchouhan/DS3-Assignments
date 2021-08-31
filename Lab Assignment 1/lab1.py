import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# imports

data = pd.read_csv("landslide_data3.csv")
# reading csv file into a pandas dataframe
data.rename(columns={"lightavgw/o0": "lightavgw"}, inplace=True)
# renaming column to remove '/'

data_top = data.columns.to_list()
# list of the headers of dataframe

"""
I made a dictionary consisting of columns and their units to properly label the axes along with their units.
"""
units = {
    "temperature": "celcius",
    "humidity": r"% relative humidity",
    "pressure": "millibars (mb)",
    "rain": "millilitres (ml)",
    "lightavgw": "lux unit",
    "lightmax": "lux unit",
    "moisture": "%",
}

df = {"mean" : [], "median" : [], "mode" : [], "minimum" : [], "maximum" : [], "standard-dev" : []}
# I'll be using a dataframe to store the properties asked in the question.
# The properties will be stored as a list for each key in the dictionary
# We can convert the dictionary to a proper dataFrame later on.
# Question 1
for column in data_top[2:]:
    # iterating over header names, skipping dates and stationid
    data_of_attribute = data.loc[:, column]
    # selecting the whole column (:) using the label (column) by loc method
    df["mean"].append(data_of_attribute.mean())
    # mean of column using mean method
    df["median"].append(data_of_attribute.median())
    # median of column using median method
    df["mode"].append(data_of_attribute.mode().to_list())
    # mode of column using mode method, using to_list incase there may not be a unique mode
    df["minimum"].append(min(data_of_attribute))
    # minimum value of this column
    df["maximum"].append(max(data_of_attribute))
    # maximum value of this column
    df["standard-dev"].append(np.std(data_of_attribute))
    # standard deviation of this column using np.std method

df = pd.DataFrame(df, index=data_top[2:])
# Now I converted the dictionary to the Data Frame using the pd.DataFrame() function, I will be using 
# the asked attributes as index of the DataFram
print("Question 1 :")
print(df.to_string(), end="\n\n")
# We can print the final dataframe using to_string method to print the full version including all of 
# columns and rows.

# Question 2a
x_axis = data.loc[:, "rain"]
# selecting x axis as the rain column
for column in data_top[2:]:
    if column != "rain":
    # iterating over column names, skipping stationid and date
        y_axis = data.loc[:, column]
        # using the loc method I selected the entire column and used it as the y_axis
        plt.scatter(x_axis, y_axis)
        # scatter plot using plt.scatter
        plt.xlabel(f'rain in {units["rain"]}')
        # labelling the x axis
        plt.ylabel(f"{column} in {units[column]}")
        # labelling the y axis with units
        plt.title(f"Scatter plot b/w rain and {column}")
        # adding title
        plt.xscale("symlog")
        """
        changing the scale of x axis as it is spread b/w 0 and 80,000, so observation and inference would be easier with use of a scale.
        I am using the symmetric log (symlog) scale so it would show the x axis for lower values (<10,000) as a regular log scale will hide the value less than 10,000 due to more range of 80,000 to 10,000. 
        """
        if max(y_axis) > 10000:
            plt.yscale("symlog") # for large y axes I have also used the symlog scale.
        plt.savefig(f"Q2a-{column}.png")
        # saving the figure
        plt.close()
        # to reset the graph to a blank state

# Question 2b
x_axis = data.loc[:, "temperature"]
for column in data_top[2:]:
    if column != "temperature":
        y_axis = data.loc[:, column]
        plt.scatter(x_axis, y_axis)
        plt.title(f"Scatter plot b/w temperature and {column}")
        plt.xlabel(f'temperature in {units["temperature"]}')
        plt.ylabel(f"{column} in {units[column]}")
        if max(y_axis) > 10000:
            plt.yscale("symlog")
        plt.savefig(f"Q2b-{column}.png")
        plt.close()

# Question 3
print("Correlation Coefficents : ")
print(data.corr().loc[:, ["rain", "temperature"]])
# finding the correlation of "rain" and "temperature" by using the corr function to find correlation 
# across every column and printing only the "rain" and "temperature" column using the loc method.
print()

# Question 4
for column in ["rain", "moisture"]:
    hist = data.hist(column=[column], bins=80)
    # Using the hist function we can plot histogram of a column, and we can specify the bins to be used 
    # as 80 for more accurate result.
    # the hist function returns a AxesSubPlot object which can be plotted by matplotlib.
    plt.title(f"histogram of {column}")
    plt.xlabel(f"{column} in {units[column]}")
    plt.savefig(f"Q4-histogram-{column}.png")
    # Labelling and saving the image.
    plt.close()

# Question 5
for stationid, df in data.groupby("stationid"):
    hist = df.hist(column=["rain"], bins=80)
    plt.title(f"histogram of {stationid}")
    plt.xlabel(f'rain in {units["rain"]}')
    plt.savefig(f"Q5-histogram-{stationid}.png")
    plt.close()


# Question 6
# plotting the boxplot of the given columns
for column in ["rain", "moisture"]:
    box = plt.boxplot(data.loc[:, column])
    # Using plt.boxplot we can plot the data, but it will also return data about the whiskers, and the quartiles.
    # The boxplot() will return a dictionary consisting of x and y data it used to make the graph in Lines2D object.
    # We can use the ydata to get the required properties.
    print(f"Boxplot of {column}\n")
    print("Q1, Q3 = ", [x.get_ydata()[0] for x in box["whiskers"]])
    # "whiskers" key will tell us the whiskers in the box plot. It will contain 4 points for the two lines it has to plot for bottom and outer whisker.
    # the y axis will thus tell us the Q1 and Q3 quartile.
    # First element
    print("Median = ", [x.get_ydata()[0] for x in box["medians"]])
    # This will similiarly tell us the Q2 quartile i.e. Median
    print("Caps = ", [x.get_ydata()[0] for x in box["caps"]], "\n")
    # This will tell us the maximum and minimum value the boxplot has gone upto.
    plt.title(f"boxplot of {column}")
    plt.ylabel(f"{column} in {units[column]}")
    if column == "rain" : plt.yscale("symlog")
    plt.grid()
    plt.savefig(f"Q6-boxplot-{column}.png")
    plt.close()

#import libraries
import numpy as np
import pandas as pd


#reading data from URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df =pd.read_csv(URL, names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

#printing dataset information
print(df.info())

#get quick statistcal summary
print(df.describe())

#shows first 10 rows of the dataset
print(df.head(10))

#loc command allows us to access a group of rows and columns
df2 = df.loc[df['sepal_length']>5.0]


#Define  marker shapes by class
import matplotlib.pyplot as plt
marker_shapes = ['.','^','*']
#Then plot the scatterplot
ax = plt.axes()
for i, species in enumerate(df['class'].unique()):
  species_data = df[df['class']== species]
  species_data.plot.scatter(x='sepal_length',
                            y = 'sepal_width',
                            marker = marker_shapes[i],
                            s=100,
                            title = "sepal Width vs Length by Species",
                            label = species,
                            figsize = (10,7),
                            ax = ax)
  
  
  #Plot Histogram
 df['petal_length'].plot.hist(title='Histogram of Petal Length')
  
  
  #Plot BoxPlot to understand the distrubution of the data based on the first quartile, median and the third quartile:
 df.plot.box(title = 'Boxplot of Sepal Length and width, and Petal Length and width')
  
  #Data preprocessing in Pandas
  #1-Encoding categorical variables: Gender, Day, Country ...
 df4 = pd.DataFrame({'Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']})
 print(pd.get_dummies(df4))
  
 # 2-Imputing missing values
URL ="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df5 =pd.read_csv(URL, names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
# randomly select 10 rows
random_index = np.random.choice(df5.index, replace = False, size = 10)
# set the sepal length values of these rows to be None
df5.loc[random_index, 'sepal_length']= None
print(df5.isnull().any())
#another way is to replace the missing value with the mean of the non_missing values
#df5.sepal_length = df5.sepal_length.fillna(df5.sepal_length.mean())
#print(df5.isnull().any())



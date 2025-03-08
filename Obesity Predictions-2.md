In this notebook we perform analysis on an obesity dataset and build several obesity prediction models


```python
import pandas as pd
```


```python
#Importing and previewing the dataset
ob = pd.read_csv(r'/Users/mjones/Desktop/Data Sets/Obesity prediction.csv')
ob.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>family_history</th>
      <th>FAVC</th>
      <th>FCVC</th>
      <th>NCP</th>
      <th>CAEC</th>
      <th>SMOKE</th>
      <th>CH2O</th>
      <th>SCC</th>
      <th>FAF</th>
      <th>TUE</th>
      <th>CALC</th>
      <th>MTRANS</th>
      <th>Obesity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>21.0</td>
      <td>1.62</td>
      <td>64.0</td>
      <td>yes</td>
      <td>no</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>Sometimes</td>
      <td>no</td>
      <td>2.0</td>
      <td>no</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>no</td>
      <td>Public_Transportation</td>
      <td>Normal_Weight</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Female</td>
      <td>21.0</td>
      <td>1.52</td>
      <td>56.0</td>
      <td>yes</td>
      <td>no</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>Sometimes</td>
      <td>yes</td>
      <td>3.0</td>
      <td>yes</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>Sometimes</td>
      <td>Public_Transportation</td>
      <td>Normal_Weight</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>23.0</td>
      <td>1.80</td>
      <td>77.0</td>
      <td>yes</td>
      <td>no</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>Sometimes</td>
      <td>no</td>
      <td>2.0</td>
      <td>no</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Frequently</td>
      <td>Public_Transportation</td>
      <td>Normal_Weight</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>27.0</td>
      <td>1.80</td>
      <td>87.0</td>
      <td>no</td>
      <td>no</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>Sometimes</td>
      <td>no</td>
      <td>2.0</td>
      <td>no</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>Frequently</td>
      <td>Walking</td>
      <td>Overweight_Level_I</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Male</td>
      <td>22.0</td>
      <td>1.78</td>
      <td>89.8</td>
      <td>no</td>
      <td>no</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Sometimes</td>
      <td>no</td>
      <td>2.0</td>
      <td>no</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Sometimes</td>
      <td>Public_Transportation</td>
      <td>Overweight_Level_II</td>
    </tr>
  </tbody>
</table>
</div>




```python
#The models won't work using string values. I use the get_dummies function to convert the binary categories into 0 and 1 values
ob = pd.get_dummies(ob, columns=['Gender', 'family_history', 'SMOKE', 'SCC', 'FAVC'])
```


```python
#Checking the newly formed columns
ob.columns
```




    Index(['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CAEC', 'CH2O', 'FAF', 'TUE',
           'CALC', 'MTRANS', 'Obesity', 'Gender_Female', 'Gender_Male',
           'family_history_no', 'family_history_yes', 'SMOKE_no', 'SMOKE_yes',
           'SCC_no', 'SCC_yes', 'FAVC_no', 'FAVC_yes'],
          dtype='object')




```python
#get_dummies() splits the columns into two. I now drop the additional ones that are no longer necessary
ob = ob.drop(columns=['Gender_Male', 'SMOKE_yes', 'family_history_yes', 'SCC_no', 'FAVC_no'])
```


```python
#Previewing the dataset after the transformations
ob.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>FCVC</th>
      <th>NCP</th>
      <th>CAEC</th>
      <th>CH2O</th>
      <th>FAF</th>
      <th>TUE</th>
      <th>CALC</th>
      <th>MTRANS</th>
      <th>Obesity</th>
      <th>Gender_Female</th>
      <th>family_history_no</th>
      <th>SMOKE_no</th>
      <th>SCC_yes</th>
      <th>FAVC_yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21.0</td>
      <td>1.62</td>
      <td>64.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>Sometimes</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>no</td>
      <td>Public_Transportation</td>
      <td>Normal_Weight</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21.0</td>
      <td>1.52</td>
      <td>56.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>Sometimes</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>Sometimes</td>
      <td>Public_Transportation</td>
      <td>Normal_Weight</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>23.0</td>
      <td>1.80</td>
      <td>77.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>Sometimes</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Frequently</td>
      <td>Public_Transportation</td>
      <td>Normal_Weight</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>27.0</td>
      <td>1.80</td>
      <td>87.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>Sometimes</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>Frequently</td>
      <td>Walking</td>
      <td>Overweight_Level_I</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22.0</td>
      <td>1.78</td>
      <td>89.8</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Sometimes</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Sometimes</td>
      <td>Public_Transportation</td>
      <td>Overweight_Level_II</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Next I need to change the other string columns. Since they have several options, the get_dummies function won't work. I need to explicitly assign new values. I start by using the unique() function to see the values that I will need to change. Then using the map() function to change the values into numeric types


```python
ob['CAEC'].unique()
```




    array(['Sometimes', 'Frequently', 'Always', 'no'], dtype=object)




```python
ob['CALC'].unique()
```




    array(['no', 'Sometimes', 'Frequently', 'Always'], dtype=object)




```python
ob['CAEC']= ob['CAEC'].map({'Sometimes':1, 'Frequently':2, 'Always':3, 'no':4})
```


```python
ob['CALC']= ob['CALC'].map({'Sometimes':1, 'Frequently':2, 'Always':3, 'no':4})
```


```python
ob['MTRANS'].unique()
```




    array(['Public_Transportation', 'Walking', 'Automobile', 'Motorbike',
           'Bike'], dtype=object)




```python
ob['MTRANS']= ob['MTRANS'].map({'Public_Transportation':1, 'Walking':2, 'Automobile':3, 'Motorbike':4, 'Bike':5})
```


```python
ob['Obesity'] = ob['Obesity'].map({'Normal_Weight':1, 'Overweight_Level_I':2, 'Overweight_Level_II':3, 'Obesity_Type_I':4, 'Obesity_Type_II':5, 'Obesity_Type_III':6, 'Insufficient_Weight':7})
```


```python
#Previewing the transformed dataset
ob.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>FCVC</th>
      <th>NCP</th>
      <th>CAEC</th>
      <th>CH2O</th>
      <th>FAF</th>
      <th>TUE</th>
      <th>CALC</th>
      <th>MTRANS</th>
      <th>Obesity</th>
      <th>Gender_Female</th>
      <th>family_history_no</th>
      <th>SMOKE_no</th>
      <th>SCC_yes</th>
      <th>FAVC_yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21.0</td>
      <td>1.62</td>
      <td>64.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21.0</td>
      <td>1.52</td>
      <td>56.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>23.0</td>
      <td>1.80</td>
      <td>77.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>27.0</td>
      <td>1.80</td>
      <td>87.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22.0</td>
      <td>1.78</td>
      <td>89.8</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#I want the Obesity (the predictor) column at the end
column_to_move = ob.pop('Obesity')
ob.insert(16,'Obesity', column_to_move)
```


```python
ob.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>FCVC</th>
      <th>NCP</th>
      <th>CAEC</th>
      <th>CH2O</th>
      <th>FAF</th>
      <th>TUE</th>
      <th>CALC</th>
      <th>MTRANS</th>
      <th>Gender_Female</th>
      <th>family_history_no</th>
      <th>SMOKE_no</th>
      <th>SCC_yes</th>
      <th>FAVC_yes</th>
      <th>Obesity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21.0</td>
      <td>1.62</td>
      <td>64.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21.0</td>
      <td>1.52</td>
      <td>56.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>23.0</td>
      <td>1.80</td>
      <td>77.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>27.0</td>
      <td>1.80</td>
      <td>87.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22.0</td>
      <td>1.78</td>
      <td>89.8</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Previewing the columns in my dataset in preparation to assign the features and the predictor variables
ob.columns
```




    Index(['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CAEC', 'CH2O', 'FAF', 'TUE',
           'CALC', 'MTRANS', 'Gender_Female', 'family_history_no', 'SMOKE_no',
           'SCC_yes', 'FAVC_yes', 'Obesity'],
          dtype='object')



Next I assign the features and the predictor variables


```python
features = ['Age', 'Height', 'FCVC', 'NCP', 'CAEC', 'CH2O', 'FAF', 'TUE',
       'CALC', 'MTRANS', 'Gender_Female', 'family_history_no', 'SMOKE_no',
       'SCC_yes', 'FAVC_yes']
```


```python
predictor = ['Obesity']
```


```python
X = ob.loc[:, features].values
```


```python
y = ob.loc[:, predictor].values
```

Next I import the standardscaler and principal component analysis packages to standardize my features as well as use PCA to minimize the number of features.


```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```


```python
X = StandardScaler().fit_transform(X)
```

I elect to use 9 principal components


```python
pca = PCA(n_components=9)
principalComponents = pca.fit_transform(X)
principalOb = pd.DataFrame(data=principalComponents, columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9'])
```


```python
principalOb.shape
```




    (2111, 9)



I create a new dataframe called finalOB using the principal components and concatenating the obesity column onto the end


```python
finalOb = pd.concat([principalOb, ob[['Obesity']]], axis=1)
```


```python
finalOb.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pc1</th>
      <th>pc2</th>
      <th>pc3</th>
      <th>pc4</th>
      <th>pc5</th>
      <th>pc6</th>
      <th>pc7</th>
      <th>pc8</th>
      <th>pc9</th>
      <th>Obesity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.625618</td>
      <td>-0.052671</td>
      <td>0.022598</td>
      <td>-0.849863</td>
      <td>-1.751855</td>
      <td>0.379976</td>
      <td>-0.192195</td>
      <td>-0.216208</td>
      <td>-2.233655</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.275378</td>
      <td>-1.093413</td>
      <td>3.084155</td>
      <td>4.356404</td>
      <td>-1.103146</td>
      <td>5.234845</td>
      <td>3.919024</td>
      <td>0.879715</td>
      <td>-0.248579</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.827578</td>
      <td>-1.622884</td>
      <td>1.154246</td>
      <td>-0.237181</td>
      <td>-0.925342</td>
      <td>0.356082</td>
      <td>-0.399949</td>
      <td>-0.738155</td>
      <td>-0.911315</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.120763</td>
      <td>-0.509067</td>
      <td>2.650700</td>
      <td>1.289046</td>
      <td>-0.341985</td>
      <td>-0.116386</td>
      <td>-1.141974</td>
      <td>-1.748218</td>
      <td>0.245270</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.093932</td>
      <td>-0.256477</td>
      <td>1.391925</td>
      <td>-0.777077</td>
      <td>1.600446</td>
      <td>-0.609013</td>
      <td>0.416862</td>
      <td>-1.141189</td>
      <td>0.182193</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
finalOb['Obesity'].unique()
```




    array([1, 2, 3, 4, 7, 5, 6])



Next I import the models I will be using as well as the train_test_split and accuracy score modules.


```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
```

Below I create and fit the models


```python
clf = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=0)
ada = AdaBoostClassifier(random_state=0)
```


```python
X = finalOb[['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9']]
y = finalOb['Obesity']
```


```python
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3)
```


```python
clf.fit(X_train, y_train.values.ravel())
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DecisionTreeClassifier(random_state=0)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier(random_state=0)</pre></div></div></div></div></div>




```python
rfc.fit(X_train, y_train.values.ravel())
```




<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(random_state=0)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(random_state=0)</pre></div></div></div></div></div>




```python
ada.fit(X_train, y_train.values.ravel())
```




<style>#sk-container-id-3 {color: black;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>AdaBoostClassifier(random_state=0)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">AdaBoostClassifier</label><div class="sk-toggleable__content"><pre>AdaBoostClassifier(random_state=0)</pre></div></div></div></div></div>



Below I add the predictive models to a list and then use a for-loop to iteratively run each model and display the accuracy score for each.


```python
models = [clf, rfc, ada]
```


```python
for model in models:
    model.fit(X_train, y_train.values.ravel())
    predictions = model.predict(X_test)
    accuracy_scores = accuracy_score(y_test, predictions)
    print(f"{model} score: {accuracy_scores}")
```

    DecisionTreeClassifier(random_state=0) score: 0.6656151419558359
    RandomForestClassifier(random_state=0) score: 0.7665615141955836
    AdaBoostClassifier(random_state=0) score: 0.38801261829652994



```python

```

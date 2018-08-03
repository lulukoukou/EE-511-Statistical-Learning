from statistics import mean
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# given features
numerical_variables = ['Lot Area', 'Lot Frontage', 'Year Built', 'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2','Bsmt Unf SF', 'Total Bsmt SF', '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area','Garage Area', 'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch', 'Pool Area']

discrete_variables = ['MS SubClass', 'MS Zoning', 'Street', 'Alley', 'Lot Shape', 'Land Contour', 'Utilities', 'Lot Config', 'Land Slope', 'Neighborhood', 'Condition 1', 'Condition 2', 'Bldg Type', 'House Style', 'Overall Qual', 'Overall Cond', 'Roof Style', 'Roof Matl', 'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type', 'Exter Qual', 'Exter Cond', 'Foundation', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'Heating', 'Heating QC', 'Central Air', 'Electrical', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Full Bath', 'Half Bath', 'Bedroom AbvGr', 'Kitchen AbvGr', 'Kitchen Qual', 'TotRms AbvGrd', 'Functional', 'Fireplaces', 'Fireplace Qu', 'Garage Type', 'Garage Cars', 'Garage Qual', 'Garage Cond', 'Paved Drive', 'Pool QC', 'Fence', 'Sale Type', 'Sale Condition']

# data pre-processing
flag = True
table = []
temp = []
feature = []
sale_price_train = []
sale_price_validation = []
sale_price_test = []
slice = 1
for line in open('AmesHousing.txt'):
    if flag:
        for item in line.split('\t'):
            feature.append(item.strip('\n'))
        flag = False
    else:
        temp = []
        index = 0
        for item in line.split('\t'):
            if feature[index] in numerical_variables:
                if not item.strip():
                    temp.append(0)
                else:
                    temp.append(item.strip('\n'))
            elif feature[index] in discrete_variables:
                if not item.strip():
                    temp.append('NaN')
                else:
                    temp.append(item.strip('\n'))
            else:
                temp.append(item.strip('\n'))
            index += 1
        table.append(temp)
        if slice%5==3:
            sale_price_validation.append(item.strip('\n'))
        elif slice%5==4:
            sale_price_test.append(item.strip('\n'))
        else:
            sale_price_train.append(item.strip('\n'))
        slice += 1
        
df = pd.DataFrame(table)
df.columns = feature
feature_copy = []
for f in feature:
    if (f not in numerical_variables) and (f not in discrete_variables):
        del df[f]
    else:
        feature_copy.append(f)
feature = feature_copy
# df includes data corresponding to features
# separate to training, validation and test set
train = []
test = []
validation = []
new_table = df.values.tolist()

for line in range(df.shape[0]):
    index = int(table[line][0])
    if index % 5 == 3:
        validation.append(new_table[line])
        
    elif index % 5 == 4:
        test.append(new_table[line])
    else:
        train.append(new_table[line])

df_train = pd.DataFrame(train)
df_train.columns = feature
df_validation = pd.DataFrame(validation)
df_validation.columns = feature
df_test = pd.DataFrame(test)
df_test.columns = feature


#one variable least squares linear regression model

Gr_Liv_Area_train = np.array(df_train['Gr Liv Area'],dtype=np.float64)
Gr_Liv_Area_validation = np.array(df_validation['Gr Liv Area'],dtype=np.float64)
Gr_Liv_Area_test = np.array(df_test['Gr Liv Area'],dtype=np.float64)
sale_price_train = np.array(sale_price_train,dtype=np.float64)
sale_price_validation = np.array(sale_price_validation,dtype=np.float64)
sale_price_test = np.array(sale_price_test,dtype=np.float64)

def best_fit_slope_and_intercept(x,y):
    m = (mean(x)*mean(y)-mean(x*y))/((mean(x)*mean(x))-(mean(x**2)))
    b = mean(y) - m*mean(x)
    return m,b

def rmse(predict, y):
    return np.sqrt(mean((predict - y) ** 2))

m, b = np.round(best_fit_slope_and_intercept(Gr_Liv_Area_train, sale_price_train),decimals=3)
reg_line_train= [(m*x)+b for x in Gr_Liv_Area_train]
plt.figure(1)
plt.xlabel('Gr_Liv_Area')
plt.ylabel('Sale_Price')
plt.title('One Variable Least Squares Model\n')
plt.scatter(Gr_Liv_Area_train, sale_price_train,s=1,color='purple')
plt.plot(Gr_Liv_Area_train, reg_line_train,color='yellow')
plt.text(4000, 500000, 'y=%s*x+%s'%(m,b))
plt.show()

reg_line_validation = [(m*x)+b for x in Gr_Liv_Area_validation]
RMSE_validation = rmse(reg_line_validation, sale_price_validation)
print ("The RMSE(Gr_Liv_Area feature)for validation is %s" %RMSE_validation)

#multiple variabes least squares model
train = []
validation = []
test = []
df = pd.get_dummies(df, columns=discrete_variables, drop_first=True)
new_table = df.values.tolist()
for line in range(df.shape[0]):
    index = int(table[line][0])
    if index % 5 == 3:
        validation.append(new_table[line])   
    elif index % 5 == 4:
        test.append(new_table[line])
    else:
        train.append(new_table[line])

df_train = pd.DataFrame(train)
df_train.columns = df.columns
df_validation = pd.DataFrame(validation)
df_validation.columns = df.columns
df_test = pd.DataFrame(test)
df_test.columns = df.columns

train = df_train.as_matrix(columns=None)
validation = df_validation.as_matrix(columns=None)
test = df_test.as_matrix(columns=None)

reg = LinearRegression()
reg.fit(train,sale_price_train)
reg_predict_validation = reg.predict(validation)
RMSE_validation = rmse(reg_predict_validation, sale_price_validation)
print ("The RMSE(all features) for validation is %s" %RMSE_validation)

#LASSO regularization
#feature normalization
mean_train = np.asarray(df_train, dtype=np.float).mean(axis=0)
std_train= np.asarray(df_train, dtype=np.float).std(axis=0)
std_train[std_train == 0]=1
data_train = np.asarray(df_train, dtype=np.float64)
data_validation = np.asarray(df_validation, dtype=np.float64)
data_test = np.asarray(df_test, dtype=np.float64)

norm_train = (data_train - mean_train)/(std_train)
norm_validation = (data_validation - mean_train)/(std_train)
norm_test = (data_test - mean_train)/(std_train)

#cross validation for alpha
norm_train_validation = np.append(norm_train,norm_validation,axis=0)
new_sale_price = np.append(sale_price_train,sale_price_validation,axis=0)
  
average = 0
first = True
k_folder = 4
step = len(norm_train_validation)/k_folder
average_w = []
RMSE_average_test = []
RMSE_average_train = []
RMSE_average_validation = []
total_w = np.zeros(shape=reg.coef_.shape, dtype=np.float64)
for i in range(50,501,50):
    reg_lasso = Lasso(alpha=i,fit_intercept=True, tol=0.00036)
    k = 0
    first = True
    RMSE_test = []
    RMSE_train = []
    RMSE_validation = []
    for j in range(k_folder):
        reg_lasso.fit(norm_train_validation[k:k+586],new_sale_price[k:k+586])
        if first:
            total_w = np.zeros(shape=reg_lasso.coef_.shape, dtype=np.float64)
            first = False
        total_w = total_w + reg_lasso.coef_
        k = k + 586
        RMSE_train.append(rmse(reg_lasso.predict(norm_train),sale_price_train))
        RMSE_validation.append(rmse(reg_lasso.predict(norm_validation),sale_price_validation))
        RMSE_test.append(rmse(reg_lasso.predict(norm_test),sale_price_test))

    RMSE_average_train.append(sum(RMSE_train)/float(k_folder))
    RMSE_average_validation.append(sum(RMSE_validation)/float(k_folder))
    RMSE_average_test.append(sum(RMSE_test)/float(k_folder))

    average = total_w/float(k_folder)
    average_w.append(average)

nonzero_w = []
Alpha = [x for x in range(50,501,50)]
for index in range(len(average_w)):
    nonzero_w.append(np.count_nonzero(average_w[index]))
    
plt.figure(2)
plt.xlabel('Alpha')
plt.ylabel('RMSE')
plt.title('RMSE for each Alpha')
plt.plot(Alpha,RMSE_average_train, label="train",linewidth=1.0)
plt.scatter(Alpha,RMSE_average_train,s=5)
plt.plot(Alpha,RMSE_average_validation, label="validation",linewidth=1.0)
plt.scatter(Alpha,RMSE_average_validation,s=5)
plt.legend(["train","validation"])
plt.show()


plt.figure(3)
plt.xlabel('Alpha')
plt.ylabel('Number of nonzero coefficients')
plt.title('Number of nonzero coefficients for each Alpha\n')
plt.scatter(Alpha,nonzero_w)
plt.show()

# final step for test set
reg_line_test = [(m*x)+b for x in Gr_Liv_Area_test]
RMSE_test = rmse(reg_line_test, sale_price_test)
print ("The RMSE(Gr_Liv_Area feature)for test is %s" %RMSE_test)

reg_predict_test = reg.predict(test)
RMSE_test = rmse(reg_predict_test, sale_price_test)
print ("The RMSE(all features) for test is %s" %RMSE_test)

plt.figure(4)
plt.xlabel('Alpha')
plt.ylabel('RMSE')
plt.title('RMSE for each Alpha')
plt.plot(Alpha,RMSE_average_test, label="test",linewidth=1.0)
plt.scatter(Alpha,RMSE_average_test, s=5)
plt.legend(["test"])
plt.show()

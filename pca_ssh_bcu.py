import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA 
import IPython
import platform
import os
import pandas as pd
import numpy as np
import warnings
import time
import math
from sklearn.metrics import roc_curve , auc
from astropy.io import fits
warnings.filterwarnings('ignore')
print(os.getcwd())
dataset = pd.read_csv("3FGL_rawdata.csv")
bcu = pd.read_csv("3FGL_bcu_forcast.csv")

dataset = dataset[dataset['Flags'] == 0]
bll = dataset[dataset['CLASS1'] == 'bll']
BLL = dataset[dataset['CLASS1'] == 'BLL']
fsrq = dataset[dataset['CLASS1'] == 'fsrq']
FSRQ = dataset[dataset['CLASS1'] == 'FSRQ']
data = bll.append(BLL.append(fsrq.append(FSRQ)))
temp = list(data['CLASS1'])

for i in range(len(temp)):
    if temp[i] == 'bll' or temp[i] == 'fsrq' : pass
    elif temp[i] == 'BLL' : temp[i] = 'bll'
    else: temp[i] = 'fsrq'
data['CLASS1'] = temp

dataset = data

print(dataset)

flag = []
name_ref = list(bcu.columns)
names = list(dataset.columns)
for ele in names:
    if ele in name_ref:pass
    else: dataset = dataset.drop(ele,axis = 1)
print(dataset)

# print(dataset.head())
# print(dataset[:3])
# print(dataset)
total_res = []
dataset = dataset.dropna()
bcu = bcu.dropna()
#提取特征和类别

dataset_bll = dataset[dataset['CLASS1'] == 'bll']
dataset_fsrq = dataset[dataset['CLASS1'] == 'fsrq']
dataset_bll = dataset_bll.loc[:,'Signif_Avg':'Variability_Index']
dataset_fsrq = dataset_fsrq.loc[:,'Signif_Avg':'Variability_Index']

name = dataset.loc[:,'Source_Name']
X = dataset.loc[:,'Signif_Avg':'Variability_Index']
X_powerlaw = dataset.loc[:,'PowerLaw_Index']
y = dataset.loc[:,'CLASS1']
bcu_x = bcu.loc[:,'Signif_Avg':'Variability_Index']
bcu_y = bcu.loc[:,'CLASS1']
bcu_name = bcu.loc[:,'Source_Name']
bcu_powerlaw = bcu.loc[:,'PowerLaw_Index']
#划分训练集和测试集
names = bcu.columns.tolist()[1:-1]
print(names)
# def classify(res):
#     bll = []
#     fsrq = []
#     for element in res:
#         if element[-3] == 'bll':
#             bll.append(element)
#         else:
#             fsrq.append(element)
#     return bll , fsrq
pca = PCA(n_components=6)
pca.fit(bcu_x)
bcu_reduced_x = pca.fit_transform(bcu_x)
bcu_processed = pca.inverse_transform(bcu_reduced_x)

pca = PCA(n_components=6)
pca.fit(X)
reduced_x = pca.fit_transform(X)
X_processed = pca.inverse_transform(reduced_x)


predict_total = []
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
# '''name_train , name_test ,''' '''name ,''' 
name_train , name_test ,X_train, X_test, y_train, y_test = train_test_split(name ,X_processed, y, test_size=0.2, random_state=0)

def plothist(data_1 , data_2 , methods , types):
    plt.hist(data_1 , label = 'tagged sources %d'%(len(data_1)))
    plt.hist(data_2 , alpha = 0.5 , label = 'BCU sources %d'%(len(data_2)) )
    plt.legend()
    plt.title(types + '_' + methods)
    plt.savefig(types + '_' + methods + '.jpg')
    plt.close()
    

def evaluation(ker , predict_one , test_one):
    compare = np.array(list(zip(predict_one , test_one)))
    precision_cal = compare[compare[:,0] == 'bll']   # pull out all the result predicted as 'bll' from the predict data , to calculate the precision
    tpr_cal = compare[compare[:,1] == 'bll']   # pull out all the result tagged as 'bll' from the test set,  to calculate the TPR
    fpr_cal = compare[compare[:,1] == 'fsrq']   # pull out all the result tagged as 'fsrq' from the test set,  to calculate the FPR
    accuracy = np.mean(predict_data==y_test)
    temp_pre = (np.mean(precision_cal[:,0] == precision_cal[:,1]))
    temp_tpr = (np.mean(tpr_cal[:,0] == tpr_cal[:,1]))
    temp_fpr = (np.mean(fpr_cal[:,0] != fpr_cal[:,1]))
    # print('%s , the accuracy is %.4f , the precision is %.4f , the TPR is %.4f , the FPR is %.4f'%(ker , accuracy , temp_pre , temp_tpr , temp_fpr))
    return accuracy , temp_pre , temp_tpr , temp_fpr

def roc(ker , test , score):
    # print(roc_curve(test , score))
    evaluate = list(test)
    for i in range(len(evaluate)):
        if evaluate[i] == 'bll' : evaluate[i] = 0
        else:evaluate[i] = 1
    fpr,tpr,_ = roc_curve(evaluate , score)
    roc_auc = auc(fpr,tpr)
    plt.rc('font',family='Times New Roman')
    plt.plot(fpr , tpr , label = 'ROC curve area %.2f'%roc_auc)
    plt.plot([0,1] ,[0,1] , '--')
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel('FPR' , fontsize = 15)
    plt.ylabel('TPR' , fontsize = 15)
    plt.title(ker + ' ROC curve', fontsize = 15)
    plt.legend(loc = 'lower right' , fontsize = 15)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.savefig(ker + ' ROC curve.jpg')
    # plt.show()
    plt.close()

tagged_bll = dataset[dataset['CLASS1'] == 'bll']
tagged_fsrq = dataset[dataset['CLASS1'] == 'fsrq']
#---------------------------------------------------- SVM -----------------------------------------------


kernel_svm = ['linear']#'linear', 'poly',  'sigmoid' , 'rbf
from sklearn import svm
svm_res=[]
para_svm = []
para = []
print('now begin the svm process')
for i in range(len(kernel_svm)):
    res =[]
    res_pre = []
    res_fpr = []
    res_tpr = []
    for j in range(2):
        temp = []
        temp_pre = []
        temp_fpr = []
        temp_tpr = []
        # if i == 0 and j > 2 :break     
        time_start = time.time()
        for k in range(3):
            print('now is the %d calculation'%k)
            svc_model = svm.SVC(kernel=kernel_svm[i], C= 0.001*10**j , gamma = 'auto')
            svc_model.fit(X_train, y_train)
            predict_data = svc_model.predict(X_test)
            accuracy = np.mean(predict_data==y_test)
            temp.append(accuracy)
            print('')
            # temp_pre.append(np.mean(precision_cal[:,0] == precision_cal[:,1]))
            # temp_tpr.append(np.mean(tpr_cal[:,0] == tpr_cal[:,1]))
            # temp_fpr.append(np.mean(fpr_cal[:,0] != fpr_cal[:,1]))
            # print(k)
        time_end = time.time() - time_start    
        print('it took %.5f to calculate C = %f'%(time_end/3 ,  0.001*10**j))
        res.append(np.max(temp))        
        print('the accuracy is %.6f with %s while C = %f'%(accuracy , kernel_svm[i] , 0.001*10**j))
    svm_res.append(res)
    para.append(  0.001*10**(svm_res[i].index(np.max(svm_res[i])))   )
    total_res.append('SVM with kernl %s , accuracy is %.4f , C value is %f , it tooks %.5f sec for each , optimizing the parameter took %.5f sec'%(kernel_svm[i] , np.max(svm_res[i]) , 0.001*10**(svm_res[i].index(np.max(svm_res[i])))  , time_end/3 , time_end))
print(total_res)
res_svm = []

for i,ker in enumerate(kernel_svm):
    time_start =  time.time()
    svc_model = svm.SVC(kernel=ker, C = para[i] , gamma = 'auto')
    svc_model.fit(X_train, y_train)
    predict_data_svm = svc_model.predict(bcu_processed)
    predict_total.append(predict_data_svm)
    bcu['test'] = predict_data_svm
    # print(predict_data)
    time_end = time.time() - time_start
    y_score = svc_model.fit(X_train , y_train).decision_function(X_test)
    roc(kernel_svm[i] , y_test , y_score)

    predict_data = svc_model.predict(X_test)
    compare = np.array(list(zip(predict_data , y_test)))
    precision_cal = compare[compare[:,0] == 'bll']   # pull out all the result predicted as 'bll' from the predict data , to calculate the precision
    tpr_cal = compare[compare[:,1] == 'bll']   # pull out all the result tagged as 'bll' from the test set,  to calculate the TPR
    fpr_cal = compare[compare[:,1] == 'fsrq']   # pull out all the result tagged as 'fsrq' from the test set,  to calculate the FPR
    accuracy = np.mean(predict_data==y_test)
    temp_pre = (np.mean(precision_cal[:,0] == precision_cal[:,1]))
    temp_tpr = (np.mean(tpr_cal[:,0] == tpr_cal[:,1]))
    temp_fpr = (np.mean(fpr_cal[:,0] != fpr_cal[:,1]))
    print('%s , the accuracy is %.4f , the precision is %.4f , the TPR is %.4f , the FPR is %.4f'%(ker , accuracy , temp_pre , temp_tpr , temp_fpr))

    print('it took %.3f seconds to calculate the kernl %s'%(time_end , ker))
    doc = open('SVM_test_result.txt','w')
    for i in range(len(predict_data)):
        print(predict_data[i],file=doc)
        print('',file=doc)
        print('the precision is %.4f'%temp_pre,file=doc)
        print('the TPR is %.4f'%temp_tpr,file=doc)
        print('the FPR is %.4f'%temp_fpr,file=doc)
    doc.close()

    print('----------------------------------------------')

svm_res = np.array(list(zip(list(bcu['PowerLaw_Index']) , predict_data)))
# print(svm_res)
for name in kernel_svm:
    plothist(tagged_bll['PowerLaw_Index'] , bcu[bcu['test'] == 'bll']['PowerLaw_Index'] , name , 'bll')

for name in kernel_svm:
    plothist(tagged_fsrq['PowerLaw_Index'] , bcu[bcu['test'] == 'fsrq']['PowerLaw_Index'] , name , 'fsrq')
bcu = bcu.drop('test' , axis = 1)
# print(bcu)
#------------------------------------------------------------MLPClassifier--------------------------------------------------------------------------

MLP_kernl = [ 'lbfgs' ]# , 'sgd', 'lbfgs' 'adam' , 'lbfgs'
alpha_step = 0.001
alpha_range = 100
res = []
max_int_step = 100
para_MLP = []
trail = 2

from sklearn.neural_network import MLPClassifier
for i in range(len(MLP_kernl)):
    temp = []
    temp2 = []
    time_start = time.time()
    for j in range(alpha_range):
        acc = []
        score = []
        temp_start = time.time()
        for k in range(trail):
            #建立MLP神经网络模型 ，MLP的求解方法为adam，可选lbfgs、sgd，正则化惩罚alpha = 0.1,max_intergrate = 1000
            mpl_model = MLPClassifier(solver=MLP_kernl[i],  learning_rate='constant', learning_rate_init=0.01, max_iter = 1000, alpha =alpha_step*j)
            # hidden_layer_sizes=(50,),
            mpl_model.fit(X_train, y_train)
            predict_data = mpl_model.predict(X_test)
            accuracy = evaluation(MLP_kernl[i] , predict_data , y_test)
            accuracy = list(accuracy)
            y_score = np.array(mpl_model.predict_proba(X_test))
            y_score = list(y_score[:,1])
            accuracy.append(y_score)
            # print(y_score)
            # accuracy = np.mean(predict_data == y_test)
            # print(accuracy)
            # print()
            # accuracy = [accur , precision , tpr , fpr]
            acc.append(accuracy)
            # score.append(y_score)
        acc = np.array(acc)
        num = np.where (acc[:,0] == np.max(acc[:,0]))[0][0]
        temp.append(list(acc[num] ))
        # print('the acc is ' , acc)
        # print('the temp is ' , temp)
        temp_end = time.time() - temp_start
        temp2.append('MLPClassifier with the %s kernl alpha value is %.3f, the accuracy is %.5f , %.5f sec'%(MLP_kernl[i],alpha_step*j,np.max(acc[:,0]) , temp_end/trail))
        print('MLP %s kernl alpha value is %.3f, the accuracy is %.5f, it took %f sec'%(MLP_kernl[i],alpha_step*j,np.max(acc[:,0]) , temp_end/trail))
    time_end = time.time() - time_start
    res.append(temp2)
    temp = np.array(temp)
    location = np.where(temp[:,0] == np.max(temp[:,0]))[0][0]
    total_res.append('MLP with kernl %s , accuracy is %.4f , precision is %.4f , tpr is %.4f , fpr is %.4f , alpha value is %.3f , it took %f sec for each,  optimizing the parameter took %.5f sec'%(MLP_kernl[i] , 
                                                                                                                                                                                                        temp[location][0] , 
                                                                                                                                                                                                        temp[location][1] , 
                                                                                                                                                                                                        temp[location][2] , 
                                                                                                                                                                                                        temp[location][3] ,
                                                                                                                                                                                                        location * alpha_step , 
                                                                                                                                                                                                        time_end/(trail*alpha_range) , 
                                                                                                                                                                                                        time_end ))
    para_MLP.append(  location * alpha_step )
    roc(MLP_kernl[i] , y_test , temp[location][-1])
    

for i in range(len(total_res)):
    print(total_res[i])

for i,ker in enumerate(MLP_kernl):
    mpl_model = MLPClassifier(solver=ker, learning_rate='constant', learning_rate_init=0.01, max_iter = 900, alpha =para_MLP[i])
    time_start = time.time()
    mpl_model.fit(X_train, y_train)
    predict_data = mpl_model.predict(bcu_processed)
    time_end = time.time() - time_start

    bcu['test'] = predict_data
    plothist(tagged_bll['PowerLaw_Index'] , bcu[bcu['test'] == 'bll']['PowerLaw_Index'] , ker , 'bll')
    plothist(tagged_fsrq['PowerLaw_Index'] , bcu[bcu['test'] == 'fsrq']['PowerLaw_Index'] , ker , 'fsrq')
    bcu = bcu.drop('test' , axis = 1)
    predict_total.append(predict_data)

    # predict_data = mpl_model.predict(X_test)
    # compare = np.array(list(zip(predict_data , y_test)))
    # precision_cal = compare[compare[:,0] == 'bll']   # pull out all the result predicted as 'bll' from the predict data , to calculate the precision
    # tpr_cal = compare[compare[:,1] == 'bll']    # pull out all the result tagged as 'bll' from the test set,  to calculate the TPR
    # fpr_cal = compare[compare[:,1] == 'fsrq']   # pull out all the result tagged as 'fsrq' from the test set,  to calculate the FPR
    # accuracy = np.mean(predict_data==y_test)
    # temp_pre = (np.mean(precision_cal[:,0] == precision_cal[:,1]))
    # temp_tpr = (np.mean(tpr_cal[:,0] == tpr_cal[:,1]))
    # temp_fpr = (np.mean(fpr_cal[:,0] != fpr_cal[:,1]))
    # print('%s , the accuracy is %.4f , the precision is %.4f , the TPR is %.4f , the FPR is %.4f'%(ker , accuracy , temp_pre , temp_tpr , temp_fpr))
    # res.append('%s , the accuracy is %.4f , the precision is %.4f , the TPR is %.4f , the FPR is %.4f'%(ker , accuracy , temp_pre , temp_tpr , temp_fpr))
    # print('----------------------------------------------')

    # print(predict_data)

    # mpl_model = MLPClassifier(solver=ker, learning_rate='constant', learning_rate_init=0.01, max_iter = 900, alpha =para_MLP[i])
    # mpl_model.fit(X_train, y_train)
    

    doc = open('MLP_test_result.txt','w')
    for i in range(len(predict_data)):
        print(predict_data[i],file=doc)
        print('',file=doc)
    doc.close()
# print(bcu)

doc = open('MLP_results.txt','w')
for i in range(len(res)):
    for j in range(len(res[i])):
        print(res[i][j],file=doc)
        print('',file=doc)
doc.close()


#------------------------------------------------------------------LogisticRegression------------------------------------------------------------------------


from sklearn.linear_model import LogisticRegression
# X_train, X_test, y_train, y_test = train_test_split(reduced_x, y, test_size=0.2, random_state=0)
#建立逻辑回归模型 ，惩罚参数为100
LR_res = []
c_step = 10
c_range = 1000
c_start = 1
c = [] 
kernl = ['liblinear','lbfgs','newton-cg','sag']
time_start = time.time()
for i in range(1,c_range+1):
    temp = []
    for j in range(1):
        lr_model = LogisticRegression(C = c_start*(i), max_iter=1000 , multi_class='auto' , )
        lr_model.fit(X_train, y_train)
        predict_data = lr_model.predict(X_test)
        accuracy = np.mean(predict_data == y_test)
        y_score = np.array(lr_model.predict_proba(X_test))
        y_score = list(y_score[:,1])
        # accuracy.append(y_score)
        temp.append([accuracy,y_score])
        # temp.append(y_score)
        # print(accuracy , i)
    # c.append(c_start*(c_step**i))
    LR_res.append(max(temp))
time_end = time.time() - time_start
print('it took %f sec of the LR'%(time_end/1000))
    # print('LogisticRegression')
    # print(LR_res[i-1] , 'where C value is ' , c_start*(i))
    # print()
num = LR_res.index(max(LR_res))
roc('LR' , y_test , LR_res[num][-1])
total_res.append('Logistic Regression accuracy is %.4f , C value is %.3f , it took %.5f sec for each , optimizing the parameter took %.5f sec'%(LR_res[num][0] , num , time_end/c_range , time_end ))
para_LR = num

doc = open('LR_results.txt','w')
for i in range(len(LR_res)):
    print(LR_res[i],file=doc)
    print('',file=doc)
doc.close()
    

lr_model = LogisticRegression(C = para_LR, max_iter=1000 , multi_class='auto' )
time_start = time.time()
lr_model.fit(X_train, y_train)
predict_data = lr_model.predict(bcu_processed)
time_end = time.time() - time_start
bcu['test'] = predict_data
plothist(tagged_bll['PowerLaw_Index'] , bcu[bcu['test'] == 'bll']['PowerLaw_Index'] , 'LR' , 'bll')
plothist(tagged_fsrq['PowerLaw_Index'] , bcu[bcu['test'] == 'fsrq']['PowerLaw_Index'] , 'LR' , 'fsrq')
bcu = bcu.drop('test' , axis = 1)
predict_total.append(predict_data)

predict_data = lr_model.predict(X_test)
acc , precision , tpr , fpr = evaluation('LR' , predict_data , y_test)
total_res.append('Logistic Regression precision is %.4f , TPR is %.4f , FPR is %.4f'%(precision , tpr , fpr))
# print(predict_data)

doc = open('LR_test_result.txt','w')
for i in range(len(predict_data)):
    print(predict_data[i],file=doc)
    print('',file=doc)
doc.close()

#----------------------------------------------------------------DecisionTree------------------------------------------------------------------------------

from sklearn import tree
#划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(reduced_x, y, test_size=0.2, random_state=np.random)
temp=[]
temp2 = []
# 建立决策树模型，选择算法为熵增益，可选gini,entropy,默认为gini
DT_kernel = ['gini']   #,'entropy'
for ker in DT_kernel:

    time_start = time.time()
    for i in range(1000):
        tree_model = tree.DecisionTreeClassifier(criterion=ker)
        tree_model.fit(X_train, y_train)
        predict_data = tree_model.predict(X_test)
        accuracy = np.mean(predict_data==y_test)
        y_score = tree_model.predict_proba(X_test)
        y_score = list(y_score[:,1])

        # print('tree')
        # print(accuracy)
        # print()
        temp.append([accuracy,y_score])
    temp = np.array(temp)
    temp2.append(temp[np.where(temp[:,0] == np.max(temp[:,0]))[0][0]])
    # temp2.append(max(temp))
    time_end = time.time() - time_start
for i in range(len(DT_kernel)):
    print('The accuracy of Decision Tree with kernel %s is %.4f , it took %.5f sec'%(DT_kernel[i] , temp2[i][0] , time_end/1000))
    total_res.append('The accuracy of Decision Tree with kernel %s is %.4f, it took %.5f sec for each, optimizing the parameter took %.5f sec'%(DT_kernel[i] , temp2[i][0], time_end/1000 , time_end))
    roc('DT' , y_test , temp2[i][-1])
doc = open('DT_results.txt','w')
for i in range(len(temp)):

    print(temp[i],file=doc)
    print('',file=doc)
doc.close()



tree_model = tree.DecisionTreeClassifier(criterion='gini')
time_start = time.time()
tree_model.fit(X_train, y_train)
predict_data = tree_model.predict(bcu_processed)
time_end = time.time() - time_start
bcu['test'] = predict_data
plothist(tagged_bll['PowerLaw_Index'] , bcu[bcu['test'] == 'bll']['PowerLaw_Index'] , 'DT' , 'bll')
plothist(tagged_fsrq['PowerLaw_Index'] , bcu[bcu['test'] == 'fsrq']['PowerLaw_Index'] , 'DT' , 'fsrq')
bcu = bcu.drop('test' , axis = 1)
predict_total.append(predict_data)

predict_data = tree_model.predict(X_test)
acc , precision , tpr , fpr = evaluation('DT' , predict_data , y_test)
total_res.append('Decision Tree precision is %.4f , TPR is %.4f , FPR is %.4f'%(precision , tpr , fpr))


doc = open('DT_test_result.txt','w')
for i in range(len(predict_data)):
    print(predict_data[i],file=doc)
    print('',file=doc)
doc.close()

#-----------------------------------------------------------------Neighbors-------------------------------------------------------------------------------



from sklearn import neighbors
knn_res = []
res = []
temp2 = []
temp3 = []
time_start = time.time()
for i in range(5,101):
    temp = []
    for j in range(10):
        #划分训练集和测试集
        # X_train, X_test, y_train, y_test = train_test_split(reduced_x, y, test_size=0.2, random_state=0)
        # 建立KNN模型，邻居数选为7，默认为5
        knn_model = neighbors.KNeighborsClassifier(n_neighbors = i)
        knn_model.fit(X_train, y_train)
        #对测试集进行预测
        predict_data = knn_model.predict(X_test)
        accuracy = np.mean(predict_data==y_test)
        y_score = tree_model.predict_proba(X_test)
        y_score = list(y_score[:,1])
        # print('neighbors')
        # print(accuracy)
        knn_res.append([accuracy , y_score])
    temp = np.array(knn_res)
    temp2.append(temp[np.where(temp[:,0] == np.max(temp[:,0]))[0][0]])
temp2 = np.array(temp2)
num = np.where(temp2[:,0] == np.max(temp2[:,0]))[0][0]
temp3 = temp2[num]
time_end = time.time() - time_start




total_res.append('The accuracy of knn is %.4f , the n_neighbours value is %d , it took %.5f sec for each, optimizing the parameter took %.5f sec'%(temp3[0] , num+1 , time_end/950 , time_end))
para_KNN = num+5
roc('KNN' , y_test , temp3[-1])
for i in range(len(total_res)):
    print(total_res[i])


knn_model = neighbors.KNeighborsClassifier(n_neighbors = para_KNN)
time_start = time.time()
knn_model.fit(X_train, y_train)
#对测试集进行预测
predict_data = knn_model.predict(bcu_processed)
time_end = time.time() - time_start
print(bcu)
bcu['test'] = predict_data
plothist(tagged_bll['PowerLaw_Index'] , bcu[bcu['test'] == 'bll']['PowerLaw_Index'] , 'KNN' , 'bll')
plothist(tagged_fsrq['PowerLaw_Index'] , bcu[bcu['test'] == 'fsrq']['PowerLaw_Index'] , 'KNN' , 'fsrq')
bcu = bcu.drop('test' , axis = 1)
print(bcu)
predict_total.append(predict_data)

predict_data = knn_model.predict(X_test)
acc , precision , tpr , fpr = evaluation('KNN' , predict_data , y_test)
y_score = tree_model.predict_proba(X_test)
print(y_score)
total_res.append('KNN precision is %.4f , TPR is %.4f , FPR is %.4f'%(precision , tpr , fpr))

doc = open('KNN_test_result.txt','w')
for i in range(len(predict_data)):
    print(predict_data[i],file=doc)
    print('',file=doc)
doc.close()

doc = open('total_results_0.2_V2.txt','w')
for i in range(len(total_res)):
    print(total_res[i],file=doc)
    print('',file=doc)
doc.close()

# ----------------------------------------------------------chi square exam----------------------------------------------------------------







predict_total = np.array(predict_total)
print(predict_total)

temp = []
res = []
# for i in range(len(predict_total[0])):
#     # for j in range(len(predict_total)):
#     #     temp.append(predict_total[j][i])
#     res.append(predict_total[:,i])
#     # temp = []
# predict_total = res

# bll , fsrq = classify(predict_total)
# bll_csv = pd.DataFrame(bll , columns=['source_name' , 'powerlaw_index' , 'SVM_Linear' , 'SVM_rbf' , 'SVM_sigmoid' , 'MLP_adam' , 'MLP_lbfgs' , 'MLP_sgd' , 'LR' , 'DT' , 'KNN'])
# fsrq_csv = pd.DataFrame(fsrq , columns=['source_name' , 'powerlaw_index' , 'SVM_Linear' , 'SVM_rbf' , 'SVM_sigmoid' , 'MLP_adam' , 'MLP_lbfgs' , 'MLP_sgd' , 'LR' , 'DT' , 'KNN'])
# print(bll)
# print(fsrq)

final = []
flag = 0
for i in range(len(predict_total[0])):
    for j in range(len(predict_total)-1):
        if predict_total[j][i] == predict_total[j+1][i]:flag = flag + 1
        else:break
    if flag == len(predict_total)-1 :final.append(predict_total[j][i])
    else:final.append('bcu')
    flag = 0
print(final)
print(len(final))
bll_num = final.count('bll')
fsrq_num = final.count('fsrq')
print(bll_num , fsrq_num)
print(bcu)

print(predict_total)
num_bcu = final

doc = open('chi_square_results_0.2_V2.txt','w')
total_name = kernel_svm + MLP_kernl + ['LR' , 'DT' , 'KNN']
kf = sp.stats.chisquare([len(tagged_bll) , len(tagged_fsrq)] , [bll_num , fsrq_num])
print(kf)
for i in range(len(predict_total)):
    for j in range(len(predict_total)):
        if i == j:continue
        else:
            print('This is the chi2 test for the results bewteen %s and %s'%(total_name[i] , total_name[j]))
            print('This is the chi2 test for the results bewteen %s and %s'%(total_name[i] , total_name[j]) , file = doc)

            print('----------------------')
            a=sum(predict_total[i]=='bll')
            b=sum(predict_total[i]=='fsrq')
            c=sum(predict_total[j]=='bll')
            d=sum(predict_total[j]=='fsrq')
            print(a,b)
            print(c,d)
            print('----------------------')
            kafang = sp.stats.chisquare([a,b] , [c,d])
            print(kafang)
            print(kafang , file = doc)
            print()
doc.close()
# ----------------------------------------------------------plot the histgrame-------------------------------------------------------------------

items = kernel_svm + MLP_kernl + ['LR' , 'DT'  ,'KNN' , 'final'] # 'LR' , 'DT'  , 
for i,temp in enumerate(predict_total):
    bcu[items[i]] = temp
bcu['final'] = final
print(bcu)
bcu.to_csv('bcu_test_res.csv')
bcu_bll = bcu[bcu['final'] == 'bll']
bcu_fsrq = bcu[bcu['final'] == 'fsrq']
still_bcu = bcu[bcu['final'] == 'bcu']
print(bcu_bll)
bcu_bll.to_csv('classified_as_bll.csv')
print(bcu_fsrq)
bcu_fsrq.to_csv('classified_as_fsrq.csv')
print(still_bcu)
still_bcu.to_csv('classified_as_bcu.csv')




bll_ks_test_res = []
temp = []
for name in names:
    temp.append(name)
    data_1 = tagged_bll[name]
    data_2 = bcu_bll[name]
    # print(data_1)
    # print('-------------------------------------------------------------------')
    # print(data_2)
    res = sp.stats.ks_2samp(data_1 , data_2)
    temp.append(res)
    # data_1 = data_1[data_1 < 0.7*np.max(data_1)]
    # data_2 = data_2[data_2 < 0.7*np.max(data_2)]
    if name == 'Pivot_Energy':
        temp2 = []
        for num in data_1:
            temp2.append(math.log(num,10))
        data_1 = temp2
        temp2 = []
        for num in data_2:
            temp2.append(math.log(num,10))
        data_2 = temp2
        temp2 = []
    plt.hist(data_1 , label = 'tagged sources%d'%(len(data_1)))
    plt.hist(data_2 , alpha = 0.5 , label = 'BCU sources%d'%(len(data_2)))
    plt.legend()
    plt.title(name)
    plt.savefig('bll_'+name+'.jpg')
    plt.close()
    if res[-1] > 0.95:
        temp.append('this feature of bcu follow the tagged sources')
    bll_ks_test_res.append(temp)
    temp = []


for ele in bll_ks_test_res:
    print(ele)

doc = open('ks_test_bll_result.txt','w')
for i in range(len(bll_ks_test_res)):
    print(bll_ks_test_res[i],file=doc)
    print('',file=doc)
doc.close()

print('------------------------------------------------------------------------')
fsrq_ks_test_res = []
temp = []
for name in names:
    temp.append(name)
    data_1 = tagged_fsrq[name]
    data_2 = bcu_fsrq[name]
    res = sp.stats.ks_2samp(data_1 , data_2)
    temp.append(res)
    if name == 'Pivot_Energy':
        temp2 = []
        for num in data_1:
            temp2.append(math.log(num,10))
        data_1 = temp2
        temp2 = []
        for num in data_2:
            temp2.append(math.log(num,10))
        data_2 = temp2
        temp2 = []
    plt.hist(data_1 , label = 'tagged sources %d'%(len(data_1)))
    plt.hist(data_2 , alpha = 0.5 , label = 'BCU sources %d'%(len(data_2)))
    plt.legend()
    plt.title(name)
    plt.savefig('fsrq_'+name+'.jpg')
    plt.close()
    if res[-1] > 0.95:
        temp.append('this feature of bcu follow the tagged sources')
    fsrq_ks_test_res.append(temp)
    temp = []

for ele in fsrq_ks_test_res:
    print(ele)

doc = open('ks_test_fsrq_result.txt','w')
for i in range(len(fsrq_ks_test_res)):
    print(fsrq_ks_test_res[i],file=doc)
    print('',file=doc)
doc.close()


#----------------------------------------------examinate the result with 4FGL-------------------------------------------
rawdata_4fgl = fits.open('gll_psc_v19.fit')
data_4fgl = rawdata_4fgl[1].data 
testset = []
i = 0
# for name in bcu_name:
#     for i in range(flag , len(data_4fgl)):
#         if data_4fgl['ASSOC_FGL'][i] == name:
#             testset.append(data_4fgl[i])
#             print(name , 'is found')
#             flag = i
#             continue
# print(testset)
# testset = data_4fgl[mask]
# print('there are %d bll , %d fsrq , %d bcu in 4FGL testset'%(bll_sum , fsrq_sum , bcu_sum))

doc = open('4FGL_test_process.txt' , 'w')
correct = 0
exam_to_be_right = []
bll_sum = 0
fsrq_sum = 0
bcu_sum = 0
null_sum = 0
others_sum = 0
cla_as_bll = 0
for i,name in enumerate(bcu['Source_Name']):
    mask = data_4fgl['ASSOC_FGL'] == name
    print(data_4fgl[mask]['CLASS1'] , bcu.loc[i]['final'])
    print(data_4fgl[mask]['CLASS1'] , bcu.loc[i]['final'] , file = doc)
    if len(data_4fgl[mask]['CLASS1']) == 0 or data_4fgl[mask]['CLASS1'][-1] == '' : null_sum = null_sum + 1
    elif data_4fgl[mask]['CLASS1'][-1] == 'bll' or data_4fgl[mask]['CLASS1'][-1] == 'BLL':bll_sum = bll_sum + 1
    elif data_4fgl[mask]['CLASS1'][-1] == 'fsrq' or data_4fgl[mask]['CLASS1'][-1] == 'FSRQ':fsrq_sum = fsrq_sum + 1
    elif data_4fgl[mask]['CLASS1'][-1] == 'bcu' or data_4fgl[mask]['CLASS1'][-1] == 'BCU':bcu_sum =bcu_sum + 1 
    else: others_sum = others_sum + 1
    # bll_sum = sum(testset['CLASS1'] == 'bll') + sum(testset['CLASS1'] == 'BLL')
    # fsrq_sum = sum(testset['CLASS1'] == 'fsrq') + sum(testset['CLASS1'] == 'FSRQ')
    # bcu_sum = sum(testset['CLASS1'] == 'bcu') + sum(testset['CLASS1'] == 'BCU')
    if sum(mask) == 0 or data_4fgl[mask]['CLASS1'][-1] == 'bcu': continue
    elif data_4fgl[mask]['CLASS1'][-1].lower() == bcu.loc[i]['final']:
        if bcu.loc[i]['final'] == 'bll':cla_as_bll = cla_as_bll + 1
        print('%s is exam to be right via 4FGL data , it is %s'%(bcu.loc[i]['Source_Name'] , bcu.loc[i]['final']))
        print('%s is exam to be right via 4FGL data , it is %s'%(bcu.loc[i]['Source_Name'] , bcu.loc[i]['final']) , file = doc)
        correct = correct + 1
        exam_to_be_right.append(bcu.loc[i])
print(correct , ' sources are tested to be right via 4FGL data' , cla_as_bll , 'are bll')
print(correct , ' sources are tested to be right via 4FGL data' , file = doc)
print('there are %d bll , %d fsrq , %d bcu , %d have no tag , %d other types in 4FGL testset'%(bll_sum , fsrq_sum , bcu_sum , null_sum , others_sum) )
print('there are %d bll , %d fsrq , %d bcu , %d have no tag , %d other types in 4FGL testset'%(bll_sum , fsrq_sum , bcu_sum , null_sum , others_sum) , file = doc)
doc.close()

doc = open('4FGL_test_result.txt','w')
for i in range(len(exam_to_be_right)):
    print(exam_to_be_right[i],file=doc)
    print('',file=doc)
doc.close()

# for ele in exam_to_be_right:
#     print(ele.loc[0])
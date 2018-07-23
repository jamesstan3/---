#!/usr/bin/env python
#coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
from numpy import*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
import lightgbm as lgb
def load_data(inputFile):
    dfData = pd.read_csv(inputFile)
    # data_X = dfData.drop(labels = ['USRID'],axis = 1)
    dfAns = pd.read_csv('./train/cleaned_train_flg.csv')
    allUserAns = pd.merge(dfData, dfAns, how = 'left', on = 'USRID')
    pos=allUserAns[allUserAns.flag==1]
    neg=allUserAns[allUserAns.flag==0]
    frac_neg=neg.sample(frac=0.6,random_state=2018)
    allUserAns=pd.concat([pos,frac_neg])
    allUserAns.fillna(-999, inplace = True)
    data_X=allUserAns.drop(labels=['USRID','flag'],axis=1)
    data_y = allUserAns['flag']
    print allUserAns.shape
    return data_X,data_y
def discretization(df,last_day_internal):
    day_df=df[last_day_internal]
    no_day_df=df.drop([last_day_internal],axis=1)
    day_df[day_df<7]=1
    day_df[(day_df>=7)&(day_df<14)]=2
    day_df[(day_df>=14)&(day_df<21)]=3
    day_df[day_df>=21]=4
    no_day_df[last_day_internal]=day_df
    no_day_df=pd.get_dummies(no_day_df,columns=[last_day_internal],prefix=['d_'+last_day_internal])
    print no_day_df.head()
    return no_day_df
def lgbTrainModel(inputFile):
    from lightgbm.sklearn import LGBMClassifier
    dfData = pd.read_csv(inputFile)
    data = dfData.drop(labels = ['USRID'],axis = 1)
    data_X = data.fillna(-999)
    dfAns = pd.read_csv('./train/cleaned_train_flg.csv')
    allUserAns = pd.merge(dfData, dfAns, how = 'left', on = 'USRID')
    allUserAns.fillna(-999, inplace = True)
    data_y = allUserAns['flag']
    # data_X,data_y=load_data(inputFile)
    # 自定义划分数据集进行线下验证,发现效果并不好
    # data_X,data_y=load_data(inputFile)
    xTrain, xTest, yTrain, yTest = train_test_split(data_X, data_y, test_size = 0.2, random_state =100)
    lgbTrain = lgb.Dataset(xTrain, yTrain)
    lgbEval = lgb.Dataset(xTest, yTest)
    lgbAll = lgb.Dataset(data_X, data_y)
    
    param_grid = {
    # 'max_depth':[2,3,4,5,6,8]
    'learning_rate': [0.05,0.1,0.15],
    'num_leaves':[5,10,20,30]
    }
    # params={'objective':'binary','is_unbalance':'True','boosting_type': 'gbdt','metric':'auc','seed':0,'max_depth':4,'n_estimators':1000,
    # 'feature_fraction':0.8,'bagging_fraction':0.8,'num_boost_round':1000,
    # 'num_leaves': 20,'learning_rate': 0.05
    # }
    params={'objective':'binary','is_unbalance':'True','boosting_type': 'gbdt','metric':'auc','seed':0,'num_leaves': 20,'learning_rate': 0.05}
    ##grid调参
    # gsearch1 = GridSearchCV(estimator =LGBMClassifier(objective='binary',is_unbalance='True',boosting_type='gbdt'),param_grid = param_grid,scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    # gsearch1.fit(data_X,data_y)
    # print gsearch1.grid_scores_,
    # print gsearch1.best_params_
    # print gsearch1.best_score_
    ##调参结束后进行模型训练
    numRound = 10000  # 不会过拟合的情况下，可以设大一点
    modelTrain = lgb.train(params, lgbTrain, numRound, valid_sets=lgbEval, early_stopping_rounds=200)
    # 用分出的部分训练集测出的最佳迭代次数在，全体训练集中重新训练
    model = lgb.train(params, lgbAll, modelTrain.best_iteration)
    model.save_model('./model/lgb.model') # 用于存储训练出的模型
    # print(model.feature_importance()) # 看lgb模型特征得分
    dfFeature = pd.DataFrame()
    dfFeature['featureName'] = model.feature_name()
    dfFeature['score'] = model.feature_importance()
    dfFeature.to_csv('./model/featureImportance.csv',index=None)
def lgbKFoldTrainModel(inputFile):
    """
    Desc：用lightgbm来跑出回归模型，并预测看看
    """
    from lightgbm.sklearn import LGBMClassifier
    dfData = pd.read_csv(inputFile)
    data = dfData.drop(labels = ['USRID'],axis = 1)
    data_X = data.fillna(-999)
    print(data_X.shape)
    dfAns = pd.read_csv('./train/cleaned_train_flg.csv')
    allUserAns = pd.merge(dfData, dfAns, how = 'left', on = 'USRID')
    # allUserAns['s1_count']=allUserAns['s1_count'].apply(lambda x:np.log(x))
    allUserAns.fillna(-999, inplace = True)
    data_y = allUserAns['flag']
    # data_X,data_y=load_data(inputFile)
    params = {
    'objective':'binary',
    'boosting_type': 'gbdt',
    'metric':'auc',
    'seed':0,
    'num_leaves': 20,
    'learning_rate': 0.05,
    }
    lgbAll = lgb.Dataset(data_X, data_y)
    # KFold交叉验证
    kf = KFold(n_splits = 5, random_state = 20)
    kf.get_n_splits(data_X)
    print(kf)
    bestIterRecord = []  # 记录每次的最佳迭代次数
    aucRecord = []  # 记录每次的最佳迭代点的auc
    numRound = 10000  # 不会过拟合的情况下，可以设大一点
    for trainIndex, testIndex in kf.split(data_X):
        print("Train Index:", trainIndex, ",Test Index:", testIndex)
        xTrain, xTest = data_X.iloc[trainIndex], data_X.iloc[testIndex]
        yTrain, yTest = data_y.iloc[trainIndex], data_y.iloc[testIndex]

        lgbTrain = lgb.Dataset(xTrain, yTrain)
        lgbEval = lgb.Dataset(xTest, yTest)
        evalAuc = {} # 存储实时的auc结果
        modelTrain = lgb.train(
            params = params, 
            train_set = lgbTrain, 
            num_boost_round = numRound, 
            valid_sets = lgbEval,
            valid_names = 'get_auc',
            evals_result = evalAuc,
            early_stopping_rounds = 200)

        bestIterRecord.append(modelTrain.best_iteration)
        aucRecord.append(evalAuc.get('get_auc').get('auc')[modelTrain.best_iteration - 1])

    bestIter = int(np.mean(bestIterRecord))  # 利用KFold求出的平均最佳迭代次数

    # 用分出的部分训练集测出的最佳迭代次数在，全体训练集中重新训练
    model = lgb.train(params, lgbAll, bestIter)
    model.save_model('lgb.model') # 用于存储训练出的模型
    dfFeature = pd.DataFrame()
    dfFeature['featureName'] = model.feature_name()
    dfFeature['score'] = model.feature_importance()
    dfFeature.to_csv('featureImportance.csv',index=None)

    # predTest = model.predict(data_X)
    print('mean of auc : ', np.mean(aucRecord))
    print('best iteration : ', bestIter)
    # print('self rmse : ', np.sqrt(mean_squared_error(loanAns, predTest)))

def lgbPredict(predictInput, modelFile):
    """
    Desc：借助已经跑出的模型，来预测A榜的正确性
    """
    model = lgb.Booster(model_file = './model/'+modelFile) #init model
    
    dfData=pd.read_csv(predictInput)
    # 去掉uid列,uid不是特征
    data = dfData.drop(labels = 'USRID',axis = 1)
    preds = model.predict(data)
    preds=pd.DataFrame(preds)
    preds.columns=['RST']
    df=pd.concat([dfData[['USRID']],preds],axis=1)
    df.to_csv('./model/test_result.csv',sep='\t',index=None)

def main():
    ##方式一尝试，线上0.78

    ##方式二尝试，线上0.824
    ##方式三尝试，线下0.85，线上0.861

    #方式四：单独把有点击行为记录的数据来训练，观察有没有提高,效果很差
    #将生产的排序特征和原始数值特征进行训练，筛选top40的特征，重新作为agg表生成的特征
    # trainInput='./feature_eng/train_corr.csv'
    # predictInput='./feature_eng/test_corr.csv'

    trainInput='trainInput.csv'
    predictInput='testInput.csv'
    # trainInput='./dataOldVersion/trainInput.csv'
    # predictInput='./dataOldVersion/testInput.csv'
    lgbKFoldTrainModel(trainInput)
    lgbTrainModel(trainInput)
    lgbPredict(predictInput,'lgb.model')


if __name__ == '__main__':
    main()

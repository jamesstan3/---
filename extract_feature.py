#!/usr/bin/python
# coding:utf8
import numpy as np
import pandas as pd
import math
from datetime import datetime
from datetime import date
def map_hours2bucket(action,hours):
    if hours>=8 and hours<=11:
        return '%s_hours_01' % action
    if hours >=12 and hours<=15:
        return '%s_hours_02' % action
    if hours >= 16 and hours<=19:
        return '%s_hours_03' % action
    if hours>=20 and hours<=23:
        return '%s_hours_04' % action
    if hours>=4 and hours <= 7:
        return '%s_hours_05' % action
    else:
        return '%s_hours_06' % action

def dealLog(trainPath):
	"""
	Desc:将操作行为日志表取出，统计模块第一等级项的count（一个月） 并且把s1-s2-s3进行one-hot编码
	output:分别对训练集和测试集进行处理，输入到模型进行分类训练
	"""
	logTrain=pd.read_csv(trainPath)
	click=logTrain[['USRID']]
	click['click']=1
	click=click.groupby(['USRID']).agg('count').reset_index().rename(columns={'click':'click_count'})
	logTrain=pd.merge(logTrain,click,on='USRID',how='left')

	import time
	day=logTrain[['USRID','day']]
	day.drop_duplicates(subset=['USRID','day'], keep='last', inplace=True)
	day = day.sort_values(['USRID','day'])
	# print day.head
	# print day.shape
	day['next_time'] = day.groupby(['USRID'])['day'].diff(-1).apply(np.abs)
	day= day.groupby(['USRID'],as_index=False)['next_time'].agg({
	    'next_time_day_mean':np.mean,
	    'next_time_day_std':np.std,
	    'next_time_day_min':np.min,
	    'next_time_day_max':np.max
	})
	log=logTrain[['USRID','OCC_TIM']]
	log['OCC_TIM_s'] = log['OCC_TIM'].apply(lambda x:time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
	log = log.sort_values(['USRID','OCC_TIM_s'])
	log['next_time'] = log.groupby(['USRID'])['OCC_TIM_s'].diff(-1).apply(np.abs)
	log = log.groupby(['USRID'],as_index=False)['next_time'].agg({
	    'next_time_s_mean':np.mean,
	    'next_time_s_std':np.std,
	    'next_time_s_min':np.min,
	    'next_time_s_max':np.max
	})
	log=pd.merge(day,log,on='USRID',how='left')
	print log.shape
	print log.head()

	##第一次只把点击模块s1-s2-s3进行亚编码的尝试,
	##1.点击时间特征分布,将一天划分六个时段，统计每个时段用户点击次数占该用户总点击次数的比例
	# click_hour_df=logTrain[['USRID','OCC_TIM']]
	# click_hour_df['click_time_hours'] = click_hour_df['OCC_TIM'].map(lambda x: int(x.split(' ')[1].split(':')[0]))
	# click_hour_df['click_time_hours'] = click_hour_df['click_time_hours'].map(lambda x : map_hours2bucket('click',x))
	# click_hour_df = click_hour_df.groupby(['USRID','click_time_hours'],as_index=False).count()
	# click_hour_df = click_hour_df.pivot(index='USRID', columns='click_time_hours', values='OCC_TIM').reset_index()
	# click_hour_df = click_hour_df.fillna(0)
	# click_hour_df['click_sum_hour'] = click_hour_df[['click_hours_01','click_hours_02','click_hours_03','click_hours_04','click_hours_05','click_hours_06']].apply(lambda x: x.sum(),axis=1)
	# click_hour_df['click_hours_01']=click_hour_df['click_hours_01']/click_hour_df['click_sum_hour']
	# click_hour_df['click_hours_02']=click_hour_df['click_hours_02']/click_hour_df['click_sum_hour']
	# click_hour_df['click_hours_03']=click_hour_df['click_hours_03']/click_hour_df['click_sum_hour']
	# click_hour_df['click_hours_04']=click_hour_df['click_hours_04']/click_hour_df['click_sum_hour']
	# click_hour_df['click_hours_05']=click_hour_df['click_hours_05']/click_hour_df['click_sum_hour']
	# click_hour_df['click_hours_06']=click_hour_df['click_hours_06']/click_hour_df['click_sum_hour']
	# del click_hour_df['click_sum_hour']
	# del logTrain['OCC_TIM']

	# ##2.一个月内的用户每天点击次数统计量（mean，max，min,count,sum）;每小时，不同细粒度统计量;周末点击次数占比；周三点击次数占比
	holiday=[7,14,21,28]
	statistic_day=logTrain[['USRID','day']]
	statistic_day['click']=1
	statistic_day= statistic_day.groupby(['USRID','day']).sum().reset_index()
	stat_feat = ['min','mean','max','median','std']
	statistic_s1 = statistic_day.groupby('USRID')['click'].agg(stat_feat).reset_index()
	statistic_s1.columns = ['USRID'] + ['click_statistic_day_' + col for col in stat_feat]
	statistic_s1.fillna(0.0,inplace=True)


	statistic_weekend=logTrain[['USRID','is_weekend']]
	click_count=logTrain[['USRID','click_count']].drop_duplicates()
	click_weekend_count=statistic_weekend.groupby(['USRID'])['is_weekend'].agg('sum').reset_index()
	statistic_weekend=pd.merge(click_count,click_weekend_count,on='USRID',how='left')
	statistic_weekend['weekend_click_ratio']=1.0*statistic_weekend['is_weekend']/statistic_weekend['click_count']
	statistic_weekend=statistic_weekend[['USRID','weekend_click_ratio']]

	statistic_wednesday=logTrain[['USRID','is_wednesday']]
	# click_count=logTrain[['USRID','click_count']].drop_duplicates()
	click_wednesday_count=statistic_wednesday.groupby(['USRID'])['is_wednesday'].agg('sum').reset_index()
	statistic_wednesday=pd.merge(click_count,click_wednesday_count,on='USRID',how='left')
	statistic_wednesday['wednesday_click_ratio']=1.0*statistic_wednesday['is_wednesday']/statistic_wednesday['click_count']
	statistic_wednesday=statistic_wednesday[['USRID','wednesday_click_ratio']]

	statistic_last2day=logTrain[['USRID','day']]
	statistic_last2day['is_last2day']=statistic_last2day['day'].apply(lambda x:1 if x>=30 else 0)
	del statistic_last2day['day']
	statistic_last2day_count=statistic_last2day.groupby(['USRID'])['is_last2day'].agg('sum').reset_index()
	statistic_last2day=pd.merge(click_count,statistic_last2day_count,on='USRID',how='left')
	statistic_last2day['last2day_click_ratio']=1.0*statistic_last2day['is_last2day']/statistic_last2day['click_count']
	statistic_last2day=statistic_last2day[['USRID','last2day_click_ratio']]


	statistic_s1=pd.merge(statistic_s1,statistic_weekend,on=['USRID'],how='left')
	statistic_s1=pd.merge(statistic_s1,statistic_wednesday,on=['USRID'],how='left')
	statistic_s1=pd.merge(statistic_s1,statistic_last2day,on=['USRID'],how='left')
	print statistic_s1.head()
	print statistic_s1.shape

	# ##3.用户对s1-s2-s3的点击次数占该用户总点击次数的比例（效果不如次数得分高，暂时考虑不转化为比例,觉得很有可能这是个陷阱特征）

	s1=logTrain[['USRID','s1']]
	s1_encoded=s1.groupby(['USRID'])['s1'].agg('count').reset_index()
	s1_encoded_sum=pd.get_dummies(s1,columns=['s1'],prefix='s1')
	s1_encoded_sum=s1_encoded_sum.groupby(['USRID']).agg('sum').reset_index()
	#点击次数最多的s1类别,top-1或者三个
	# for idx in list(s1_encoded_sum.index):
	# 	ser_index=s1_encoded_sum.ix[idx,1:].sort_values(ascending=False)[0:2].index
	# 	lis=np.array(ser_index)
	# 	for i in xrange(0,len(lis)):
	# 		col='s1_top'+str(i)
	# 		value=lis[i].split('_')[-1]
	# 		s1_encoded_sum.ix[idx,col]=value
	##########################################
	s1_encoded=pd.merge(s1_encoded,s1_encoded_sum,on='USRID',how='left')
	s1_encoded.rename(columns={'s1':'s1_count'},inplace=True)

	##用户点击EVT_LBL第一级别的种类数,用户点击EVT_LBL第二级别的种类数
	stat_feat=['mean','max']
	s1_nums=logTrain[['USRID','s1']].drop_duplicates()
	s1_nums=s1_nums.groupby(['USRID'])['s1'].agg('count').reset_index()
	# statistic_s1_nums=s1_nums.groupby(['USRID'])['s1'].agg(stat_feat).reset_index()
	# statistic_s1_nums.columns = ['USRID'] + ['statistic_s1_nums_' + col for col in stat_feat]
	# s1_nums=pd.merge(s1_nums,statistic_s1_nums,on=['USRID'],how='left')

	s2_nums=logTrain[['USRID','s2']].drop_duplicates()
	s2_nums=s2_nums.groupby(['USRID'])['s2'].agg('count').reset_index()
	# statistic_s2_nums=s2_nums.groupby(['USRID'])['s2'].agg(stat_feat).reset_index()
	# statistic_s2_nums.columns = ['USRID'] + ['statistic_s2_nums_' + col for col in stat_feat]
	# s2_nums=pd.merge(s2_nums,statistic_s2_nums,on=['USRID'],how='left')

	s1_encoded=pd.merge(s1_encoded,s1_nums,on='USRID',how='left')
	s1_encoded=pd.merge(s1_encoded,s2_nums,on='USRID',how='left')
	

	##4.最后一次点击距离预测多长时间；用户对s1等级各个类别的点击次数的月统计量；登录天数day_count,来划分三种类型的用户;登录天数的统计分布
	log_days=logTrain[['USRID','day']]
	log_days.drop_duplicates(subset=['USRID','day'],keep='last',inplace=True)
	# ##计算连续登陆的最大天数,转化为比例max_internal/day_count
	# internal_dic={'USRID':[],'max_internal':[]}
	# for name,group in log_days.groupby('USRID'):
	# 	group['c'] = ((group['day'].shift(1).fillna(0) + 1).astype(int) != group['day']).cumsum()
	# 	value=max(group['c'].value_counts().values)
	# 	internal_dic['USRID'].append(name)
	# 	internal_dic['max_internal'].append(value)
	# internal_df=pd.DataFrame(internal_dic)
	# day_count=log_days.groupby(['USRID'])['day'].agg('count').reset_index()
	# day_count=pd.merge(day_count,internal_df,on='USRID',how='left')
	# day_count['max_internal']=1.0*day_count['max_internal']/day_count['day']
	# day_count['max_internal']=day_count['max_internal'].fillna(day_count['max_internal'].mean())
	# day_count=day_count[['USRID','max_internal']]

	stat_feat = ['mean','std']
	statistic_day= log_days.groupby('USRID')['day'].agg(stat_feat).reset_index()
	statistic_day.columns = ['USRID'] + ['statistic_log_days_' + col for col in stat_feat]
	last_day=logTrain[['USRID','day']]
	last_day=last_day.groupby(['USRID']).agg('max').reset_index()
	last_day['last_day_internal']=32-last_day['day']
	del last_day['day']
	last_day=pd.merge(statistic_day,last_day,on='USRID',how='left')
	# last_day=pd.merge(last_day,day_count,on='USRID',how='left')
	####5.点击时间间隔，最后一次点击是否超过平均点击时间间隔
	time_df=logTrain[['USRID','day']]
	time_df['last_click_time']=32-time_df['day']
	time_df = time_df.sort_values(by=['USRID','last_click_time'])

	last_click_time = time_df.copy()
	last_click_time.drop_duplicates(subset='USRID', keep='last', inplace=True)
	last_click_time = last_click_time[['USRID','last_click_time']]
	last_click_time['per_click_time_interval'] = 32
	time_df = time_df.groupby(['USRID','last_click_time'],as_index=False).sum()
	del time_df['day']

	for idx in list(last_click_time.index):
		uid=last_click_time.ix[idx,'USRID']
		ser=time_df[time_df.USRID==uid]['last_click_time']
		interval =np.array(ser)
		if len(interval)==1:
			per_click_time_interval=32-interval[0]
		else:
			per_click_time_interval=(interval.max() - interval.min())/(len(interval)-1)
		last_click_time.ix[idx,'per_click_time_interval']=per_click_time_interval
	# last_click_time['is_exceed_click_interval']=last_click_time['last_click_time']-last_click_time['per_click_time_interval']
	del last_click_time['last_click_time']##与前面last_day_internal重复，故删除


	df=pd.merge(s1_encoded,last_day,on='USRID',how='left')
	# df=pd.merge(df,click_hour_df,on='USRID',how='left')
	df=pd.merge(df,statistic_s1,on='USRID',how='left')
	df=pd.merge(df,last_click_time,on='USRID',how='left')
	df=pd.merge(df,log,on='USRID',how='left')

	print df.shape
	print df.dtypes
	fileType=trainPath.split('/')[1]
	df.to_csv('dealed_'+str(fileType)+'_log.csv',index=None)


def dealAgg(trainPath):
	"""
	Desc:对于枚举特征对V2，V3，V4，V5，V6进行类别编码。对V1，V7-V30这些特征进行排序特征生成，比较！ 

	"""
	aggTrain=pd.read_csv(trainPath)
	##第一步：简单的类别编码;尝试进行特征组合,在生成特征的同时计算与标签列的皮尔逊相关系数，保留top—K个
	df=pd.get_dummies(aggTrain,columns=['V2','V4','V5'],prefix=['V2','V4','V5'])
	print df.shape
	fileType=trainPath.split('/')[1]
	df.to_csv('./dealed_'+str(fileType)+'_agg.csv',index=None)

	#排序特征+类别编码特征
	# numeric_feature=['V1','V3']
	# for i in xrange(6,31):
	# 	numeric_feature.append('V'+str(i))
	# cat_feature=['V2','V4','V5']
	# cat_df=aggTrain[['USRID']+cat_feature]
	# cat_df=pd.get_dummies(cat_df,columns=['V2','V4','V5'],prefix=['V2','V4','V5'])

	# test = aggTrain[['USRID']+numeric_feature]
	# test_rank = pd.DataFrame(test.USRID,columns=['USRID'])
	# for feature in numeric_feature:
	#     test_rank['r'+feature] = test[feature].rank(method='max')
	# test_rank=pd.merge(test_rank,cat_df,on='USRID',how='left')
	# fileType=trainPath.split('/')[1]
	# test_rank.to_csv('./feature_eng/'+str(fileType)+'_rank.csv',index=None)
def extract_agg_features():
	##第二步：通过lgb训练agg表生成的组合特征，然后读取importance前列的特征，保留若干个,和agg原始v1,V3,V6-v30特征组合
	aggTrain=pd.read_csv('./feature_eng/train_corr.csv')
	aggTest=pd.read_csv('./feature_eng/test_corr.csv')
	print aggTrain.shape
	print aggTest.shape
	fea=pd.read_csv('./feature_eng/featureImportance.csv')
	fea=fea.sort_values(by=['score'],ascending=False)
	#1.保留原始v特征,再添加组合特征
	# cols=['USRID','V1']
	# for x in xrange(6,31):
	# 	cols.append('V'+str(x))
	# fu_cols=[]
	# add_cols=list(fea[0:100].featureName)
	# for c in add_cols:
	# 	if c in cols:
	# 		continue
	# 	fu_cols.append(c)
	# cols.extend(fu_cols)
	#2.只根据得分筛选前30特征
	cols=['USRID']
	cols.extend(list(fea[0:30].featureName))
	mergeTrain=aggTrain[cols]
	mergeTest=aggTest[cols]
	print aggTrain.head()
	print aggTest.head()
	print mergeTrain.shape
	print mergeTest.shape
	mergeTrain.to_csv('./feature_eng/dealed_train_fu_agg.csv',index=None)
	mergeTest.to_csv('./feature_eng/dealed_test_fu_agg.csv',index=None)
	pass
def makeMergeInput():
	"""Desc:第三步：融合log表基本特征，查看线上线下结果有没有很好的提升！
	"""
	aggTrain=pd.read_csv('./dealed_train_agg.csv')
	aggTest=pd.read_csv('./dealed_test_agg.csv')
	logTrain=pd.read_csv('./dealed_train_log.csv')
	logTest=pd.read_csv('./dealed_test_log.csv')
	mergeTrain = pd.merge(aggTrain, logTrain, how = 'left', on = 'USRID')
	mergeTest=pd.merge(aggTest,logTest,how='left',on='USRID')
	print("aggTrain shape:",aggTrain.shape)
	print("logTrain shape:",logTrain.shape)
	print("aggTest shape:",aggTest.shape)
	print("logTest shape:",logTest.shape)
	print mergeTrain.shape
	print mergeTest.shape

	mergeTrain.to_csv('./trainInput.csv',index=None)
	mergeTest.to_csv('./testInput.csv',index=None)
def makeMergeRank():
	"""Desc:将agg表提取的rank特征和原始特征融合起来，
	"""
	rankAggTrain=pd.read_csv('./rank_features/train_rank.csv')
	rankAggTest=pd.read_csv('./rank_features/test_rank.csv')
	aggTrain=pd.read_csv('./train/cleaned_train_agg.csv')
	aggTest=pd.read_csv('./test/cleaned_test_agg.csv')

	#去除重复的特征列
	aggTrain.drop(['V2','V3','V4','V5','V6'],axis=1,inplace=True)
	aggTest.drop(['V2','V3','V4','V5','V6'],axis=1,inplace=True)
	print('rankAggTrain:',rankAggTrain.shape)
	print('rankAggTest:',rankAggTest.shape)
	print('aggTrain:',aggTrain.shape)
	print('aggTest:',aggTest.shape)
	mergeTrain = pd.merge(rankAggTrain,aggTrain, how = 'left', on = 'USRID')
	mergeTest=pd.merge(rankAggTest,aggTest,how='left',on='USRID')
	print mergeTrain.shape
	print mergeTest.shape

	cols_df=pd.read_csv('featureImportance.csv')
	cols_df=cols_df.sort_values(by=['score'])
	cols=list(cols_df['featureName'][0:20].values)
	mergeTrain.drop(labels=cols,axis=1,inplace=True)
	mergeTest.drop(labels=cols,axis=1,inplace=True)
	print mergeTrain.shape
	print mergeTest.shape

	mergeTrain.to_csv('./rank_features/fuse_rank_train_input.csv',index=None)
	mergeTest.to_csv('./rank_features/fuse_rank_test_input.csv',index=None)

def makeMergeAll():
	"""Desc:承接makeMergeRank，再将原始agg特征添加进来;通过特征选择，top40的agg表特征
	"""
	rankAggTrain=pd.read_csv('./rank_features/fuse_rank_train_input.csv')
	logTrain=pd.read_csv('dealed_train_log.csv')
	rankAggTest=pd.read_csv('./rank_features/fuse_rank_test_input.csv')
	logTest=pd.read_csv('dealed_test_log.csv')

	print("rankAggTrain shape:",rankAggTrain.shape)
	print("logTrain shape:",logTrain.shape)

	print("rankAggTest shape:",rankAggTest.shape)
	print("logTest shape:",logTest.shape)
	mergeTrain = pd.merge(rankAggTrain, logTrain, how = 'left', on = 'USRID')
	mergeTest=pd.merge(rankAggTest,logTest,how='left',on='USRID')
	print mergeTrain.shape
	print mergeTest.shape

	mergeTrain.to_csv('./rank_features/allTrainInput.csv',index=None)
	mergeTest.to_csv('./rank_features/allTestInput.csv',index=None)

def main():
	logTrainPath='./train/cleaned_train_log.csv'
	aggTrainPath='./train/cleaned_train_agg.csv'
	logTestPath='./test/cleaned_test_log.csv'
	aggTestPath='./test/cleaned_test_agg.csv'
	dealLog(logTrainPath)
	# dealAgg(aggTrainPath)
	dealLog(logTestPath)
	# dealAgg(aggTestPath)
	# extract_agg_features()
	
	makeMergeInput()
	# makeMergeRank()
	# makeMergeAll()

if __name__ == '__main__':
	main()

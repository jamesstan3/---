#!/usr/bin/python
# coding:utf8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
"""Desc:主要对比赛的原始数据进行数据清洗，转化出基本的一些特征
"""
def explore_data(trainPath):
	df=pd.read_csv(trainPath)
	print df.describe(include='all')
	# df.info()
	#查看哪些列是类别列
	# cols=df.columns
	# for x in xrange(0,len(cols)-1):
	# 	print cols[x]
	# 	print df[cols[x]].value_counts()

def trainFlag(trainPath):
	##分离独立每一个列
	df=pd.read_csv(trainPath)
	Uid_Flag=df.ix[:,0]
	df['USRID']=Uid_Flag.apply(lambda line:int(line.split('\t')[0]))
	df['flag']=Uid_Flag.apply(lambda line:int(line.split('\t')[-1]))
	df.drop(df.columns[[0]],axis=1,inplace=True)
	print df.head()
	inputfile=trainPath.split('/')[-1]
	df.to_csv('./train/cleaned_'+inputfile,index=None)
def trainLog(trainPath):
	from datetime import datetime
	##分离独立每一个列，将点击模块三个级别独立出来，命名为s1，s2，s3,因为只有一个月的时间所以只保留天
	df=pd.read_csv(trainPath)

	orig_clos=df.columns[0].split('\t')
	Log_header=df.ix[:,0]
	for x in xrange(0,len(orig_clos)):
		df[orig_clos[x]]=Log_header.apply(lambda line:line.split('\t')[x])
	
	df['s1']=df['EVT_LBL'].apply(lambda x:x.split('-')[0])
	df['s2']=df['EVT_LBL'].apply(lambda x:x.split('-')[1])
	df['s3']=df['EVT_LBL'].apply(lambda x:x.split('-')[2])
	# print df.head()
	# df['day']=df['OCC_TIM'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S').day)
	df['day']=df['OCC_TIM'].map(lambda x: int(x.split(' ')[0].split('-')[-1]))
	df['is_weekend']=df['OCC_TIM'].astype('str').apply(lambda x:1 if date(int(x[0:4]),int(x[5:7]),int(x[8:10])).weekday()+1>=6 else 0)
	df['is_wednesday']=df['day'].apply(lambda x:1 if x in[7,14,21,28] else 0)
	df=df[['USRID','EVT_LBL','s1','s2','s3','day','is_weekend','is_wednesday','OCC_TIM','TCH_TYP']]
	# df=df[['USRID','s1','s2','s3','day','OCC_TIM','TCH_TYP']]
	print df.head()
	inputfile=trainPath.split('/')[-1]
	df.to_csv('./train/cleaned_'+inputfile,index=None)
def trainAgg(trainPath):
	##分离独立每一个列，并设置每列的类型是浮点型,
	df=pd.read_csv(trainPath)
	orig_clos=df.columns[0].split('\t')
	Agg_header=df.ix[:,0]
	for x in xrange(0,len(orig_clos)):
		df[orig_clos[x]]=Agg_header.apply(lambda line:line.split('\t')[x])
	df.drop(df.columns[[0]],axis=1,inplace=True)
	# df=df[df.columns].astype(float)
	print df.info
	inputfile=trainPath.split('/')[-1]
	df.to_csv('./test/cleaned_'+inputfile,index=None)


def main():
	# explore_data('./train/train_agg2.csv')
	# trainFlag("./train/train_flg.csv")
	# trainAgg("./train/train_agg.csv")
	trainLog("./train/train_log.csv")
	trainLog("./test/test_log.csv")
	

if __name__ == '__main__':
	main()

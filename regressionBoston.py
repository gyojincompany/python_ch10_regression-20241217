import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression  # 선형회귀모델
from sklearn.model_selection import train_test_split  # train set, test set 데이터를 분리
from sklearn.metrics import mean_squared_error, r2_score  # 평가지표 MSE, R2

data_url = "http://lib.stat.cmu.edu/datasets/boston"

raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# \s+ -> 길이가 정해져 있지 않은 공백
# skiprows=22 -> 데이터가 아닌 데이터 설명라인 22줄 건너띄기
# header=None -> 열이름이 없는 데이터
print(raw_df)
# print(raw_df.values[0::2,:])  # 0, 2, 4.... 짝수행의 데이터만 슬라이싱
# print(raw_df.values[1::2,:2])  # 1, 3, 5.... 홀수행의 데이터 중 0,1,2 열만 슬라이싱
data = np.hstack([raw_df.values[0::2,:],raw_df.values[1::2,:2]])  # 독립변수 들

# print(raw_df.values[1::2, 2])  # MEDV->종속변수(집값)
target = raw_df.values[1::2, 2]  # 종속변수 주택 가격

print(data.shape)  # 506행 13열
print(target.shape)  # 506개

feature_names = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]# 독립변수들의 칼럼 이름

boston_df = pd.DataFrame(data, columns=feature_names)
print(boston_df)
boston_df["PRICE"] = target  # 종속변수인 주택가격을 PRICE 칼럼이름으로 추가
print(boston_df)


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

# 데이터 분리 -> 훈련데이터와 평가데이터로 7:3 비율로 분할
Y = boston_df["PRICE"]  # 종속변수->주택가격
X = boston_df.drop(["PRICE"], axis=1, inplace=False)  # inplace=True -> 원본 변경
# 종속변수 PRICE 열을 제외한 나머지 독립변수 13개의 열을 X에 저장

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)  # random_state 값 seed 값
# print(X_test)

lr = LinearRegression()  # 선형회귀분석 모델 생성
lr.fit(X_train, Y_train)

Y_predict = lr.predict(X_test)  # 훈련데이터 셋으로 만든 회귀모델에 평가데이터로 예측 수행
# print(Y_predict)

# 평가지표 MSE, RMSE, R2 값 구하기
mse = mean_squared_error(Y_test, Y_predict)  # MSE(Mean Squared Error)
rmse = np.sqrt(mse)  # RMSE->MSE의 제곱근
r2 = r2_score(Y_test, Y_predict)  # R Squared 값

print(f"MSE : {mse:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R2 Score : {r2:.3f}")

# 회귀계수 구하기
intercept = lr.intercept_  # 절편
coef = lr.coef_  # 회귀계수

print(f"Y 절편값 : {intercept}")
print(f"회귀 계수 : {coef}")
print(f"회귀식:Y(PRICE) = {intercept}+{coef[0]}X1+{coef[1]}X2+{coef[2]}X3+{coef[3]}X4+{coef[4]}X5+"
      f"{coef[5]}X6+{coef[6]}X7+{coef[7]}X8+{coef[8]}X9+{coef[9]}X10+{coef[10]}X11+{coef[11]}X12+{coef[12]}X13")

coef_X = pd.Series(data=coef, index=X.columns)
print(coef_X)

fig, axs = plt.subplots(figsize=(16, 16), nrows=5, ncols=3)  # 3*5=15개의 그래프(13개만 출력)

x_features = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"] # 그래프들의 이름

for i, feature in enumerate(x_features):
    row = int(i/3)# 행 인덱스(0, 1, 2, 3, 4)
    col = i%3# 열 인덱스(0, 1, 2)
    sns.regplot(x=feature, y="PRICE", data=boston_df, ax=axs[row][col])

plt.show()

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
df = pd.read_csv('cabbage_weather.csv')
df['date'] = pd.to_datetime(df['date'])
df['temp'] = df['temp'].interpolate()

# 2. 새로운 입력 변수 만들기 (feature engineering)
df['price_yesterday'] = df['price'].shift(1)      # 어제 가격
df['temp_7d'] = df['temp'].rolling(7).mean()       # 최근 7일 평균기온
df['rain_7d'] = df['rain'].rolling(7).mean()       # 최근 7일 평균강수량

# 3. 새 변수 만들면서 생긴 빈 값 제거
df = df.dropna().reset_index(drop=True)

# 4. 입력(X)과 정답(y)
X = df[['temp', 'rain', 'price_yesterday', 'temp_7d', 'rain_7d']]
y = df['price']

# 5. 훈련/시험 나누기 (앞 80% / 뒤 20%)
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 6. RandomForest 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. 예측 + 오차
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
print(f'평균 오차(MAE): {mae:.0f}원')

# 변수 중요도 확인
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print()
print('변수 중요도:')
for _, row in importance.iterrows():
    print(f"  {row['feature']}: {row['importance']*100:.1f}%")

# 변수 중요도 그래프
plt.figure(figsize=(8, 5))
plt.barh(importance['feature'], importance['importance'])
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('importance.png')
print('중요도 그래프 저장 완료: importance.png')

# 8. 그래프
plt.figure(figsize=(12, 5))
plt.plot(df['date'][split:], y_test.values, label='Actual', marker='o', markersize=3)
plt.plot(df['date'][split:], pred, label='Predicted', marker='x', markersize=3)
plt.title('Cabbage Price: Actual vs Predicted (improved)')
plt.xlabel('Date')
plt.ylabel('Price (won/10kg)')
plt.legend()
plt.grid(True)
plt.savefig('result.png')
print('그래프 저장 완료: result.png')
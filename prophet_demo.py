import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
df = pd.read_csv(url)

df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 获取图表对象并添加图例
fig = model.plot(forecast)
ax = fig.gca()
ax.set_title('温度预测')
ax.legend(['真实值', '预测值', '预测区间'], loc='upper left')

plt.show()

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 加载数据
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
df = pd.read_csv(url)

# 重命名列名
df.columns = ['ds', 'y'] #Prophet模型要求日期时间数据的列名必须是ds
# 转换日期格式
df['ds'] = pd.to_datetime(df['ds']) # 日期时间数据列的数据类型必须是datetime

# 创建prophet模型实例
model = Prophet()

# 训练模型
model.fit(df)

# 使用model.make_future_dataframe方法创建了一个包含未来365天日期的DataFrame，这些日期将用于预测
future = model.make_future_dataframe(periods=365)

# 使用model.predict方法对未来365天的日期进行预测
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

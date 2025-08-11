import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import chinese_calendar
import os

from matplotlib import font_manager
font_list = [f.name for f in font_manager.fontManager.ttflist]
if 'Microsoft YaHei' in font_list:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
elif 'SimHei' in font_list:
    plt.rcParams['font.sans-serif'] = ['SimHei']
else:
    print("⚠️ 未找到可用中文字体，中文可能无法正常显示")
plt.rcParams['axes.unicode_minus'] = False

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统备选字体

# 1. 读取数据
df = pd.read_csv(os.path.join("data", "PRSA_data_2010.1.1-2014.12.31.csv"))

# 2. 生成datetime列
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']]) # 将指定的列转换为日期时间格式

# 3. 新增日期列
df['date'] = df['datetime'].dt.date # 将datetime类型转换为date类型

# 4. 按天聚合计算均值
df_daily = df.groupby('date').agg({
    'pm2.5': 'mean',
    'TEMP': 'mean',
    'DEWP': 'mean',
    'PRES': 'mean',
    'Iws': 'mean',
}).reset_index()

# 5. 重命名列，符合Prophet格式
df_daily.rename(columns={'date': 'ds', 'pm2.5': 'y'}, inplace=True)
df_daily['ds'] = pd.to_datetime(df_daily['ds'])  # 转换为datetime类型

# 6. 选择需要的列
df_daily = df_daily[['ds', 'y', 'TEMP', 'DEWP', 'PRES', 'Iws']]

# 7. 处理缺失值
print("缺失值统计：")
print(df_daily.isnull().sum())
df_daily = df_daily.dropna() # 删除包含缺失值的行，更新DataFrame

# 8. 利用chinese_calendar自动生成节假日DataFrame
def generate_holiday_df(start_date, end_date):
    dates = pd.date_range(start_date, end_date)
    holidays = []
    for d in dates:
        if chinese_calendar.is_holiday(d):
            holidays.append({'holiday': 'chinese_holiday', 'ds': d, 'lower_window': 0, 'upper_window': 1})        
    return pd.DataFrame(holidays)

holidays = generate_holiday_df(df_daily['ds'].min(), df_daily['ds'].max())

# 9. 创建Prophet模型，添加节假日和回归变量
model = Prophet(holidays=holidays)
for regressor in ['TEMP', 'DEWP', 'PRES', 'Iws']:
    model.add_regressor(regressor)

# 10. 训练模型
model.fit(df_daily)

# 11. 生成未来365天日期
future = model.make_future_dataframe(periods=365)

# 12. 未来回归变量简单填充（用最后一天数据填充）
last_row = df_daily.iloc[-1]
for regressor in ['TEMP', 'DEWP', 'PRES', 'Iws']:
    future[regressor] = last_row[regressor]

# 13. 预测
forecast = model.predict(future)

# 14. 结果可视化
# 确保result目录存在
os.makedirs('result', exist_ok=True)

# 主预测图 - 保存并显示
fig1 = model.plot(forecast)
plt.title('未来一年每日PM2.5预测（含节假日影响）')
# 添加中文图例
ax = fig1.gca()
lines = ax.get_lines()
if len(lines) >= 1:
    lines[0].set_label('实际值')
if len(lines) >= 2:
    lines[1].set_label('预测值')
plt.legend()
plt.savefig('result/predict.png')
plt.show()
plt.close()

# 组件分析图 - 保存并显示
fig2 = model.plot_components(forecast)
plt.suptitle('PM2.5预测组件分析')
plt.savefig('result/components.png')
plt.show()
plt.close()

print("预测结果已保存至result文件夹")

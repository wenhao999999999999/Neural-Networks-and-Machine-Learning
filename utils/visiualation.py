import numpy as np
import matplotlib.pyplot as plt
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

# 读取结果文件
result_dir = "./results/informer_ETTh1_ftM_sl336_ll336_pl384_dm512_nh8_el3_dl2_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_'Exp'_0"  # 结果文件夹路径，根据实际情况修改
pred = np.load(os.path.join(result_dir, "pred.npy"))
true = np.load(os.path.join(result_dir, "true.npy"))
metrics = np.load(os.path.join(result_dir, "metrics.npy"))  # 可选


# 只画第一个样本的（适用于pred/true为3维或2维的情况）
pred_plot = pred[-0, :, -1]
true_plot = true[-0, :, -1]


plt.figure(figsize=(12, 6))
plt.plot(true_plot, label='真实值', color='blue', alpha=1, linewidth=1)
plt.plot(pred_plot, label='预测值', color='red', alpha=1, linewidth=1)
plt.title('Informer模型预测结果与真实值对比')
plt.xlabel('时间步')
plt.ylabel('数值')
plt.legend()
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

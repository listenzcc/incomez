# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 输入官方岗位工资分位数
# 单位：万元/年
# ==========================================

data = {
    "岗位": [
        "高层管理岗", "中层管理岗", "基层管理岗", "管理类员工岗",
        "高级职称", "中级职称", "初级职称",
        "高级技能及以上", "中级技能", "初级技能"
    ],
    "P10": [5.35, 4.85, 4.15, 3.60, 5.50, 4.44, 3.90, 4.38, 4.26, 3.39],
    "P25": [7.32, 6.48, 5.48, 4.80, 8.13, 5.96, 5.07, 5.82, 5.50, 4.46],
    "P50": [12.00, 9.83, 8.04, 6.67, 13.27, 8.86, 6.96, 8.25, 7.47, 6.00],
    "P75": [22.69, 16.64, 13.58, 10.02, 21.90, 14.22, 10.20, 12.00, 10.58, 8.41],
    "P90": [42.28, 28.50, 23.46, 15.82, 33.60, 21.43, 15.10, 17.25, 15.21, 11.66],
}

df = pd.DataFrame(data)

# 目标均值：城镇非私营单位平均工资
# 129,441 元/年 = 12.9441 万元/年
target_mean = 12.9441

# ==========================================
# 2. 根据分位数估计对数正态分布的 sigma
#
# 对数正态分布：
# ln(Q_p) = mu + sigma * z_p
#
# 同一岗位内做相对中位数的对数差，
# 可以消去该岗位的 mu。
# ==========================================

z = {
    "P10": -1.2815516,
    "P25": -0.6744898,
    "P50": 0.0,
    "P75": 0.6744898,
    "P90": 1.2815516,
}

x_values = []
y_values = []

for _, row in df.iterrows():
    median = row["P50"]

    for col in ["P10", "P25", "P75", "P90"]:
        # 以岗位中位数为基准，消去岗位自身的位置参数
        x_values.append(z[col])
        y_values.append(np.log(row[col] / median))

x_values = np.array(x_values)
y_values = np.array(y_values)

# 最小二乘估计共同 sigma，回归线通过原点
sigma = np.sum(x_values * y_values) / np.sum(x_values**2)

# 让模型均值精确等于目标均值
mu = np.log(target_mean) - sigma**2 / 2

model_mean = np.exp(mu + sigma**2 / 2)
model_median = np.exp(mu)

print("=== 模型参数 ===")
print(f"sigma = {sigma:.4f}")
print(f"mu    = {mu:.4f}")
print(f"模型均值 = {model_mean:.4f} 万元/年")
print(f"模型中位数 = {model_median:.4f} 万元/年")
print(f"模型均值 = {model_mean * 10000:,.0f} 元/年")

# ==========================================
# 3. 计算模型分位数
# ==========================================

percentiles = {
    "P10": -1.2815516,
    "P25": -0.6744898,
    "P50": 0.0,
    "P75": 0.6744898,
    "P90": 1.2815516,
}

result = pd.DataFrame({
    "分位数": list(percentiles.keys()),
    "模型值（万元/年）": [
        np.exp(mu + sigma * z_value)
        for z_value in percentiles.values()
    ],
})

result["模型值（元/年）"] = result["模型值（万元/年）"] * 10000
result["模型值（元/月）"] = result["模型值（元/年）"] / 12

print("\n=== 模型分位数 ===")
print(result.round(2).to_string(index=False))

result.to_csv(
    "synthetic_nonprivate_wage_quantiles.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 4. 绘制年工资概率密度曲线
# ==========================================

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC"
]
plt.rcParams["axes.unicode_minus"] = False

x = np.linspace(0.05, 70, 2000)  # 万元/年

pdf = (
    np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma**2))
    / (x * sigma * np.sqrt(2 * np.pi))
)

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.plot(x, pdf, linewidth=2.5, label="合成工资分布")

ax.axvline(
    target_mean,
    linestyle="--",
    linewidth=1.5,
    label=f"目标均值：{target_mean:.4f} 万元"
)

ax.axvline(
    model_median,
    linestyle=":",
    linewidth=1.5,
    label=f"模型中位数：{model_median:.2f} 万元"
)

i = np.argmax(pdf)
x_max = x[i]
pdf_max = pdf[i]
print(f'{x_max=}, {pdf_max=}')

ax.axvline(
    x_max,
    linestyle="-.",
    color='red',
    linewidth=1.5,
    label=f"模型极值点：{x_max:.2f} 万元"
)

ax.set_title("城镇非私营单位工资：对数正态长尾模型")
ax.set_xlabel("年工资（万元）")
ax.set_ylabel("概率密度")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()

fig.savefig(
    "synthetic_nonprivate_wage_distribution.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()

# %%
# %%

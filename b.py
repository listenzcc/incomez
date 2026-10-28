
# %%
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 按优先级查找系统中已安装的中文字体
font_candidates = [
    "Microsoft YaHei",  # Windows
    "SimHei",           # Windows 黑体
    "PingFang SC",      # macOS
    "Heiti SC",         # macOS
    "Noto Sans CJK SC",  # Linux
    "Noto Sans SC",
    "WenQuanYi Zen Hei"  # 部分 Linux
]

installed_fonts = {
    font.name for font in font_manager.fontManager.ttflist
}

for font in font_candidates:
    if font in installed_fonts:
        plt.rcParams["font.sans-serif"] = [font]
        print(f"使用中文字体：{font}")
        break
else:
    print("未找到常见中文字体，请安装中文字体后重试。")

# 解决坐标轴负号显示为方块的问题
plt.rcParams["axes.unicode_minus"] = False
# %%
# 单位：元/人/年
data = {
    "年份": [2021, 2022, 2023, 2024, 2025],
    "城镇均值": [47412, 49283, 51821, 54188, 56502],
    "城镇中位数": [43504, 45123, 47122, 49302, 51115],
    "农村均值": [18931, 20133, 21691, 23119, 24456],
    "农村中位数": [16902, 17734, 18748, 19605, 20711],
}

df = pd.DataFrame(data)


def fit_lognormal(mean_income, median_income):
    """
    用收入均值和中位数估计对数正态分布参数。
    ln(X) ~ Normal(mu, sigma^2)
    """
    if mean_income <= 0 or median_income <= 0:
        raise ValueError("均值和中位数必须为正数")

    if mean_income < median_income:
        raise ValueError("均值小于中位数，请检查输入数据")

    mu = np.log(median_income)
    sigma = np.sqrt(2 * np.log(mean_income / median_income))

    return mu, sigma


def lognormal_pdf(x, mu, sigma):
    """对数正态分布的概率密度函数。"""
    x = np.asarray(x, dtype=float)

    return np.where(
        x > 0,
        np.exp(-(np.log(np.maximum(x, 1e-300)) - mu) ** 2
               / (2 * sigma**2))
        / (np.maximum(x, 1e-300) * sigma * np.sqrt(2 * np.pi)),
        0.0
    )


# 一、逐年拟合城镇和农村收入分布
params = {}
result = []

for _, row in df.iterrows():
    year = int(row["年份"])

    for group in ["城镇", "农村"]:
        mean_income = row[f"{group}均值"]
        median_income = row[f"{group}中位数"]

        mu, sigma = fit_lognormal(mean_income, median_income)
        params[(year, group)] = (mu, sigma)

        result.append({
            "年份": year,
            "地区": group,
            "mu": mu,
            "sigma": sigma,
            "原始均值": mean_income,
            "拟合均值": np.exp(mu + sigma**2 / 2),
            "原始中位数": median_income,
            "拟合中位数": np.exp(mu),
        })

result_df = pd.DataFrame(result)

print("拟合参数与拟合检查：")
print(result_df.round(2).to_string(index=False))


# 二、绘制2021—2025年收入分布
# 每年一行，城镇和农村各一张图
fig, axes = plt.subplots(5, 2, figsize=(13, 20))

for i, year in enumerate(df["年份"]):
    for j, group in enumerate(["城镇", "农村"]):
        ax = axes[i, j]

        mu, sigma = params[(int(year), group)]
        row = df[df["年份"] == year].iloc[0]

        mean_income = row[f"{group}均值"]
        median_income = row[f"{group}中位数"]

        # 横轴延伸到理论99.5%分位数，以展示右侧长尾
        x_max = np.exp(mu + 2.576 * sigma)
        x = np.linspace(1, x_max, 1500)
        y = lognormal_pdf(x, mu, sigma)

        ax.plot(x, y, label="对数正态拟合")
        ax.axvline(
            median_income, linestyle=":",
            label=f"中位数 {median_income:,.0f}"
        )
        ax.axvline(
            mean_income, linestyle="--",
            label=f"均值 {mean_income:,.0f}"
        )

        ax.set_title(f"{year}年{group}居民收入")
        ax.set_xlabel("年人均可支配收入（元）")
        ax.set_ylabel("概率密度")
        ax.legend()

plt.tight_layout()
fig.savefig(
    f"{ax.title.get_text()}.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()


# 三、计算理论收入分位数
# 这些是模型推算值，不是官方公布的实际分位数
quantile_levels = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
quantile_rows = []

for year in df["年份"]:
    for group in ["城镇", "农村"]:
        mu, sigma = params[(int(year), group)]

        row = {
            "年份": int(year),
            "地区": group,
        }

        for q in quantile_levels:
            # 对数正态分布的理论分位数
            z = {
                0.10: -1.2815516,
                0.25: -0.6744898,
                0.50: 0.0,
                0.75: 0.6744898,
                0.90: 1.2815516,
                0.95: 1.6448536,
                0.99: 2.3263479,
            }[q]

            row[f"P{int(q * 100)}"] = np.exp(mu + sigma * z)

        quantile_rows.append(row)

quantile_df = pd.DataFrame(quantile_rows)

print("\n各年收入分位数估计（元/人/年）：")
print(quantile_df.round(0).to_string(index=False))


# 四、可选：生成模拟收入样本
# 模拟样本仅用于研究模型行为，不代表真实居民微观调查数据
rng = np.random.default_rng(42)
sample_size = 100_000

samples = {}

for key, (mu, sigma) in params.items():
    samples[key] = rng.lognormal(
        mean=mu,
        sigma=sigma,
        size=sample_size
    )

# 示例：2025年城镇和农村模拟收入的描述统计
for group in ["城镇", "农村"]:
    sample = samples[(2025, group)]

    print(f"\n2025年{group}模拟收入：")
    print(pd.Series(sample).describe(
        percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    ).round(0))


# %%


# 中文字体：按系统安装情况选择

font_candidates = [
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
]

installed_fonts = {f.name for f in font_manager.fontManager.ttflist}
for font in font_candidates:
    if font in installed_fonts:
        plt.rcParams["font.sans-serif"] = [font]
        break

plt.rcParams["axes.unicode_minus"] = False

years = [2021, 2022, 2023, 2024, 2025]

# 年份用颜色区分
colors = {
    2021: "tab:blue",
    2022: "tab:orange",
    2023: "tab:green",
    2024: "tab:red",
    2025: "tab:purple",
}

# 城镇、农村用线型区分
line_styles = {
    "城镇": "-",
    "农村": "--",
}

# 均值和中位数用不同标记
stat_markers = {
    "均值": "o",
    "中位数": "s",
}

fig, ax = plt.subplots(figsize=(14, 8))

# 横轴范围覆盖所有分布的99.5%分位数
x_max = max(
    np.exp(mu + 2.576 * sigma)
    for mu, sigma in params.values()
)
x = np.linspace(1, x_max, 2500)

# 绘制十条收入分布曲线
for year in years:
    row = df.loc[df["年份"] == year].iloc[0]

    for group in ["城镇", "农村"]:
        mu, sigma = params[(year, group)]
        y = lognormal_pdf(x, mu, sigma)

        ax.plot(
            x, y,
            color=colors[year],
            linestyle=line_styles[group],
            linewidth=2,
        )

        # 取原始统计数据，而不是模拟样本
        mean_income = row[f"{group}均值"]
        median_income = row[f"{group}中位数"]

        # 标记均值和中位数对应的横坐标
        for stat_name, income in [
            ("均值", mean_income),
            ("中位数", median_income),
        ]:
            density = lognormal_pdf(
                np.array([income]), mu, sigma
            )[0]

            # 在分布曲线上标记该收入值
            ax.scatter(
                income,
                density,
                color=colors[year],
                marker=stat_markers[stat_name],
                s=65,
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
            )

# 图例一：年份对应颜色
year_handles = [
    Line2D(
        [0], [0],
        color=colors[year],
        linewidth=2,
        label=str(year),
    )
    for year in years
]

# 图例二：地区对应线型
group_handles = [
    Line2D(
        [0], [0],
        color="black",
        linestyle=line_styles[group],
        linewidth=2,
        label=group,
    )
    for group in ["城镇", "农村"]
]

# 图例三：均值和中位数对应标记
stat_handles = [
    Line2D(
        [0], [0],
        color="black",
        marker=stat_markers["均值"],
        linestyle="None",
        markersize=7,
        label="均值",
    ),
    Line2D(
        [0], [0],
        color="black",
        marker=stat_markers["中位数"],
        linestyle="None",
        markersize=7,
        label="中位数",
    ),
]

legend1 = ax.legend(
    handles=year_handles,
    title="年份（颜色）",
    loc="upper right",
)
ax.add_artist(legend1)

legend2 = ax.legend(
    handles=group_handles,
    title="地区（线型）",
    loc="upper center",
)
ax.add_artist(legend2)

ax.legend(
    handles=stat_handles,
    title="统计量（标记）",
    loc="upper left",
)

ax.set_title("2021—2025年城镇与农村居民收入分布")
ax.set_xlabel("年人均可支配收入（元）")
ax.set_ylabel("概率密度")
ax.grid(True, alpha=0.25)

plt.tight_layout()
fig.savefig(
    f"{ax.title.get_text()}-dots.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()

# %%


# =========================
# 1. 中文字体设置
# =========================
font_candidates = [
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
]

installed_fonts = {f.name for f in font_manager.fontManager.ttflist}

for font in font_candidates:
    if font in installed_fonts:
        plt.rcParams["font.sans-serif"] = [font]
        break

plt.rcParams["axes.unicode_minus"] = False

# =========================
# 2. 绘图参数
# =========================
years = [2021, 2022, 2023, 2024, 2025]

# 颜色区分年份
colors = {
    2021: "tab:blue",
    2022: "tab:orange",
    2023: "tab:green",
    2024: "tab:red",
    2025: "tab:purple",
}

# 线型区分城乡
line_styles = {
    "城镇": "-",
    "农村": "--",
}

# 标记区分均值和中位数
markers = {
    "均值": "o",
    "中位数": "s",
}

fig, ax = plt.subplots(figsize=(14, 8))

# 统一横轴范围，延伸到最高的理论99.5%分位数
x_max = max(
    np.exp(mu + 2.576 * sigma)
    for mu, sigma in params.values()
)
x = np.linspace(1, x_max, 2500)

# =========================
# 3. 绘制收入分布曲线
# =========================
for year in years:
    row = df.loc[df["年份"] == year].iloc[0]

    for group in ["城镇", "农村"]:
        mu, sigma = params[(year, group)]
        density = lognormal_pdf(x, mu, sigma)

        # 收入分布曲线
        ax.plot(
            x,
            density,
            color=colors[year],
            linestyle=line_styles[group],
            linewidth=2,
        )

        # 标记均值和中位数
        mean_income = row[f"{group}均值"]
        median_income = row[f"{group}中位数"]

        for name, income in [
            ("均值", mean_income),
            ("中位数", median_income),
        ]:
            y = lognormal_pdf(
                np.array([income]), mu, sigma
            )[0]

            ax.scatter(
                income,
                y,
                color=colors[year],
                marker=markers[name],
                s=55,
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
            )

# =========================
# 4. 绘制错层的“中位数 → 均值”横线
# =========================
y_bottom, y_top = ax.get_ylim()
y_range = y_top - y_bottom

# 将横线放在图底部，并分成10个高度层
y_base = y_bottom + y_range * 0.025
y_step = y_range * 0.012
tick_height = y_range * 0.004

for year_idx, year in enumerate(years):
    row = df.loc[df["年份"] == year].iloc[0]

    # 每年占两个高度层；城镇和农村分别错开
    level = year_idx * 2 + group_idx
    y_bar = y_base - level * y_step

    for group_idx, group in enumerate(["城镇", "农村"]):
        median_income = row[f"{group}中位数"]
        mean_income = row[f"{group}均值"]

        # 中位数到均值的横线
        ax.hlines(
            y=y_bar,
            xmin=median_income,
            xmax=mean_income,
            color=colors[year],
            linewidth=2.5,
            zorder=6,
        )

        # 横线两端的短竖线
        ax.vlines(
            [median_income, mean_income],
            ymin=y_bar - tick_height,
            ymax=y_bar + tick_height,
            color=colors[year],
            linewidth=1.2,
            zorder=6,
        )

        ax.text(x=mean_income, y=y_bar,
                s=f' +{mean_income-median_income}', verticalalignment='center')

# 扩展纵轴下限，确保所有错层横线完整显示
ax.set_ylim(
    # y_bottom,
    y_bar - 2*y_step,
    max(y_top, y_base + 9 * y_step + 2 * tick_height)
)

# =========================
# 5. 设置图例
# =========================
# 年份：颜色
year_handles = [
    Line2D(
        [0], [0],
        color=colors[year],
        linewidth=2,
        label=str(year),
    )
    for year in years
]

# 城乡：线型
group_handles = [
    Line2D(
        [0], [0],
        color="black",
        linestyle=line_styles[group],
        linewidth=2,
        label=group,
    )
    for group in ["城镇", "农村"]
]

# 统计量：标记形状
stat_handles = [
    Line2D(
        [0], [0],
        color="black",
        marker=markers["均值"],
        linestyle="None",
        markersize=7,
        label="均值",
    ),
    Line2D(
        [0], [0],
        color="black",
        marker=markers["中位数"],
        linestyle="None",
        markersize=7,
        label="中位数",
    ),
]

legend1 = ax.legend(
    handles=year_handles,
    title="年份（颜色）",
    loc="upper right",
)
ax.add_artist(legend1)

legend2 = ax.legend(
    handles=group_handles,
    title="地区（线型）",
    loc="upper center",
)
ax.add_artist(legend2)

ax.legend(
    handles=stat_handles,
    title="统计量（标记）",
    loc="upper left",
)

# =========================
# 6. 标题与坐标轴
# =========================
ax.set_title("2021—2025年城镇与农村居民收入分布")
ax.set_xlabel("年人均可支配收入（元）")
ax.set_ylabel("概率密度")
ax.grid(True, alpha=0.25)

plt.tight_layout()
fig.savefig(
    f"{ax.title.get_text()}-dots.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()

# %%
# %%

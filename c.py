# %%

# 二、所有年份画在同一张图上
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# 年份颜色
years = [2021, 2022, 2023, 2024, 2025]
colors = {
    2021: "tab:blue",
    2022: "tab:orange",
    2023: "tab:green",
    2024: "tab:red",
    2025: "tab:purple",
}

# 城镇实线，农村虚线
line_styles = {
    "城镇": "-",
    "农村": "--",
}

fig, ax = plt.subplots(figsize=(12, 7))

# 使用统一的横轴范围，方便直接比较
x_max = max(
    np.exp(mu + 2.576 * sigma)
    for mu, sigma in params.values()
)
x = np.linspace(1, x_max, 2000)

for year in years:
    for group in ["城镇", "农村"]:
        mu, sigma = params[(year, group)]
        y = lognormal_pdf(x, mu, sigma)

        ax.plot(
            x,
            y,
            color=colors[year],
            linestyle=line_styles[group],
            linewidth=2,
        )

ax.set_title("2021—2025年城镇与农村居民收入分布")
ax.set_xlabel("年人均可支配收入（元）")
ax.set_ylabel("概率密度")
ax.grid(True, alpha=0.25)

# 图例一：颜色代表年份
year_handles = [
    Line2D(
        [0], [0],
        color=colors[year],
        linewidth=2,
        label=str(year)
    )
    for year in years
]

# 图例二：线型代表城乡
group_handles = [
    Line2D(
        [0], [0],
        color="black",
        linestyle=line_styles[group],
        linewidth=2,
        label=group
    )
    for group in ["城镇", "农村"]
]

legend1 = ax.legend(
    handles=year_handles,
    title="年份（颜色）",
    loc="upper right"
)
ax.add_artist(legend1)

ax.legend(
    handles=group_handles,
    title="地区（线型）",
    loc="upper center"
)

plt.tight_layout()
plt.show()

# %%

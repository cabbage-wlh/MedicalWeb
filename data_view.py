import matplotlib.pyplot as plt
import pandas as pd

# CSV文件路径
csv_file_path = "D:/毕设/py-django/flaskProject/data/view.csv"


def age_view():
    data = pd.read_csv(csv_file_path)
    # 假设CSV文件中的年龄列名为'年龄'
    # 确保年龄列是数值类型
    data["年龄"] = pd.to_numeric(data["年龄"], errors="coerce")
    data_list = data["年龄"].to_list()
    # 定义年龄区间的最小值和最大值
    age_groups = {i: 0 for i in range(0, 101, 10)}

    # 计算每个年龄段的人数
    for age in data_list:
        for group in range(0, 101, 10):
            if age >= group and age < group + 10:
                age_groups[group] += 1

    # 提取年龄段和对应的人数
    x = list(age_groups.keys())
    y = list(age_groups.values())
    # print("age")
    # print(x,y)

    # 创建条形图
    bars = plt.bar(x, y, width=8, align="center", color="skyblue")

    # 设置图形标题和坐标轴标签
    plt.title("Age Distribution")
    plt.xlabel("Age Group")
    plt.ylabel("Number of People")

    # 在每个柱状图顶部显示统计人数
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, height, height, ha="center", va="bottom"
        )

    # 设置横坐标刻度标签
    plt.xticks(range(0, 101, 10), [f"{i}-{i + 9}" for i in range(0, 101, 10)])
    return x,y
    # 显示图形
    # plt.savefig("age_distribution_with_count.png")
    # plt.show()


def few_category_view():
    data = pd.read_csv(csv_file_path)
    few_category_data = {
        "race": data[
            "种族"
        ].value_counts(),  # [AMERICAN INDIAN OR ALASKA NATIV, ASIAN, BLACK OR AFRICAN AMERICAN, WHITE]
        "gender": data["性别"].value_counts(),  # [FEMALE, MALE]
        "survived": data["生存状态"].value_counts(),  # [0, 1]
        "tumor": data["肿瘤状态"].value_counts(),  # [TUMOR FREE, WITH TUMOR]
        "receptor": data["孕激素受体"].value_counts(),  # [Negative, Indeterminate, Positive]
        "neoadjuvant therapy": data["新辅助治疗"].value_counts(),  # [No, Yes]
        "surgery type": data[
            "手术类型"
        ].value_counts(),  # [Lumpectomy, Simple Mastectomy, Modified Radical Mastectomy, Other]
        'position':data['肿瘤位置'].value_counts(),
        'number':data['淋巴结数量'].value_counts(),

    }
    categories1 = ["gender", "survived", "tumor", "neoadjuvant therapy"]
    categories2 = ["race", "receptor", "surgery type"]
    categories3 = ["position",]
    return few_category_data, categories1,categories2
    for category in [categories1, categories2]:
        for sub_category in category:
            print(few_category_data[sub_category])
    color_dict = {
        "categories1": ["#F1DBE7", "#E0F1F7", "#DBD8E9", "#DEECD9"],
        "categories2": ["#A5C496", "#C7988C", "#8891DB"],
    }
    # plt.figure(figsize=(20, 10))
    # 创建第一个图
    fig1, ax1 = plt.subplots(figsize=(16, 8))
    for category, color in zip(categories1, color_dict["categories1"]):
        bars = ax1.bar(
            few_category_data[category].index.astype(str),
            few_category_data[category].values,
            color=color,
            label=category,
        )
        for bar in bars:
            yval = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval,
                int(yval),
                ha="center",
                va="bottom",
            )  # va: vertical alignment

    ax1.set_xlabel("Sub-Category")
    ax1.set_ylabel("Counts")
    ax1.set_title("Two-category category visualization")
    ax1.legend(loc="upper right")
    plt.savefig("part-1.png")

    # 创建第二个图
    fig2, ax2 = plt.subplots(figsize=(27, 18))
    for category, color in zip(categories2, color_dict["categories2"]):
        bars = ax2.bar(
            few_category_data[category].index.astype(str),
            few_category_data[category].values,
            color=color,
            label=category,
        )
        for bar in bars:
            yval = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval,
                int(yval),
                ha="center",
                va="bottom",
                fontsize=20,
            )  # va: vertical alignment

    ax2.set_xlabel("Sub-Category", fontsize=24)
    ax2.set_ylabel("Counts", fontsize=24)
    ax2.set_title("Three/Four-classification category visualization", fontsize=24)
    ax2.legend(loc="upper right", fontsize=24)
    # plt.xticks(fontsize=14)  # Set x-axis tick label font size
    plt.yticks(fontsize=20)
    plt.savefig("part-2.png")


# 调用函数
age_view()
few_category_view()

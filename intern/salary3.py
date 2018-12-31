import pandas as pd

from pyecharts import Bar, Line, Overlap, Pie

"""
对salary中对数据进行清洗和初步分析,
主要从工种的角度进行男性、女性工资差异的定量分析
绘制salary的图
"""

salaryInfo = pd.DataFrame(
    pd.read_csv('/Users/helena/Documents/french-employment-by-town/net_salary_per_town_categories.csv'))

# 查找是否有空值，以下没有nan值
# print(salaryInfo.isnull())

# 在将salary进行预处理, 先从性别的角度考虑
salaryInfo = salaryInfo[salaryInfo["CODGEO"].apply(lambda x: str(x).isdigit())]
salaryInfo["CODGEO"] = salaryInfo["CODGEO"].astype(int)
# salaryInfo_a = salaryInfo.set_index('CODGEO')
gender_salary = salaryInfo.ix[:,
                ['LIBGEO', 'SNHMC14', 'SNHMP14', 'SNHME14', 'SNHMO14', 'SNHMFC14', 'SNHMFP14', 'SNHMFE14', 'SNHMFO14',
                 'SNHMHC14', 'SNHMHP14', 'SNHMHE14', 'SNHMHO14']]
gender_salary = gender_salary.rename(
    columns={'SNHMC14': 'mean_exective', 'SNHMP14': 'mean_middle_manager', 'SNHME14': 'mean_employee',
             'SNHMO14': 'mean_worker',
             'SNHMFC14': 'f_exective', 'SNHMFP14': 'f_middle_manager', 'SNHMFE14': 'f_employee', 'SNHMFO14': 'f_worker',
             'SNHMHC14': 'm_exective', 'SNHMHP14': 'm_middle_manager', 'SNHMHE14': 'm_employee',
             'SNHMHO14': 'm_worker',
             'LIBGEO': 'Town_name'})
# print(gender_salary.head(5))
# 男性工资
gender_salary["m_salary"] = gender_salary["m_exective"] + gender_salary["m_middle_manager"] + gender_salary[
    "m_employee"] + gender_salary["m_worker"]
gender_salary["f_salary"] = gender_salary["f_exective"] + gender_salary["f_middle_manager"] + gender_salary[
    "f_employee"] + gender_salary["f_worker"]
gender_salary["total_salary"] = gender_salary["m_salary"] + gender_salary["f_salary"]
gender_salary = gender_salary.ix[:, ['m_exective', 'm_middle_manager', 'm_employee', 'm_worker',
                                     'f_exective', 'f_middle_manager', 'f_employee', 'f_worker',
                                     'Town_name', 'm_salary', 'f_salary', 'total_salary',
                                     'mean_exective', 'mean_middle_manager', 'mean_employee', 'mean_worker']]

high_salary = gender_salary.sort_values(["total_salary"], ascending=False).head(10)
x = high_salary["Town_name"].values.tolist()
print(x)
y_total = high_salary["total_salary"].values.tolist()
y_male = high_salary["m_salary"].values.tolist()
y_female = high_salary["f_salary"].values.tolist()

# 总工资最高的top-10 城市
# print(y_total)
y_total = [252.8, 245.7, 227.6, 218.8, 217.2, 205.5, 202.1, 195.7, 192.9, 190.7]
bar1 = Bar(" Top-10 Cities with High Salaries in France")
bar1.add("Salaries per hour", x, y_total, mark_line=["average"], bar_category_gap='25%',
         xaxis_rotate=20, legend_pos='right')

line1 = Line("Top-10 Cities with High Salaries in France")
line1.add("value", x, y_total, is_stack=True, is_label_show=True, legend_pos='right')

overlap = Overlap(width=1200, height=600)
overlap.add(bar1)
overlap.add(line1, yaxis_index=1, is_add_yaxis=True)
overlap.render()

# top-10的城市中男性和女性的工资差异水平
bar2 = Bar("Gender Differences in Salary")
bar2.add("male", x, y_male, mark_line=["average"], bar_category_gap='25%',
         xaxis_rotate=20, legend_pos='right')
bar2.add("female", x, y_female, mark_line=["average"], bar_category_gap='25%',
         xaxis_rotate=20, legend_pos='right')
bar2.render()

# 对top-1的城市用饼状图单独分析，看不同工种之间的工资差异
print(high_salary.iloc[0])
# m_exective                             58
# m_middle_manager                     56.3
# m_employee                           16.6
# m_worker                             35.3
# f_exective                           31.8
# f_middle_manager                     18.6
# f_employee                           14.9
# f_worker                             21.3
# Town_name           Saint-Nom-la-Bretèche
# m_salary                            166.2
# f_salary                             86.6
# total_salary                        252.8
# mean_exective                        51.5
# mean_middle_manager                  37.2
# mean_employee                        15.6
# mean_worker                          33.5
# Name: 4225, dtype: object
attr = ["m_exective", "m_middle_manager", "m_employee", "m_worker",
        "f_exective", "f_middle_manager", "f_employee", "f_worker"]
v1 = [58, 56.3, 16.6, 35.3, 31.8, 18.6, 14.9, 21.3]
pie = Pie("Salary Differences Between Work Types & Gender")
pie.add("Saint-Nom-la-Bretèche", attr, v1, is_label_show=True, radius=[30, 75],
        legend_orient='vertical', legend_pos='right', rosetype='radius')
pie.render()

# Saint-Nom-la-Bretèche 的不同工种的平均工资的差异
# attr2 = ["exective","middle_manager","employee",'worker']
# v2 = [51.5,37.2,15.6,33.5]
# pie2 = Pie("Salary Differences per category")
# pie2.add("Saint-Nom-la-Bretèche", attr2, v2, is_label_show=True, radius=[30, 75],
#          legend_orient='vertical', legend_pos='right', rosetype='radius')
# pie2.render()

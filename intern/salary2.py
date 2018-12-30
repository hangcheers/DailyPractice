import pandas as pd
from pyecharts import Bar, Line, Overlap

"""
对salary中对数据进行清洗和初步分析,
主要从年龄段（18-25、26-50、>50）的角度进行男性、女性工资差异的定性分析
绘制salary的图
"""

salaryInfo = pd.DataFrame(
    pd.read_csv('/Users/helena/Documents/french-employment-by-town/net_salary_per_town_categories.csv'))
salaryInfo = salaryInfo[salaryInfo["CODGEO"].apply(lambda x: str(x).isdigit())]
salaryInfo["CODGEO"] = salaryInfo["CODGEO"].astype(int)
age_salary = salaryInfo.ix[:, ['LIBGEO', 'SNHMF1814', 'SNHMF2614', 'SNHMF5014',
                               'SNHMH1814', 'SNHMH2614', 'SNHMH5014',
                               'SNHM1814', 'SNHM2614', 'SNHM5014']]
age_salary = age_salary.rename(
    columns={'LIBGEO': 'Town_name', 'SNHMF1814': 'f_age18-25', 'SNHMF2614': 'f_age26-50', 'SNHMF5014': 'f_age>50',
             'SNHMH1814': 'm_age18-25', 'SNHMH2614': 'm_age26-50', 'SNHMH5014': 'm_age>50',
             'SNHM1814': 'mean_age18-25', 'SNHM2614': 'mean_age26-50', 'SNHM5014': 'mean_age>50'
             }
)
attr = ["age 18 ~ 25", "age 25 ~ 50", "age >50"]
columns1 = ["m_age18-25", "m_age26-50", "m_age>50"]
male_mean_salary = age_salary[columns1].mean().tolist()
# print(male_mean_salary)
# list中的元素保留两位小数
y1_male = [9.81, 14.49, 17.69]
# for i in range(len(male_mean_salary)):
#     male_mean_salary1=[]
#     b=[round(male_mean_salary[i],2)]
#     male_mean_salary1=male_mean_salary1+b
# print(male_mean_salary1)
columns2 = ["f_age18-25", "f_age26-50", "f_age>50"]
female_mean_salary = age_salary[columns2].mean().tolist()
# print(female_mean_salary)
y2_female = [9.16, 12.06, 13.18]
columns3 = ["mean_age18-25", "mean_age26-50", "mean_age>50"]
mean_salary = age_salary[columns3].mean().tolist()
# print(mean_salary)
y3_mean = [9.55, 13.50, 15.88]

# bar 1 表示男性、女性在三个年龄段的工资差异
bar1 = Bar(" wage difference  ")
bar1.add("male", attr, y1_male, mark_line=["average"], is_label_show=True, bar_category_gap='45%',
         yaxis_name='Net Salary Per Hour', yaxis_margin=2)
bar1.add("female", attr, y2_female, mark_line=["average"], is_label_show=True, bar_category_gap='45%',
         yaxis_name='Net Salary Per Hour', yaxis_margin=2)

# line 1 表示在三个年龄段的平均工资差异
line1 = Line("mean salary")
line1.add("mean_middle_manager", attr, y3_mean, is_stack=True, is_label_show=True)
overlap = Overlap(width=800, height=400)
overlap.add(bar1)
overlap.add(line1, yaxis_index=1, is_add_yaxis=True)
overlap.render()

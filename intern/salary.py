import pandas as pd
from pyecharts import Bar, Line, Overlap

"""
对salary中对数据进行清洗和初步分析,
主要从区域(即town)的角度进行男性、女性工资差异的定量分析
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
high_salary = gender_salary.sort_values(by=["m_exective"], ascending=False).head(5)
# print(high_salary)
high_salary1 = gender_salary.sort_values(by=["m_middle_manager"], ascending=False).head(5)
high_salary2 = gender_salary.sort_values(by=["m_employee"], ascending=False).head(5)
high_salary3 = gender_salary.sort_values(by=["m_worker"], ascending=False).head(5)
# 高工资城市名称
high_salary_town = high_salary["Town_name"].values.tolist()
high_salary_town1 = high_salary1["Town_name"].values.tolist()
high_salary_town2 = high_salary2["Town_name"].values.tolist()
high_salary_town3 = high_salary3["Town_name"].values.tolist()
# 不同岗位男性工资
high_salary_m_exective = high_salary["m_exective"].values.tolist()
high_salary_m_middle_manager = high_salary1["m_middle_manager"].values.tolist()
high_salary_m_employee = high_salary2["m_employee"].values.tolist()
high_salary_m_worker = high_salary3["m_worker"].values.tolist()
# 不同岗位女性工资
high_salary_f_exective = high_salary["f_exective"].values.tolist()
high_salary_f_middle_manager = high_salary1["f_middle_manager"].values.tolist()
high_salary_f_employee = high_salary2["f_employee"].values.tolist()
high_salary_f_worker = high_salary3["f_worker"].values.tolist()
# 不同岗位平均工资
mean_exective = high_salary["mean_exective"].values.tolist()
mean_middle_manager = high_salary1["mean_middle_manager"].values.tolist()
mean_employee = high_salary2["mean_employee"].values.tolist()
mean_worker = high_salary3["mean_worker"].values.tolist()

# 所有town在一起得到的不同岗位的平均工资
attr = ["mean_exective", "mean_middle_manager", "mean_employee", "mean_worker"]
mean_salary = high_salary[attr].mean().tolist()
# print(mean_salary)
mean_salary = [47.94, 25.72, 15.68, 29.96]
attr_m = ["m_exective", "m_middle_manager", "m_employee", "m_worker"]
mean_salary1 = high_salary[attr_m].mean().tolist()
attr_f = ["f_exective", "f_middle_manager", "f_employee", "f_worker"]
mean_salary2 = high_salary[attr_f].mean().tolist()
# print(mean_salary2)
mean_salary2 = [32.26, 17.64, 14.54, 18.34]

# 不同工种的男性和女性的工资差异用柱状图表示出来
# bar 1 表示executive职位的男女差别
attr1 = high_salary_town

y_male1 = high_salary_m_exective
y_female1 = high_salary_f_exective
y_mean1 = mean_exective

bar1 = Bar(" gender difference in executive ")
bar1.add("male_executive", attr1, y_male1, mark_line=["average"], is_label_show=True, bar_category_gap='35%',
         xaxis_rotate=20, yaxis_name='Net Salary Per Hour for Executive', yaxis_margin=2)
bar1.add("female_executive", attr1, y_female1, mark_line=["average"], is_label_show=True, bar_category_gap='35%',
         xaxis_rotate=20, yaxis_name='Net Salary Per Hour for Executive', yaxis_margin=2)

# line 1 表示executive的平均工资
line1 = Line("mean salary")
line1.add("mean_executive", attr1, y_mean1, is_stack=True, is_label_show=True)

overlap = Overlap(width=1200, height=600)
overlap.add(bar1)
overlap.add(line1, yaxis_index=1, is_add_yaxis=True)
overlap.render()

# bar 2 表示middle_management职位的男女差别
attr2 = high_salary_town1
y_male2 = high_salary_m_middle_manager
y_female2 = high_salary_f_middle_manager
y_mean2 = mean_middle_manager

bar2 = Bar(" gender difference in middle_manager ")
bar2.add("male_middle_manager", attr2, y_male2, mark_line=["average"], is_label_show=True, bar_category_gap='35%',
         xaxis_rotate=20, yaxis_name='Net Salary Per Hour for Middle Manager', yaxis_margin=2)
bar2.add("female_middle_manager", attr2, y_female2, mark_line=["average"], is_label_show=True, bar_category_gap='35%',
         xaxis_rotate=20, yaxis_name='Net Salary Per Hour for Middle Manager', yaxis_margin=2)

# line 2 表示middle_management的平均工资
line2 = Line("mean salary")
line2.add("mean_middle_manager", attr2, y_mean2, is_stack=True, is_label_show=True)
overlap = Overlap(width=1200, height=600)
overlap.add(bar2)
overlap.add(line2, yaxis_index=1, is_add_yaxis=True)
overlap.render()

# bar 3 表示employee职位的男女差别
attr3 = high_salary_town2
y_male3 = high_salary_m_employee
y_female3 = high_salary_f_employee
y_mean3 = mean_employee

bar3 = Bar(" gender difference in employee ")
bar3.add("male_middle_manager", attr3, y_male3, mark_line=["average"], is_label_show=True, bar_category_gap='25%',
         xaxis_rotate=20, yaxis_name='Net Salary Per Hour for Employee', yaxis_margin=2)
bar3.add("female_middle_manager", attr3, y_female3, mark_line=["average"], is_label_show=True, bar_category_gap='25%',
         xaxis_rotate=20, yaxis_name='Net Salary Per Hour for Employee', yaxis_margin=2)

# line 3 表示employee的平均工资
line3 = Line("mean salary")
line3.add("mean_middle_manager", attr3, y_mean3, is_stack=True, is_label_show=True)
overlap = Overlap(width=1200, height=600)
overlap.add(bar3)
overlap.add(line3, yaxis_index=1, is_add_yaxis=True)
overlap.render()

# bar 4 表示worker职位的男女差别
attr4 = high_salary_town3
y_male4 = high_salary_m_worker
y_female4 = high_salary_f_worker
y_mean4 = mean_worker

bar4 = Bar(" gender difference in worker ")
bar4.add("male_middle_manager", attr4, y_male4, mark_line=["average"], is_label_show=True, bar_category_gap='25%',
         xaxis_rotate=20, yaxis_name='Net Salary Per Hour for Worker', yaxis_margin=2)
bar4.add("female_middle_manager", attr4, y_female4, mark_line=["average"], is_label_show=True, bar_category_gap='25%',
         xaxis_rotate=20, yaxis_name='Net Salary Per Hour for Worker', yaxis_margin=2)

# line 4 表示worker的平均工资
line4 = Line("mean salary")
line4.add("mean_middle_manager", attr4, y_mean4, is_stack=True, is_label_show=True)
overlap = Overlap(width=1200, height=600)
overlap.add(bar4)
overlap.add(line4, yaxis_index=1, is_add_yaxis=True)
overlap.render()

# mean salary


# bar 5 表示男女在不同职位的平均工资的差别
bar5 = Bar("salaries per category and sex")
bar5.add("male", attr, mean_salary1, mark_line=["average"], is_label_show=True, xaxis_rotate=20,
         yaxis_name='Net Salary Per Hour', yaxis_margin=2, bar_category_gap="60%")
bar5.add("female", attr, mean_salary2, mark_line=["average"], is_label_show=True, bar_category_gap='25%',
         xaxis_rotate=20, yaxis_name='Net Salary Per Hour ', yaxis_margin=2)

# line 5 表示人们在不同职位的平均工资的差异
line5 = Line("mean salary")
line5.add("mean_middle_manager", attr, mean_salary, is_stack=True, is_label_show=True)
overlap = Overlap(width=1200, height=600)
overlap.add(bar5)
overlap.add(line5, yaxis_index=1, is_add_yaxis=True)
overlap.render()

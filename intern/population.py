"""

从年龄的角度对人口密度分布进行数量上的统计

"""
import pandas as pd
import seaborn as sns;

sns.set()
from pyecharts import Bar, Line, Overlap, HeatMap

populationInfo = pd.DataFrame(pd.read_csv('/Users/helena/Documents/french-employment-by-town/population.csv',
                                          low_memory=False))
populationInfo = populationInfo[populationInfo["CODGEO"].apply(lambda x: str(x).isdigit())]
populationInfo["CODGEO"] = populationInfo["CODGEO"].astype(int)

# print(populationInfo.head())
# 将population表格中的LIBGEO 改为town_name
populationInfo = populationInfo.rename(
    columns={'LIBGEO': 'Town_name', 'AGEQ80_17': 'AGE', 'SEXE': 'SEX', 'NB': 'Number'})
to_drop = ['NIVGEO', 'MOCO', 'SEX']
populationInfo.drop(columns=to_drop, inplace=True)
# print(populationInfo.head())

# 按照每个town对18～25岁人口总数量进行求和, 生成新的表格Num_1
group1 = populationInfo[(populationInfo["AGE"] >= 18) & (populationInfo["AGE"] < 26)]
Num_1 = group1["Number"].groupby(group1["Town_name"]).agg('sum').reset_index()
Num_1 = Num_1.rename(columns={"Number": "Nb age18~25"})

# 按照每个town对26～50岁人口总数量进行求和, 生成新的表格Num_2
group2 = populationInfo[(populationInfo["AGE"] >= 26) & (populationInfo["AGE"] < 50)]
Num_2 = group2["Number"].groupby(group2["Town_name"]).agg('sum').reset_index()
Num_2 = Num_2.rename(columns={"Number": "Nb age26~50"})

# 按照每个town对>50岁人口总数量进行求和, 生成新的表格Num_3
group3 = populationInfo[(populationInfo["AGE"] >= 50)]
Num_3 = group3["Number"].groupby(group3["Town_name"]).agg('sum').reset_index()
Num_3 = Num_3.rename(columns={"Number": "Nb>age50"})

# 将上面三个小表格merge成一个大表格
df1 = pd.merge(Num_1, Num_2, on="Town_name", how='left')
Age_data = pd.merge(df1, Num_3, on="Town_name", how='left')
Age_data["Total"] = Age_data["Nb age18~25"] + Age_data["Nb age26~50"] + Age_data["Nb>age50"]

# 按降序排列
Age_data = Age_data.sort_values(["Total"], ascending=False).head(10)
# top-10
# print(Age_data)
x = Age_data["Town_name"].values.tolist()
y_total = Age_data["Total"].values.tolist()
bar1 = Bar(" Top-10 Populated Towns in France ")
bar1.add("population", x, y_total, mark_line=["average"], bar_category_gap='25%',
         xaxis_rotate=20)
#
line1 = Line("Top-10 Populated Towns in France")
line1.add("value", x, y_total, is_stack=True, is_label_show=True)

overlap = Overlap(width=1200, height=600)
overlap.add(bar1)
overlap.add(line1, yaxis_index=1, is_add_yaxis=True)
overlap.render()

#          Town_name  Nb age18~25  Nb age26~50  Nb>age50    Total
# 21355        Paris       392269       641207    716155  1749631
# 17751    Marseille       107813       220239    298904   626956
# 16988         Lyon       108848       134020    137681   380549
# 29894     Toulouse       111518       123306    121151   355975
# 20331         Nice        41952        84194    136468   262614
# 20060       Nantes        62075        75771     83900   221746
# 19274  Montpellier        62980        68588     73934   205502
# 29082   Strasbourg        53761        69318     76375   199454
# 3484      Bordeaux        57254        64404     67703   189361
# 24863  Saint-Denis        36809        71619     66761   175189

# 制作heatmap图
x_axis = ["age18~25", "age26~50", "age>50"]
y_Paris = [392269, 641207, 716155]
y_Mars = [107813, 220239, 298904]
y_Lyon = [108848 ,      134020 ,   137681]
bar2 = Bar("The Population Repartition by Age")
bar2.add("Paris", x_axis, y_Paris, mark_line=["average"], bar_category_gap='35%', is_label_show=True,legend_pos='right')
bar2.add("Marseille", x_axis,y_Mars,mark_line=["average"],bar_category_gap='35%', is_label_show=True,legend_pos='right')
bar2.add("Lyon",x_axis,y_Lyon,mark_line=["average"], bar_category_gap='35%', is_label_show=True,legend_pos='right')
bar2.render()
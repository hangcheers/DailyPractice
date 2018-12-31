import pandas as pd

from pyecharts import Bar, Line, Overlap

"""
这里主要对firm进行数据分析和清洗
"""
firmInfo = pd.DataFrame(
    pd.read_csv('/Users/helena/Documents/french-employment-by-town/base_etablissement_par_tranche_effectif.csv',
                ))
firmInfo = firmInfo[firmInfo["CODGEO"].apply(lambda x: str(x).isdigit())]
firmInfo["CODGEO"] = firmInfo["CODGEO"].astype(int)
firmInfo = firmInfo.rename(columns={"LIBGEO": "Town_name"})
# 按公司的规模大小进行分类
# <100 人的为小公司
firmInfo["small"] = firmInfo["E14TS1"] + firmInfo["E14TS6"] + firmInfo["E14TS10"] + firmInfo["E14TS20"] + firmInfo[
    "E14TS50"]
# 100< employees <= 500 的为中型公司
firmInfo["medium"] = firmInfo["E14TS100"] + firmInfo["E14TS200"]
# >500 人的为大公司
firmInfo["large"] = firmInfo["E14TS500"]
# 总数
firmInfo["total"] = firmInfo["small"] + firmInfo["medium"] + firmInfo["large"]
size_firm = firmInfo.ix[:, ['CODGEO', 'Town_name', 'small', 'medium', 'large', 'total']]
size_firm = size_firm.sort_values(["total"], ascending=False).head(10)
# print(size_firm)
#       CODGEO     Town_name   small  medium  large   total
# 30784   75056        Paris  109334    1268    180  110782
# 4453    13055    Marseille   19525     230     20   19775
# 28522   69123         Lyon   16455     225     29   16709
# 12418   31555     Toulouse   12021     190     28   12239
# 2014     6088         Nice   10875      85      9   10969
# 12981   33063     Bordeaux    8208      95     13    8316
# 27951   67482   Strasbourg    7181      85     12    7278
# 17075   44109       Nantes    7075     115     14    7204
# 13633   34172  Montpellier    6925      76     12    7013
# 23381   59350        Lille    6531      96     12    6639

x = size_firm["Town_name"].values.tolist()
y_total = size_firm["total"].values.tolist()
bar1 = Bar(" Top-10 Cities with Developed Industry in France")
bar1.add("Number of company", x, y_total, mark_line=["average"], bar_category_gap='25%',
         xaxis_rotate=20, legend_pos='right')

line1 = Line("Top-10 Cities with Developed Industry in France")
line1.add("value", x, y_total, is_stack=True, is_label_show=True, legend_pos='right')

overlap = Overlap(width=1200, height=600)
overlap.add(bar1)
overlap.add(line1, yaxis_index=1, is_add_yaxis=True)
overlap.render()

# top-3城市的具体的公司分布情况
x_axis = ["small_size", "medium_size", "large_size"]
y_Paris = [109334, 1268, 180]
y_Mars = [19525, 230, 20]
y_Lyon = [16455, 225, 29]
bar2 = Bar("The Firm Distribution in Top-3 Cities")
bar2.add("Paris", x_axis, y_Paris, mark_line=["average"], bar_category_gap='35%', is_label_show=True,
         legend_pos='right')
bar2.add("Marseille", x_axis, y_Mars, mark_line=["average"], bar_category_gap='35%', is_label_show=True,
         legend_pos='right')
bar2.add("Lyon", x_axis, y_Lyon, mark_line=["average"], bar_category_gap='35%', is_label_show=True, legend_pos='right')
bar2.render()
# 重点分析巴黎，因为无论是人口还是公司的数量上巴黎都占了绝对的优势
# 分析在公司在巴黎的分布情况

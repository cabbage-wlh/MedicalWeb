from dataConnect import *
from data_view import age_view, few_category_view

# 获取年份信息
year_counts = fetch('填表年份', 'total_sql')
print(year_counts)
year_k = []
year_v = []
if year_counts is not None:
    for key, value in year_counts.items():
        # 将键添加到 keys_list
        if key and value:
            year_k.append(key)
        # 将值添加到 values_list
            year_v.append(value)
#print(year_v,year_k)
#肿瘤原发部位
position=fetch('肿瘤原发部位','total_sql')
#print(position)
csv_file_path = "D:/毕设/py-django/flaskProject/data/view.csv"
age_x, age_y = age_view()
few_data, category_1, category_2 = few_category_view()
new_data = {}
for category in category_1 + category_2:
    new_data[f'{category}'] = {}
    for k, v in zip(few_data[category].index, few_data[category].values):
        # print(category,k,v)
        if k == 1:
            k = 'SURVIVED'
        if k == 0:
            k = 'DEAD'
        if k == 'Yes':
            k = 'ACCEPT'
        if k == 'No':
            k = 'REJECT'
        new_data[f'{category}'][f'{k}'] = v
# print(new_data)
tmp = [{'name': key, 'value': value} for key, value in few_data[category_2[0]].items()]
#print(tmp)
tmp = [{'name': key, 'value': value} for key, value in few_data[category_2[1]].items()]
#print(tmp)
tmp = [{'name': key, 'value': value} for key, value in few_data[category_2[2]].items()]
#print(tmp)

#月份统计
month=fetch('填表月份','total_sql')
#print(month)
month_k = []
month_v = []
#print("[Sorting...]")
month_sorted=sorted(month.items(), key=lambda x:x[0])
#print(month_sorted)
if month is not None:
    for x in month_sorted:
        key=x[0]
        value=x[1]
        # 将键添加到 keys_list
        if key and value:
            month_k.append(str(key))
        # 将值添加到 values_list
            month_v.append(value)
#print(month_k,month_v)
new_month={}
mapping_dict={'1':'Jan','2':'Feb','3':'Mar','4':'Apr','5':'May','6':'Jun','7':'Jul','8':'Aug','9':'Sep','10':'Oct','11':'Nov','12':'Dec'}
month_k=[mapping_dict[item] for item in month_k]
#print(month_k)


class SourceDataDemo:

    def __init__(self):
        # 默认的标题
        self.title = '乳腺癌患者概况'
        # 两个小的form看板
        self.counter = {'name': '全球乳腺癌患者数量(2020)', 'value':'2300000' }
        self.counter2 = {'name': '全国乳腺癌患者数量(2020)', 'value': '420000'}
        # 总共是6个图表，数据格式用json字符串，其中第3个图表是有3个小的图表组成的
        self.echart1_data = {
            'title': 'hello',
            'data': [
                {"name": "含深深深", "value": 47},
                {"name": "教育培训", "value": 52},
                {"name": "房地产", "value": 90},
                {"name": "生活服务", "value": 84},
                {"name": "汽车销售", "value": 99},
                {"name": "旅游酒店", "value": 37},
                {"name": "五金建材", "value": 2},
            ]

        }
        self.echart2_data = {
            'title': '各年龄段病人分布情况',
            'data': [{"name": f"{item_x}-{item_x + 9}", "value": item_y}
                     for item_x, item_y in zip(age_x[:-1], age_y[:-1])]
        }
        self.echarts3_1_data = {
            'title': '种族',
            'data': [{'name': key, 'value': value} for key, value in few_data[category_2[0]].items()]

        }
        self.echarts3_2_data = {
            'title': '孕激素受体',
            'data': [{'name': key, 'value': value} for key, value in few_data[category_2[1]].items()]

        }
        self.echarts3_3_data = {
            'title': '手术类型',
            'data': [{'name': key, 'value': value} for key, value in few_data[category_2[2]].items()]

        }
        self.echart4_data = {
            'title': '每月确诊人数',

            'data': [{"name":'确诊人数/月',"value":month_v}],
            'xAxis': month_k,
        }
        self.echart5_data = {
            'title': '肿瘤原发部位',

            'data': [{"name": f"{item_x}", "value": item_y}
                     for item_x, item_y in position.items()]
        }
        # 这是一个环状图，有颜色的加上没颜色的正好等于100，半径是外圈直径和内圈直径，猜测是左闭右开

        self.echart6_data = {
            # 'data': [
            #     {"name": "", "value": [0.34,
            #                                      0.28,
            #                                      0.29,
            #                                      0.32,
            #                                      0.37,
            #                                      0.28,
            #                                      0.30]},
            # ],
            # 'xAxis': ["周一", "周二", "周三", "周四", "周五", "周六", "周日"],
            'title': '每年确诊人数',
            "data": [
                {"name": "年份统计", "value": year_v},
            ],
            'xAxis': year_k,
        }
        self.map_1_data = {
            'symbolSize': 1000,
            'data': [
                {'name': '海门', 'value': 239},
                {'name': '鄂尔多斯', 'value': 231},
            ]
        }

    @property
    def echart1(self):
        echart = {
            "data": new_data,
            "name": ["", ""],
            "title": "部分二元属性数据统计",
            'categories': category_1,
            "xAxis": [27264, 27269, 27266, 27268, 27265, 27270, 2826, 27275, 27276, 27277],
            "series": [[95.39, 109.0, 24.08, 77.03, 17.4, 30.44, 44.33, 48.48, 61.14, 79.05],
                       [102.92, 146.4, 32.43, 113.93, 24.33, 51.92, 80.87, 63.79, 78.13, 101.34]]
        }

        return echart

    @property
    def echart2(self):
        data = self.echart2_data
        echart = {
            'title': data.get('title'),
            'xAxis': [i.get("name") for i in data.get('data')],
            'series': [i.get("value") for i in data.get('data')]
        }
        return echart

    @property
    def echarts3_1(self):
        data = self.echarts3_1_data
        echart = {
            'title': data.get('title'),
            'xAxis': [i.get("name") for i in data.get('data')],
            # 'xAxis': ['W','B','Y','I'],
            'data': data.get('data'),
        }
        return echart

    @property
    def echarts3_2(self):
        data = self.echarts3_2_data
        echart = {
            'title': data.get('title'),
            'xAxis': [i.get("name") for i in data.get('data')],
            'data': data.get('data'),
        }
        return echart

    @property
    def echarts3_3(self):
        data = self.echarts3_3_data
        echart = {
            'title': data.get('title'),
            'xAxis': [i.get("name") for i in data.get('data')],
            'data': data.get('data'),
        }
        return echart

    @property
    def echart4(self):
        data = self.echart4_data
        print(data)
        echart = {
            'title': data.get('title'),
            'names': [i.get("name") for i in data.get('data')],
            'xAxis': data.get('xAxis'),
            'data': data.get('data'),
        }
        return echart

    @property
    def echart5(self):
        data = self.echart5_data
        echart = {
            'title': data.get('title'),
            'xAxis': [i.get("name") for i in data.get('data')],
            'series': [i.get("value") for i in data.get('data')],
            'data': data.get('data'),
        }
        return echart

    @property
    def echart6(self):
        data = self.echart6_data
        echart = {
            'title': data.get('title'),
            'names': [i.get("name") for i in data.get('data')],
            'xAxis': data.get('xAxis'),
            'data': data.get('data'),
        }
        return echart

    @property
    def map_1(self):
        data = self.map_1_data
        echart = {
            'symbolSize': data.get('symbolSize'),
            'data': data.get('data'),
        }
        return echart


class SourceData(SourceDataDemo):

    def __init__(self):
        super().__init__()
        self.title = 'TCGA-BRCA病人概况'

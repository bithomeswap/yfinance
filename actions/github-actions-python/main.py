# -*- coding:utf-8 -*-
#【雅虎的数据跟同花顺、老虎证券能够对应上】
# pip install yfinance
import yfinance as yf
import json
import requests
import pandas as pd
import numpy as np

def postmessage(text):
    BASEURL = 'http://wxpusher.zjiecode.com/api'
    #【查询订阅用户数量】
    pagenum=1
    payload = {
        'appToken': "AT_tFRZgjToc6XnG5dzR2MGyv1DzECNYOIU",
        'page': str(pagenum),
        'pageSize': "50",
    }
    query_user=requests.get(url=f'{BASEURL}/fun/wxuser', params=payload).json()
    # print(f"{query_user}")
    uidslist=[]
    if len(query_user["data"]["records"])>0:
        for query in query_user["data"]["records"]:
            print(query["uid"])
            uidslist.append(query["uid"])
    # print(f"{uidslist}")
    #【推送消息】
    payload = {
        'appToken': "AT_tFRZgjToc6XnG5dzR2MGyv1DzECNYOIU",
        'content': str(text),#文本消息
        'topicIds':["12417"],
        # 'uids': ["UID_qkmjMTBknX0I5ZZoVY3IBFv7WVV1"],#消息单发
        'uids':uidslist,#消息群发
    }
    requests.post(url=f'{BASEURL}/send/message', json=payload).json()

#【新浪获取标普500成分股数据】
# num设置的大一些，page设置为1，就不需要翻页了
url="http://stock.finance.sina.com.cn/usstock/api/jsonp.php//US_CategoryService.getChengfen?page=1&num=60000&sort=symobol&asc=0&market=&id=&type=2"
res=requests.get(url).text
# print(res)
res=res[res.find("["):-3]#删除无关字符
# print(res)
spList=json.loads(res)#自动转换格式了【utf8字符也有进行处理】返回json格式
df=pd.DataFrame(spList)
print(df)

#【遍历数据获取基本面数据】
for symbol in df["symbol"].tolist():
    try:
        print(symbol)
        thisticker=yf.Ticker(symbol)
        thisinfo=thisticker.info
        df.loc[df["symbol"]==symbol,"总股本"]=thisinfo["sharesOutstanding"]
        df.loc[df["symbol"]==symbol,"净利润"]=thisinfo["netIncomeToCommon"]
        df.loc[df["symbol"]==symbol,"总收入"]=thisinfo["totalRevenue"]#营收
        # df.loc[df["symbol"]==symbol,"账面价值"]=thisinfo["bookValue"]#净资产=总股本*账面价值即净资产
        df.loc[df["symbol"]==symbol,"总市值"]=thisinfo["marketCap"]
        # df.loc[df["symbol"]==symbol,"企业价值"]=thisinfo["enterpriseValue"]
        # df.loc[df["symbol"]==symbol,"总负债"]=thisinfo["totalDebt"]
        df.loc[df["symbol"]==symbol,"市净率"]=thisinfo["priceToBook"]#净资产=总市值/市净率【净资产为负数的时候无法获取数据】
        df.loc[df["symbol"]==symbol,"每股收入"]=thisinfo["revenuePerShare"]#总股本*每股收入即总营收
    except:#净资产为负数的时候会报错无法获取【相当于提前把净资产为负数的过滤掉了】
        print("报错",symbol)

# #【对基本面数据进行过滤】
# df.to_csv("基本面数据.csv")
# df=pd.read_csv("基本面数据.csv")
df["市净率"] = df["市净率"].replace("", pd.NA)  # 将空字符串替换为pd.NA
df=df.dropna(subset=["市净率"])
df["净资产"]=df["总市值"]/df["市净率"]

def reverse_signlog(X):#反标准化
    # 步骤1: (np.exp(X)-1)将转换后的数据乘以e以消除对数的影响
    temp_data = np.exp(X) - 1
    # 步骤2: np.sign(X)恢复原始数据的符号
    return np.sign(X)*(np.exp(X)-1)
def signlog(X):#数据标准化【目的是进行机器学习的时候方便】
    #np.sign((X)：返回 X 的符号。如果 X 是正数，返回1；如果 X 是负数，返回-1；如果 X 是0，返回0。
    #np.log(1.0 + abs(X))：计算 1.0 + abs(X) 的自然对数，其中 np 通常指的是 Python 中的 NumPy 库，np.log 表示自然对数。
    # factor value因子值
    return np.sign(X)*np.log(1.0 + abs(X))

df=df[["symbol","总市值","净利润","总收入","净资产","category"]]#category为行业数据
# df=df[["symbol","总市值","净利润","总收入","净资产",]]#category为行业数据
df=df.fillna(0).set_index("symbol")
print(df)

#行业因子处理
df["category"] = df["category"].replace("/","")
df["category"] = df["category"].replace("","未知")  # 将空字符串替换为pd.NA
category=df["category"].unique().tolist()
print("行业category",category)
for sector in category:#遍历所有行业【之前报错是因为有一个值为0的列】
    df.loc[df["category"]==sector,str(sector)]=1#使该行业的成分股在该行业名列的值为1
    df.loc[~(df["category"]==sector),str(sector)]=0
del df["category"]#
print("行业处理后",df)
# df.to_csv("行业处理后.csv")

baselist=["总市值",
        "净利润",
        "总收入",
        "净资产",#净资产对收益关系比较大，应该保留
        # "资产负债率", #没这个数据
        ]
for thisfactor in baselist:#直接处理因子避免空值
    print("当前因子为",thisfactor)
    # print(type(df[thisfactor][0]))
    df[thisfactor]=signlog(df[thisfactor].astype(float))


import math
from sklearn.svm import SVR
# SVR model
svr=SVR(kernel="rbf")
# training model #预测的目标是市值【这是一个估值模型】
Y=df["总市值"]
X=df.drop("总市值", axis=1)
model=svr.fit(X,Y)#根据前一天的基本面和市值情况预测
# stocks并选股
r=Y-pd.Series(svr.predict(X),Y.index,name="估值")#计算的低估程度【负数为低估】
r=r.reset_index(drop=False)
r=r.rename(columns={"symbol":"代码"})
r=pd.DataFrame(r)
# 估值数据和基本面数据拼接
df=df.reset_index(drop=False)
df=df.rename(columns={"symbol":"代码"})
r=r.merge(df[["总市值","代码"]],on="代码")
r=r.rename(columns={0:"估值"},inplace=False)
r["总市值（反标准化）"]=reverse_signlog(df["总市值"])#对标准化的市值进行反标准化
# r.to_csv("r.csv")
r=r[r["估值"]<0]#只选择低估的标的
print(type(r))#DataFrame格式
numbuystock=30#一倍池数量
dftwo=r.nsmallest(math.ceil(1.5*numbuystock),"估值")
dfone=r.nsmallest(math.ceil(numbuystock),"估值")
print(dftwo,dfone)
postmessage(
f"""
【估值数据】
低估dfone
{dfone}
低估dftwo
{dftwo}
""")

# # Apple公司的股票代码
# apple_symbol = "AAPL"
# # 创建Apple股票的Ticker对象
# apple = yf.Ticker(apple_symbol)
# print(apple)#基本面数据【最新一年的年报{跟同花顺能对上}】

# # 获取年度报告数据
# financials = apple.financials.reset_index(drop=False)
# # financials=financials.T
# financials.to_csv("financials.csv", index=False)# 将基本面数据保存为CSV文件
# # 获取季度报告数据
# quarterly_financials = apple.quarterly_financials.reset_index(drop=False)
# # quarterly_financials=quarterly_financials.T
# quarterly_financials.to_csv("quarterly_financials.csv", index=False)# 将基本面数据保存为CSV文件

# # 获取总股本和净资产等数据
# info = apple.info
# # 【数据展示】
# del info["companyOfficers"]#删除字典当中的元素
# try:
#     infodf = pd.DataFrame(info)
# except:
#     infodf = pd.DataFrame([info])
# print(infodf)
# infodf=infodf.rename(columns={
#     "sharesOutstanding":"总股本",
#     "netIncomeToCommon":"净利润",#【与同花顺一致】总净利润
#     "totalRevenue":"总收入",#【与同花顺一致】总营收
#     "bookValue":"账面价值",#净资产: {bookValue*sharesOutstanding}
#     "marketCap":"总市值",
#     "enterpriseValue":"企业价值",
#     # "totalDebt":"总负债",#【数据对不上跟同花顺】
#     "priceToBook":"市净率",
#     "revenuePerShare":"每股收入",
#     # "ebitda":"息税折旧摊销前利润",
#     # "debtToEquity":"净资产除以息税折旧摊销前利润",#【并不是kimi上的负债权益比】
# })
# infodf["净资产"]=infodf["总股本"]*infodf["账面价值"]#【与同花顺一致】
# infodf.to_csv("infodf.csv",index=False)#输出为csv文件



# 【info说明】
# address1, city, state, zip, country：这些字段通常用于描述公司的物理地址。address1 是地址的第一行，city 是城市，state 是州或省份，zip 是邮政编码，country 是国家。
# phone：公司的联系电话。
# website：公司的官方网站地址。
# industry, industryKey, industryDisp：industry 指公司所在的行业，industryKey 可能是行业的唯一标识符，industryDisp 是行业名称的显示形式。
# sector, sectorKey, sectorDisp：sector 指公司所在的行业板块，sectorKey 可能是板块的唯一标识符，sectorDisp 是板块名称的显示形式。
# longBusinessSummary：公司的详细业务概述。
# fullTimeEmployees：公司全职员工的数量。
# auditRisk, boardRisk, compensationRisk, shareHolderRightsRisk, overallRisk：这些是关于公司治理风险的不同方面，包括审计风险、董事会风险、薪酬风险、股东权利风险和总体风险。
# governanceEpochDate：公司治理评估的日期。
# compensationAsOfEpochDate：薪酬数据的截止日期。
# irWebsite：投资者关系网站的地址。
# maxAge：数据的最大年龄，即数据的新鲜度。
# priceHint：价格提示，可能是关于股票价格变动的指示。
# previousClose, open, dayLow, dayHigh：前一天的收盘价、开盘价、当日最低价和最高价。
# regularMarketPreviousClose, regularMarketOpen, regularMarketDayLow, regularMarketDayHigh：常规市场前一天的收盘价、开盘价、当日最低价和最高价。
# dividendRate, dividendYield：股息率和股息收益率。
# exDividendDate：除息日，即股票不再附带即将发放的股息的日期。
# payoutRatio：派息比率，即公司支付的股息占利润的比例。
# fiveYearAvgDividendYield：五年平均股息收益率。
# beta：贝塔系数，衡量股票相对于整个市场的波动性。
# trailingPE, forwardPE：分别指过去12个月的市盈率（Trailing P/E）和预期市盈率（Forward P/E）。
# volume, regularMarketVolume, averageVolume, averageVolume10days, averageDailyVolume10Day：成交量、常规市场成交量、平均成交量、10日平均成交量和10日平均每日成交量。
# bid, ask, bidSize, askSize：买方报价、卖方报价、买方报价数量和卖方报价数量。
# marketCap：市值，即公司股票的总市场价值。
# fiftyTwoWeekLow, fiftyTwoWeekHigh：52周最低价和最高价。
# priceToSalesTrailing12Months：过去12个月的市销率。
# fiftyDayAverage, twoHundredDayAverage：50日平均价和200日平均价。
# trailingAnnualDividendRate, trailingAnnualDividendYield：过去一年的年度股息率和股息收益率。
# currency：财务数据的货币单位。
# enterpriseValue：企业价值，即公司的总价值，包括市值加上债务、少数股东权益和优先股，减去现金和现金等价物。
# profitMargins：利润率，即公司利润占收入的比例。
# floatShares：流通股数量，即市场上可交易的股票数量。
# sharesOutstanding：在外流通股总数。
# sharesShort, sharesShortPriorMonth, sharesShortPreviousMonthDate：空头股票数量、上个月空头股票数量和上个月空头股票数量的日期。
# dateShortInterest：空头兴趣的日期。
# sharesPercentSharesOut, heldPercentInsiders, heldPercentInstitutions：空头股票占在外流通股的比例、内部人士持股比例和机构持股比例。
# shortRatio：空头比率，即空头股票数量与日均成交量的比例。
# shortPercentOfFloat：空头占流通股的百分比。
# impliedSharesOutstanding：暗示的在外流通股数量。
# bookValue：账面价值，即公司总资产减去总负债后每股的价值。
# priceToBook：市净率，即股票价格与账面价值的比率。
# lastFiscalYearEnd, nextFiscalYearEnd：上一财年结束日期和下一财年结束日期。
# mostRecentQuarter：最近一个季度。
# earningsQuarterlyGrowth, netIncomeToCommon：季度收益增长和归属于普通股股东的净利润。
# trailingEps, forwardEps：过去12个月的每股收益（Trailing EPS）和预期每股收益（Forward EPS）。
# lastSplitFactor, lastSplitDate：最后一次股票分割的因子和日期。
# enterpriseToRevenue, enterpriseToEbitda：企业价值与收入比、企业价值与EBITDA比。
# 52WeekChange, SandP52WeekChange：52周价格变动和标准普尔500指数52周价格变动。
# lastDividendValue, lastDividendDate：最后一次股息的价值和日期。
# exchange, quoteType, symbol, underlyingSymbol：交易所、报价类型、股票代码和底层资产代码。
# shortName, longName：公司简称和全称。
# firstTradeDateEpochUtc, timeZoneFullName, timeZoneShortName：首次交易日期（UTC时间）、时区全称和时区简称。
# uuid：通用唯一识别码。
# messageBoardId：信息板ID。
# gmtOffSetMilliseconds：与格林尼治标准时间的偏移量（毫秒）。
# currentPrice, targetHighPrice, targetLowPrice, targetMeanPrice, targetMedianPrice：当前价格、目标最高价、目标最低价、目标平均价和目标中位数价。
# recommendationMean, recommendationKey, numberOfAnalystOpinions：平均推荐评级、推荐评级关键字和分析师意见数量。
# totalCash, totalCashPerShare：总现金和每股现金。
# ebitda：息税折旧摊销前利润。
# totalDebt：总负债。
# quickRatio, currentRatio：速动比率和流动比率。
# totalRevenue：总收入。
# debtToEquity：负债权益比。
# revenuePerShare：每股收入。
# returnOnAssets, returnOnEquity：资产回报率和股东权益回报率。
# freeCashflow, operatingCashflow：自由现金流和经营现金流。
# earningsGrowth, revenueGrowth：收益增长和收入增长。
# grossMargins, ebitdaMargins, operatingMargins：毛利率、EBITDA利润率和营业利润率。
# financialCurrency：财务货币单位。
# trailingPegRatio：根据过去12个月的盈利和预期增长率计算的PEG比率。






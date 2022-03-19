import requests
import re

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.72 Safari/537.36'}

def GetStockData():
    url = 'http://37.push2his.eastmoney.com/api/qt/stock/kline/get?cb=jQuery331025420252272686183_1626515664856&secid=1.000001&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=1&end=20500101&lmt=2562&_=1626515664863'
    response = requests.get(url=url, headers= HEADERS)
    html = response.text
    html = eval(re.findall(r'1626515664856\((.*?)\);', html)[0])['data']['klines']
    with open('上证指数.csv', 'w',encoding='utf-8-sig') as f:
        f.write('序号,时间,开盘价,收盘价,当日最高,当日最低,当天成交量,当天成交额,振幅,涨跌幅,涨跌额,换手率\n')
        for index, data in enumerate(html):
            print(data)
            f.write(str(index +1) + ',' + data + '\n')
if __name__ == '__main__':
    GetStockData()
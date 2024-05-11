 
# 导入必要的库
import requests
import json
import time
import pandas as pd
 
 
# 函数：发起请求到京东并获取特定页面的数据
def start(page):
    # 构建京东商品评论页面的URL
    url = ('https://club.jd.com/comment/productPageComments.action?'
           '&productId=100088493573'  # 商品ID
           f'&score=0'  # 0表示所有评论，1表示好评，2表示中评，3表示差评，5表示追加评论
           '&sortType=5'  # 排序类型（通常使用5）
           f'&page={page}'  # 要获取的页面数
           '&pageSize=10'  # 每页评论数
           '&isShadowSku=0'
           '&fold=1')
 
    # 设置headers以模拟浏览器请求
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15"
    }
 
    time.sleep(2)
    # 发送GET请求获取数据
    response = requests.get(url=url, headers=headers)
    # 将返回的JSON数据解析为字典
    data = json.loads(response.text)
    return data
 
 
# 解析函数：从返回的数据中提取所需信息
def parse(data):
    items = data['comments']
    for i in items:
        yield (
            i['content']
        )
 
 
# CSV函数：将数据写入CSV文件
def csv(items, file_path='1000.csv'):
    # 如果文件不存在，创建文件并写入列名
    try:
        pd.read_csv(file_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['content'])
        df.to_csv(file_path, index=False, encoding='utf-8')
 
    # 将数据写入CSV文件，header参数用于控制是否写入列名
    df = pd.DataFrame(items, columns=['content'])
    df.to_csv(file_path, index=False, mode='a', header=False, encoding='utf-8')
 
 
# 主函数：控制整个爬取过程
def main():
    total_pages = 70  # 设置要爬取的总页数
 
    for j in range(total_pages):
        time.sleep(1.5)
        current_page = j + 1
        # 发起请求并获取数据
        data = start(current_page)
        # 解析数据
        parsed_data = parse(data)
        # 将数据写入CSV文件
        csv(parsed_data)
        print('第' + str(current_page) + '页抓取完毕')
 
 
# 如果作为独立脚本运行，则执行主函数
if __name__ == '__main__':
    main()

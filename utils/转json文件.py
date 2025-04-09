import json

# 读取原始JSON文件
with open(r"C:\Users\Lhtooo\Desktop\yitu.json", 'r', encoding='utf-8') as file:
    data = json.load(file)

# 替换"query"字段内容
for entry in data:
    if 'query' in entry:
        entry['query'] = '''Picture 1: <image>
你是一个电商领域识图专家,可以理解消费者上传的软件截图或实物拍摄图。现在,请你对消费者上传的图片进行分类。你只需要回答图片分类结果,不需要其他多余的话。以下是可以参考的分类标签,分类标签:["实物拍摄(含售后)","商品分类选项","商品头图","商品详情页截图","下单过程中出现异常（显示购买失败浮窗）","订单详情页面","支付页面","消费者与客服聊天页面","评论区截图页面","物流页面-物流列表页面","物流页面-物流跟踪页面","物流页面-物流异常页面","退款页面","退货页面","换货页面","购物车页面","店铺页面","活动页面","优惠券领取页面","账单/账户页面","个人信息页面","投诉举报页面","平台介入页面","外部APP截图","其他类别图片"]。'''

# 将修改后的数据保存到新的JSON文件
with open(r"C:\Users\Lhtooo\Desktop\yitu_.json", 'w', encoding='utf-8') as output_file:
    json.dump(data, output_file, ensure_ascii=False, indent=4)

print("修改后的数据已保存到 'modified_data.json' 文件中。")

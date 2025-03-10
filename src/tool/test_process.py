# coding:utf-8
"""
https://zhuanlan.zhihu.com/p/690049759
"""
import re

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# 示例：分析一条电影评论的情感
sia = SentimentIntensityAnalyzer()
review = "这部电影真是太精彩了！"
sentiment = sia.polarity_scores(review)
print(sentiment)

import jieba

# 示例：使用jieba进行中文分词
sentence = "自然语言处理是一门令人着迷的技术"
words = jieba.lcut(sentence)
print(words)

import re

# 示例：从HTML片段中提取电话号码
html_snippet = '<div>联系电话：12345678</div>'
pattern = r'联系电话：(\d{8})'
match = re.search(pattern, html_snippet)
print(match)
if match:
    phone_number = match.group(1)
    print(phone_number)

text = " Hello, World ! @#$%^&*( ) "
clean_text = text.strip().replace(r'[^\w\s]', '').lower()
print(clean_text)

from datetime import datetime

# 示例：转换不同格式日期为YYYY-MM-DD格式
date_strings = ['2022-03-31', 'Mar 31, 2022']
standard_dates = [datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
                  if date.count('-') == 2 else datetime.strptime(date, '%b %d, %Y').strftime('%Y-%m-%d')
                  for date in date_strings]
print(standard_dates)

search_term = 'Python'
document = 'Python是一种广泛应用的编程语言'

# 示例：检查文档中是否包含搜索项
if search_term in document:
    print(f'找到了 "{search_term}" 在文档中')
else:
    print('未在文档中找到搜索项')

text = "Hello, World!"
length = len(text)
print(length)  # 输出：13

sentence = "I love programming."
words = sentence.split(" ")
print(words)
joined_words = " ".join(words)
print(joined_words)

# 使用%操作符格式化字符串
name = "Alice"
age = 25
print("%s is %d years old." % (name, age))

# 使用str.format()方法格式化
print("{} is {} years old.".format(name, age))

# 使用f-string格式化
print(f"{name} is {age} years old.")

string = "Python Programming"
substring = string[7:14]  # 从索引7开始至索引14前结束
print(substring)  # 输出："Programming"

# 切片步长为-1，反转字符串
reversed_substring = string[::-1]
print(reversed_substring)  # 输出："gnimmargorP nohtyP"

# 将ASCII编码的bytes转为Unicode字符串
ascii_bytes = b'Hello, World!'
unicode_str = ascii_bytes.decode('utf-8')
print(unicode_str)

# 将Unicode字符串转为指定编码的bytes
encoded_bytes = unicode_str.encode('latin-1')
print(encoded_bytes)

from collections import Counter

text = "The quick brown fox jumps over the lazy dog"
word_counts = Counter(text.split())
print(word_counts.most_common())  # 输出：[(‘the’, 2), (‘quick’, 1), ...]
'''
re.match() 从字符串起始位置尝试匹配整个模式，只有完全匹配才返回结果；
re.search() 在字符串中搜索首个匹配项，即使匹配发生在字符串中间；
re.findall() 返回所有非重叠匹配项的列表；
re.finditer() 返回一个迭代器，产生所有非重叠匹配对象；
re.sub() 替换匹配项，根据给定规则对字符串进行修改；
re.compile() 编译正则表达式以提高性能，编译后的对象可以多次调用上述方法。
'''
import re

# 示例：使用re.search()查找字符串中首次出现的数字
text = "The year is 2023."
match = re.search(r'\d+', text)
if match:
    print(match.group(0))  # 输出：2023

# 编译正则表达式并多次使用
pattern = re.compile(r'\d+\.\d+')
numbers = pattern.findall("The numbers are 3.14, 2.71, and 1.62.")
print(numbers)  # 输出：['3.14', '2.71', '1.62']


def validate_email(email):
    pattern = r'^[\w\.-]+@[\w-]+\.[\w\.-]+$'
    return bool(re.match(pattern, email))


email = "example@example.com"
if validate_email(email):
    print("Valid email address")
else:
    print("Invalid email address")


def extract_urls(text):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)


text_with_urls = "Visit us at https://www.example.com or http://www.example.net"
urls = extract_urls(text_with_urls)
for url in urls:
    print(url)
mobile_pattern = r'^1[3-9]\d{9}$'
phone = "13812345678"
if re.match(mobile_pattern, phone):
    print("Valid mobile number")

# 身份证号验证（简化版，不考虑校验位有效性）
id_card_pattern = r'^\d{15}|\d{18}$'
id_num = "123456789012345678"
if re.match(id_card_pattern, id_num):
    print("Possible ID number format")

text = "apple pie bread pudding"
normal_match = re.search('apple .* pudding', text)
lazy_match = re.search("apple .*? pudding", text)
assert normal_match.group(0) == "apple pie bread pudding"
assert lazy_match.group(0) == "apple pudding"

text = "goodbye world, hello universe"
lookahead_match = re.findall(r'\w+(?=world)', text)
print(lookahead_match)  # 输出：['goodbye']

# 正向后瞻例子，假设我们只想提取包裹在双引号内的内容
quoted_text = 'say "hello" to the "world"'
quote_content = re.findall(r'(?<=").*?(?=")', quoted_text)
print(quote_content)  # 输出：['hello', 'world']


html_text = "<p>Hello <strong>World!</strong></p>"
clean_text = re.sub(r'<[^>]+>', '', html_text)
print(clean_text)  # 输出：Hello World!

# 更精细地删除CSS样式
css_styles_removed = re.sub(r'(style=".+")|(style=\'.+\')', '', html_text, flags=re.IGNORECASE)

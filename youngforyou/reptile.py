import json
import re
import requests
import datetime
from bs4 import BeautifulSoup
import os

# 获取当天的日期,并进行格式化,用于后面文件命名，格式:20200420
today = datetime.date.today().strftime('%Y%m%d')


def crawl_wiki_data():
    """
    爬取百度百科中《青春有你2》中参赛选手信息，返回html
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 Edg/128.0.0.0'

    }
    url = 'https://baike.baidu.com/item/青春有你第二季'

    try:
        response = requests.get(url, headers=headers)
        print(response.status_code)

        # 将一段文档传入BeautifulSoup的构造方法,就能得到一个文档的对象, 可以传入一段字符串
        soup = BeautifulSoup(response.text, 'lxml')
        # print(response.text)
        # 返回的是class为table-view log-set-param的<table>所有标签
        # tables = soup.find_all('table', {'class': 'table-view log-set-param'})
        tables = soup.find_all('table', {'class': 'tableBox_hIjb7'})
        print(tables[6])
        crawl_table_title = "参赛学员"
        return tables[6]
        # for table in tables:
        #     # 对当前节点前面的标签和字符串进行查找
        #     table_titles = table.find_previous('div').find_all('h3')
        #     for title in table_titles:
        #         if (crawl_table_title in title):
        #             return table
    except Exception as e:
        print(e)


def parse_wiki_data(table_html):
    '''
    从百度百科返回的html中解析得到选手信息，以当前日期作为文件名，存JSON文件,保存到work目录下
    '''
    bs = BeautifulSoup(str(table_html), 'lxml')
    all_trs = bs.find_all('tr')

    error_list = ['\'', '\"']

    stars = []

    for tr in all_trs[1:]:
        all_tds = tr.find_all('td')

        star = {}

        # 姓名
        star["name"] = all_tds[0].text
        # 个人百度百科链接
        link_element = all_tds[0].find('a')
        if(link_element):
            star["link"] = 'https://baike.baidu.com' + link_element.get('href')
        else:
            star["link"] = ""
        # 籍贯
        star["zone"] = all_tds[1].text
        # 星座
        star["constellation"] = all_tds[2].text
        # 身高
        # star["height"] = all_tds[3].text
        # 体重
        # star["weight"] = all_tds[4].text

        # 花语,去除掉花语中的单引号或双引号
        flower_word = all_tds[3].text
        for c in flower_word:
            if c in error_list:
                flower_word = flower_word.replace(c, '')
        star["flower_word"] = flower_word

        # 公司
        if not all_tds[4].find('a') is None:
            star["company"] = all_tds[4].find('a').text
        else:
            star["company"] = all_tds[4].text

        stars.append(star)

    directory = 'work'
    if not os.path.exists(directory):
        os.makedirs(directory)

            # 写入文件
    file_path = os.path.join(directory, f'{today}.json')

    json_data = json.loads(str(stars).replace("\'", "\""))
    with open(file_path, 'w', encoding='UTF-8') as f:
        json.dump(json_data, f, ensure_ascii=False)


def crawl_pic_urls():
    '''
    爬取每个选手的百度百科图片，并保存
    '''
    with open('work/' + today + '.json', 'r', encoding='UTF-8') as file:
        json_array = json.loads(file.read())

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
    }

    for star in json_array:



        name = star['name']
        link = star['link']
        # ！！！请在以下完成对每个选手图片的爬取，将所有图片url存储在一个列表pic_urls中！！！
        pic_urls = []
        if link== "":
            print("找不到链接")
            continue

        response = requests.get(link, headers=headers,timeout=15)
        soup = BeautifulSoup(response.text, 'lxml')
        images = soup.find_all('img')
        for image in images:
            # 对当前节点前面的标签和字符串进行查找
            if 'alt' in image.attrs:
                alt_text = image.attrs['alt']
                if alt_text == '百度百科':
                    continue

            if 'src' in image.attrs:
                src_text = image.attrs['src']
                if src_text.endswith("png"):
                    continue
                pic_urls.append(src_text)
            # for title in table_titles:
                # if (crawl_table_title in title):
                #     return table
        print(name )
        print(pic_urls)
        # pic_urls.append(link)
        # ！！！根据图片链接列表pic_urls, 下载所有图片，保存在以name命名的文件夹中！！！
        down_pic(name, pic_urls)


def down_pic(name, pic_urls):
    '''
    根据图片链接列表pic_urls, 下载所有图片，保存在以name命名的文件夹中,
    '''
    path = 'work/' + 'pics/' + name + '/'

    if not os.path.exists(path):
        os.makedirs(path)

    for i, pic_url in enumerate(pic_urls):
        try:
            pic = requests.get(pic_url, timeout=15)
            string = str(i + 1) + '.jpg'
            with open(path + string, 'wb') as f:
                f.write(pic.content)
                print('成功下载第%s张图片: %s' % (str(i + 1), str(pic_url)))
        except Exception as e:
            print('下载第%s张图片时失败: %s' % (str(i + 1), str(pic_url)))
            print(e)
            continue


def show_pic_path(path):
    '''
    遍历所爬取的每张图片，并打印所有图片的绝对路径
    '''
    pic_num = 0
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            pic_num += 1
            print("第%d张照片：%s" % (pic_num, os.path.join(dirpath, filename)))
    print("共爬取《青春有你2》选手的%d照片" % pic_num)


if __name__ == '__main__':
    # 爬取百度百科中《青春有你2》中参赛选手信息，返回html
    # html = crawl_wiki_data()

    # 解析html,得到选手信息，保存为json文件
    # parse_wiki_data(html)

    # 从每个选手的百度百科页面上爬取图片,并保存
    crawl_pic_urls()

    # 打印所爬取的选手图片路径
    # show_pic_path('youngforyou/work/pics/')

    print("所有信息爬取完成！")

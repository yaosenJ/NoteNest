import copy
import json
import logging
from collections import defaultdict
from itertools import groupby
from traceback import print_exc
from typing import List
import pandas as pd
import pymysql
from config import market_db_config
from gpt_core import graph_searcher
from gpt_core.chains.utils import stream_to_redis
from gpt_core.data import StockNews
from gpt_core.data.schema import NewsSchema


def finance_table_to_redis(quesiton_openid, f_question_analysis, ):
    tabel_data = graph_searcher.search_main(f_question_analysis, type='table')
    table_data = copy.deepcopy(tabel_data)
    logging.info("查询图表数据完成，开始写入redis。")
    stream_to_redis(quesiton_openid, json.dumps(table_data), 'table')
    logging.info("开始写入redis完")
    return table_data


def query_price(symbol: list, start_date: str, end_date: str) -> dict:
    """获取指定时间段内的股票价格数据。

    """
    conn = pymysql.connect(**market_db_config)

    try:

        # 个股行情查询
        query = f"""
        SELECT 
            trade_date, 
            symbol, 
            open, 
            close, 
            low, 
            high, 
            volume, 
            amount, 
            pct_chg, 
            pe, 
            pb, 
            total_mv, 
            circ_mv
        FROM market_data_day_k 
        WHERE symbol in {tuple(symbol)} 
            AND trade_date >= '{start_date}' 
            AND trade_date <= '{end_date}'
        ORDER BY trade_date DESC
        """
        query = query.replace(',)', ')')
        logging.info(f'查询行情数据，sql：{query}')
        # 执行查询
        df = pd.read_sql(query, conn)
        df = df.rename(
            columns={'pct_chg': '涨跌幅', 'high': '最高价', 'low': '最低价', 'close': '收盘价', 'open': '开盘价', 'volume': '成交量（手）',
                     'amount': '成交额（万）'})

        df = df.sort_values(by='trade_date', ascending=False)
        latest_row = df.iloc[0]
        latest_values = f"交易日: {latest_row['trade_date']}, 成交量（手）: {latest_row['成交量（手）']}, 涨跌幅: {latest_row['涨跌幅']}, PE: {latest_row['pe']}, PB: {latest_row['pb']}, 总市值(万): {latest_row['total_mv']}, 流通市值(万): {latest_row['circ_mv']}"
        df = df.drop(columns=['成交量（手）', 'total_mv', 'circ_mv'])

        return {
            'status': 'ok',
            'message': f'{latest_values} 下面是每天的open、close、low、high \n {df.to_markdown(index=False)} '
        }

    except Exception as e:
        print_exc()
        return {
            'status': 'error',
            'message': str(e)
        }
    finally:
        conn.close()


def query_finance(symbol: List[str], reportdate: List[str]):
    """获取多个报告期的详细财务数据。

    Args:
        symbol (str): 股票代码
        reportdate (List[str]): 报告期列表，每个日期格式为 YYYY-MM-DD

    Returns:
        dict: 包含财务指标的字典，包括盈利能力、财务健康和现金流指标
    """
    conn = pymysql.connect(**market_db_config)

    try:

        # 定义财务指标分类
        accounts = {
            "盈利能力分析": [
                "基本每股收益",
                "稀释每股收益",
                "销售毛利率",
                "净利润(含少数股东损益)",
                "净利润(不含少数股东损益)",
                "息税前利润",
                "息税折旧摊销前利润",
                "净资产收益率",
                "销售净利率"
            ],
            "偿债能力分析": [
                "资产负债率",
                "流动比率",
                "速动比率",
                "净债务",
                "带息债务"
            ],
            "运营效率分析": [
                "应收账款周转率",
                "存货周转率",
                "总资产周转率",
                "固定资产周转率"
            ],
            "现金流分析": [
                "经营活动产生的现金流量净额",
                "企业自由现金流量",
                "股东自由现金流量",
                "现金及现金等价物净增加额"
            ],
            "市场表现和股东价值": [
                "每股经营活动产生的现金流量净额",
                "每股净资产",
                "每股留存收益"
            ],
            "股东回报": [
                "归属于母公司(或股东)的综合收益总额",
                "股东权益合计(不含少数股东权益)",
                "每股留存收益"
            ],
            "资本结构与融资": [
                "股东权益合计",
                "长期借款",
                "短期借款",
                "应付债券"
            ]
        }

        # 展平指标列表
        all_accounts = [item for sublist in accounts.values() for item in sublist]

        latest_reportdate_sql = f'select reportdate from fin_account where symbol="{symbol[0]}" and account like "%净利润%"  order by reportdate desc limit 1'
        with conn.cursor() as c:
            c.execute(latest_reportdate_sql, )
            rows = c.fetchall()
            reportdate.append(rows[0][0].strftime('%Y-%m-%d'))
        # 构建查询
        query = f"""
        SELECT symbol, account, value, reportdate 
        FROM fin_account 
        WHERE symbol IN {tuple(symbol)}
            AND reportdate IN {tuple(reportdate)}
            AND account IN {tuple(all_accounts)}
        """
        query = query.replace(',)', ')')
        logging.info(f'查询财务报表数据，sql：{query}')

        df = pd.read_sql(query, conn)

        return {
            'status': 'ok',
            'message': df.to_markdown(index=False)
        }
    except Exception as e:
        print_exc()
        return {
            'status': 'error',
            'message': str(e)
        }
    finally:
        conn.close()


def industry_price(industries: list = None, companycodes: list = None, start_date: str = None, end_date: str = None):
    """根据industries参数查询行业行情,根据companycode参数查询对应股票所属行业行情。

    Args:
        industries (list): 行业列表 如 ['白酒', '证券']
        companycodes (list): 公司代码列表 如 ['600000.SH', '600001.SH']
        start_date (str, optional): 开始日期，格式为 YYYY-MM-DD
        end_date (str, optional): 结束日期，格式为 YYYY-MM-DD

    Returns:
        dict: 包含行业行情数据的字典,根据industries参数查询行业行情,根据companycode参数查询对应股票所属行业行情
    """

    if not industries and not companycodes:
        raise ValueError("industries和companycode不能同时为空")

    if industries:
        industry_conditions = " OR ".join([f"name LIKE '{industry}%'" for industry in industries])
        sql = f'''
            SELECT name, open, close, low, high,
               volume, amount
            FROM industry_index_market
            WHERE name IN (
                SELECT MAX(name)
                FROM industry_index_market
                WHERE {industry_conditions}
                GROUP BY LEFT(name, 2)
            )
        '''
    else:

        sql = f'''
            SELECT t1.ts_code, t1.industry_name, t2.open, t2.close, t2.low, t2.high,
               t2.volume, t2.amount
            FROM industry_component_list AS t1
            JOIN industry_index_market AS t2 ON t1.industry_code = t2.ts_code
            WHERE t1.industry_grade='L3' and t1.ts_code IN ({','.join([f"'{code}'" for code in companycodes])}); 
        '''

    # 添加日期过滤条件
    if start_date:
        sql += f" AND trade_date >= '{start_date}'"
    if end_date:
        sql += f" AND trade_date <= '{end_date}'"
    conn = pymysql.connect(**market_db_config)

    try:
        logging.info(f'查询行业指数数据，sql：{sql}')
        df = pd.read_sql(sql, conn)
        return {
            'status': 'ok',
            'message': df.to_markdown(index=False)
        }
    except Exception as e:
        print_exc()
        return {
            'status': 'error',
            'message': str(e)
        }
    finally:
        conn.close()


def query_news(symbol: list, start_date: str = None, end_date: str = None):
    try:
        news = StockNews.query.filter(StockNews.stock.in_(symbol)).order_by(
            StockNews.date.desc()).limit(10)
        if start_date:
            news = news.filter(StockNews.date >= start_date)
        if end_date:
            news = news.filter(StockNews.date <= end_date)

        news_data = NewsSchema().dump(news, many=True)
        # for it in news_data:
        news_data.sort(key=lambda x: x['stock'])  # 需要先按 'stock' 排序
        news_ret = [{'stock': companyname, "company_news": list(company_news)} for companyname, company_news in
                    groupby(news_data, lambda x: x['stock'])]
    except Exception as e:
        print_exc()
        return {
            'status': 'error',
            'message': str(e)
        }
    return {
        'status': 'ok',
        'message': news_ret
    }


def concept_price(concepts: list, start_date: str = None,
                  end_date: str = None):
    """根据concepts参数查询行业行情。

        Args:
            industries (list): 概念列表 如 ['Deepseek', '人形机器人']
            start_date (str, optional): 开始日期，格式为 YYYY-MM-DD
            end_date (str, optional): 结束日期，格式为 YYYY-MM-DD

        Returns:
            dict: 包含概念行情数据的字典,根据concepts参数查询行业行情。
        """
    concepts = tuple(concepts)
    sql = f'''
           SELECT concept_name, open, close, low, high,
              vol, amount
           FROM concept_dc_day
           WHERE concept_name in {concepts}
           
       '''

    sql = sql.replace(',)', ')')

    # 添加日期过滤条件
    if start_date:
        sql += f" AND trade_date >= '{start_date}'"
    if end_date:
        sql += f" AND trade_date <= '{end_date}'"
    conn = pymysql.connect(**market_db_config)

    try:
        logging.info(f'查询概念指数数据，sql：{sql}')
        df = pd.read_sql(sql, conn)
        return {
            'status': 'ok',
            'message': df.to_markdown(index=False)
        }
    except Exception as e:
        print_exc()
        return {
            'status': 'error',
            'message': str(e)
        }
    finally:
        conn.close()


def query_sectors_stocks(sectors: list, order_by_fileds=None, order_by='desc', limit=3, sectors_type='concept'):
    """
    成份股
    """
    sectors_map = {
        'concept': {'table': 'concept_dc_stock', 'sector_name': 't1.concept_name'},
        'industry': {'table': 'industry_component_list', 'sector_name': 't1.industry_name'}
    }
    table = sectors_map[sectors_type]['table']
    sector_name = sectors_map[sectors_type]['sector_name']
    sql = f""
    _sql = ''

    for sector in sectors:
        if _sql:
            sql += 'UNION ALL'
        _sql = f"""
            (SELECT
                    {sector_name}, t2.name
                FROM
                    {table} AS t1
                JOIN
                    std_data AS t2
                    ON t1.ts_code = t2.ts_code
                WHERE
                    {sector_name} like '{sector}'
                    AND t2.trade_date = (select trade_date from std_data order by trade_date desc limit 1)
                ORDER BY
                    std_pct_chg DESC
                LIMIT {int(limit)})  
        """
        sql += _sql
    print(sql)
    conn = pymysql.connect(**market_db_config)
    with conn.cursor() as c:
        c.execute(sql)
        rows = c.fetchall()

    ret = defaultdict(list)
    symbols = []
    if rows:
        for row in rows:
            sector_name = row[0]
            company_name = row[1]
            ret[sector_name].append(company_name)
            symbols.append(company_name)
    return ret, list(set(symbols))


# def industry_stocks(concepts: list = None, stock_market_start_date: str = None, stock_market_end_date: str = None,
#                    order_by_fileds=None, order_by='desc', limit=10):

def pct_chg_select_sectors(sector='concept'):
    """
    根据涨幅选择概念或者行业
    """

    exclude_concepts = (
        '昨日连板_含一字', '昨日连板', '昨日触板', '昨日涨停_含一字', '昨日涨停',
        '微盘股', '科创板做市股', '科创板做市商', '深证100R', '沪股通', '上证380', '中证500', '京津冀', '创业成份', '上证180_', '上证50_', '央视50_',
        '参股新三板', '深成500', '预盈预增', '预亏预减', '转债标的', '次新股', 'HS300_', 'AB股', '融资融券', '股权激励', 'B股', '股权转让', '注册制次新股'
    )
    if sector == 'concept':
        sql = f'''
            select concept_name
            from concept_dc_day 
            where trade_date=(select trade_date from concept_dc_day order by trade_date desc limit 1)  and concept_name not in {exclude_concepts}
            order by pct_chg desc 
            limit 3
        '''
    else:
        sql = '''
            select name 
            from industry_index_market 
            where trade_date=(select trade_date from industry_index_market order by trade_date desc limit 1) 
            order by pct_chg desc 
            limit 3
        '''
    conn = pymysql.connect(**market_db_config)
    with conn.cursor() as c:
        c.execute(sql)
        rows = c.fetchall()
    sectors = [row[0] for row in rows if rows]
    return sectors

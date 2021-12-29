
import pandas as pd
from multiprocessing.pool import Pool
import requests


"""
File used to get data from wto api. Keys needed to get access.
"""



"""
These wto api calls were developed by larocca89@gmail.com, I simply converted the responses to pandas dataframes at the end of each function.
"""

def get_indicators(key,
                   indicator_code='all',
                   name=None,
                   topics='all',
                   product_classification='all',
                   trade_partner='all',
                   frequency='all',
                   lang=1,
                   proxies=None):
    if name is None:
        endpoint = f'https://api.wto.org/timeseries/v1/indicators?i={indicator_code}&t={topics}' \
                   f'&pc={product_classification}&tp={trade_partner}&frq={frequency}&lang={lang}' \
                   f'&subscription-key={key}'
        response = requests.get(endpoint, proxies=proxies)
        assert response.status_code == 200, "There was an error in the request"
        returnedData = response.json()

        #data = pd.concat([pd.DataFrame.from_dict(data, orient='index', columns=[data['name']]) for data in returnedData], axis = 1)

        # ok so this has been done one one line but at what cost
        data = pd.concat([pd.pivot_table(pd.DataFrame.from_dict(data, orient='index').reset_index(), columns='index', aggfunc='first') for data in returnedData], axis = 0).reset_index(drop=True)

        return data
    else:
        endpoint = f'https://api.wto.org/timeseries/v1/indicators?i={indicator_code}&name={name}&t={topics}' \
                   f'&pc={product_classification}&tp={trade_partner}&frq={frequency}&lang={lang}' \
                   f'&subscription-key={key}'
        response = requests.get(endpoint, proxies=proxies)
        assert response.status_code == 200, "There was an error in the request"
        returnedData = response.json()

        #data = pd.concat([pd.DataFrame.from_dict(data, orient='index', columns=[data['name']]) for data in returnedData], axis = 1)

        # ok so this has been done one one line but at what cost
        data = pd.concat([pd.pivot_table(pd.DataFrame.from_dict(data, orient='index').reset_index(), columns='index', aggfunc='first') for data in returnedData], axis = 0).reset_index(drop=True)

        return data

def get_time_series_datapoints(indicator_code,  # strings only
                               key,  # strings only
                               reporting_economy='all',
                               partner_economy='default',
                               time_period='default',
                               product_sector='default',
                               product_sub_sector='false',
                               frmt='json',
                               output_mode='full',
                               decimals='default',
                               offset=0,  # number of records to skip
                               max_records=500,  # maximum number of records
                               heading_style='H',
                               language=1,  # 1 = English; 2 = French; 3 = Spanish
                               metadata='false',
                               proxies=None):

    endpoint = f"https://api.wto.org/timeseries/v1/data?i={indicator_code}&r={reporting_economy}" \
               f"&p={partner_economy}&ps={time_period}&pc={product_sector}&spc={product_sub_sector}&fmt={frmt}" \
               f"&mode={output_mode}&dec={decimals}&off={offset}&max={max_records}" \
               f"&head={heading_style}&lang={language}" \
               f"&meta={metadata}&subscription-key={key}"

    response = requests.get(endpoint, proxies=proxies)

    if response.status_code != 200:
        print("There was an error in the request: ", reporting_economy, "-", response.status_code)

    returnedData = response.json()

    # ok so this has been done one one line but at what cost to readability
    # plz write some comment reviewing what is going on here
    data = pd.concat([
            pd.pivot_table(
                pd.DataFrame.from_dict(data, orient='index').reset_index(),
                columns='index',
                aggfunc='first')
            for data in returnedData['Dataset']], axis = 0).reset_index(drop=True)

    return data

def get_reporting_economies(key,
                            name=None,
                            economy='all',
                            region='all',
                            group='all',
                            lang=1,
                            proxies=None):
    if name is None:
        endpoint = f'https://api.wto.org/timeseries/v1/reporters?ig={economy}&reg={region}&gp={group}' \
                   f'&lang={lang}&subscription-key={key}'
        response = requests.get(endpoint, proxies=proxies)
        assert response.status_code == 200, "There was an error in the request"
        returnedData = response.json()

        #data = pd.concat([pd.DataFrame.from_dict(data, orient='index', columns=[data['name']]) for data in returnedData], axis = 1)

        # ok so this has been done one one line but at what cost
        data = pd.concat([pd.pivot_table(pd.DataFrame.from_dict(data, orient='index').reset_index(), columns='index', aggfunc='first') for data in returnedData], axis = 0).reset_index(drop=True)

        return data
    else:
        endpoint = f'https://api.wto.org/timeseries/v1/reporters?name={name}&ig={economy}&reg={region}&gp={group}' \
                   f'&lang={lang}&subscription-key={key}'
        response = requests.get(endpoint, proxies=proxies)
        assert response.status_code == 200, "There was an error in the request"
        returnedData = response.json()

        #data = pd.concat([pd.DataFrame.from_dict(data, orient='index', columns=[data['name']]) for data in returnedData], axis = 1)

        # ok so this has been done one one line but at what cost
        data = pd.concat([pd.pivot_table(pd.DataFrame.from_dict(data, orient='index').reset_index(), columns='index', aggfunc='first') for data in returnedData], axis = 0).reset_index(drop=True)

        return data

def get_products(key,
                 name=None,
                 product_classification='all',
                 lang=1,
                 proxies=None):
    if name is None:
        endpoint = f'https://api.wto.org/timeseries/v1/products?pc={product_classification}&lang={lang}' \
                     f'&subscription-key={key}'
        response = requests.get(endpoint, proxies=proxies)
        assert response.status_code == 200, "There was an error in the request"
        returnedData = response.json()

        #data = pd.concat([pd.DataFrame.from_dict(data, orient='index', columns=[data['name']]) for data in returnedData], axis = 1)

        # ok so this has been done one one line but at what cost
        data = pd.concat([pd.pivot_table(pd.DataFrame.from_dict(data, orient='index').reset_index(), columns='index', aggfunc='first') for data in returnedData], axis = 0).reset_index(drop=True)

        return data
    else:
        endpoint = f'https://api.wto.org/timeseries/v1/products?name={name}&pc={product_classification}' \
                   f'&lang={lang}&subscription-key={key}'
        response = requests.get(endpoint, proxies=proxies)
        assert response.status_code == 200, "There was an error in the request"
        returnedData = response.json()

        #data = pd.concat([pd.DataFrame.from_dict(data, orient='index', columns=[data['name']]) for data in returnedData], axis = 1)

        # ok so this has been done one one line but at what cost
        data = pd.concat([pd.pivot_table(pd.DataFrame.from_dict(data, orient='index').reset_index(), columns='index', aggfunc='first') for data in returnedData], axis = 0).reset_index(drop=True)

        return data


"""
Functions to retrieve an entire dataset from wto api
"""

def setup_HS_M_0010(key):

    # http://www.wcoomd.org/en/topics/nomenclature/instrument-and-tools/hs-nomenclature-2017-edition/hs-nomenclature-2017-edition.aspx

    # only required field other than key
    indicators = get_indicators(key=key)
    reportEcon = get_reporting_economies(key=key)
    products = get_products(key=key)

    # look for datasets
    code = "HS_M"
    of_interest = indicators[indicators.code.str.contains(code)]

    # only want high level 100 top HS codes
    queryProduct = products[(products['hierarchy'].str.len()==2) & (products['productClassification']=="HS")]

    # extract productSector codes in a string seperated by commas to query with
    ps_query_string = ''

    for code in queryProduct.code:
        ps_query_string = ps_query_string + code + ','

    # this adds extra comma for last code so we remove
    ps_query_string = ps_query_string[:-1]

    # HS_M_0010

    # ITS_MTV_AX : Merchandise exports by product group – annual (Million US dollar)
    # ITS_MTV_AM : Merchandise imports by product group – annual (Million US dollar)

    # ITS_CS_QAX : Commercial services exports by main sector – preliminary annual estimates based on quarterly statistics (2005-2020) (Million US dollar
    # ITS_CS_QAM : Commercial services imports by main sector – preliminary annual estimates based on quarterly statistics (2005-2020) (Million US dollar)

    return ps_query_string, reportEcon

def setup_ITS_MTV_AX(key):

    # http://www.wcoomd.org/en/topics/nomenclature/instrument-and-tools/hs-nomenclature-2017-edition/hs-nomenclature-2017-edition.aspx

    # only required field other than key
    indicators = get_indicators(key=key)
    reportEcon = get_reporting_economies(key=key)
    products = get_products(key=key)

    # look for datasets
    code = "HS_M"
    of_interest = indicators[indicators.code.str.contains(code)]

    # sitc3 codes
    # codes = ["31", "32", "33"]
    codes = ["30"]
    queryProduct = products.query('hierarchy == @codes & productClassification == "SITC3"')
    # extract productSector codes in a string seperated by commas to query with

    ps_query_string = ''
    for code in queryProduct.code:
        ps_query_string = ps_query_string + code + ','

    # this adds extra comma for last code so we remove
    ps_query_string = ps_query_string[:-1]

    reportEcon_string = ''
    for code in reportEcon.code:
        reportEcon_string = reportEcon_string + code + ','

    # this adds extra comma for last code so we remove
    reportEcon_string = reportEcon_string[:-1]

    return ps_query_string, reportEcon_string

def query_HS_M_0010(key_str:str, years:list):

    ps_query_string, reportEcon = setup_HS_M_0010(key_str)


    logging.basicConfig(filename=f'log{years[0]}-{years[1]}.log',
                        filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s')


    listOfData = []

    #indicators = ['ITS_MTV_AX', 'ITS_MTV_AM', 'ITS_CS_QAX', 'ITS_CS_QAM']
    indicators = 'HS_M_0010'
    #indicators = 'ITS_MTV_AX'
    # indicators = 'ITS_MTV_AM'

    total = len(reportEcon)

    okList = []
    failList = []
    fail = 0
    ok = 0

    for year in range(years[0], years[1]):

        for econ in reportEcon.code:

            try:
                data = get_time_series_datapoints(
                                                    indicator_code=indicators,
                                                    reporting_economy=econ,
                                                    product_sector=ps_query_string,
                                                    time_period = year,
                                                    max_records=1000000, # max is 1 million
                                                    key=key_str)
                #listOfData.append(data)
                data.to_csv(f'{indicators}/{econ}-{year}.csv')

                print(f'ok: {econ} {year} {key_str}')
                logging.info(f'ok: {econ} {year} {key_str}')

            except:

                print(f'fail: {econ} {year} {key_str}')
                logging.info(f'fail: {econ} {year} {key_str}')

    print(f'COMPLETE -------------- {years[0]} {years[1]} {key_str}')

def query_ITS(key_str:str, years:list):

    ps_query_string, reportEconString = setup_ITS_MTV_AX(key_str)


    logging.basicConfig(filename=f'log{years[0]}-{years[1]}.log',
                        filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s')


    listOfData = []

    #indicators = ['ITS_MTV_AX', 'ITS_MTV_AM', 'ITS_CS_QAX', 'ITS_CS_QAM']
    #indicators = 'HS_M_0010'
    #indicators = 'ITS_MTV_AM'
    # indicators = 'ITS_MTV_AM'
    # indicators = 'ITS_MTV_MX'
    indicators = 'ITS_MTV_MM'

    okList = []
    failList = []
    fail = 0
    ok = 0

    for year in range(years[0], years[1]):

        try:
            data = get_time_series_datapoints(
                                                indicator_code=indicators,
                                                reporting_economy=reportEconString,
                                                product_sector=ps_query_string,
                                                time_period = year,
                                                max_records=1000000, # max is 1 million
                                                key=key_str)
            #listOfData.append(data)
            data.to_csv(f'{indicators}/{year}.csv')

            print(f'ok: {year} {key_str}')
            logging.info(f'ok: {year} {key_str}')

        except:

            print(f'fail: {year} {key_str}')
            logging.info(f'fail: {year} {key_str}')

    print(f'COMPLETE -------------- {years[0]} {years[1]} {key_str}')



keys = ['',
            '',
            '',
            '']

# years = [[1996, 2003],
#             [2003, 2010],
#             [2010, 2017],
#             [2017, 2021]]

years = [[2006, 2010],
            [2010, 2013],
            [2013, 2017],
            [2017, 2021]]

# years = [[2002, 2003],[2006, 2007], [2016, 2017]]

if __name__ == '__main__':
    with Pool(4) as p:
        p.starmap(query_ITS, [z for z in zip(keys, years)])



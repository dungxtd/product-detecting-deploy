import requests
import json
from bs4 import BeautifulSoup
# from outputs import output_json
import re
import unicodedata

# CONFIGURATION
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36",
    "x-client-data": "CIW2yQEIo7bJAQjEtskBCKmdygEIy9zKAQiWocsBCOC7zAEIpr3MAQjS68wBCIHuzAEI9fHMAQjC8swBCJ3zzAEIn/PMAQiM98wBCJf3zAE=Decoded:message ClientVariations {  repeated int32 variation_id = [3300101, 3300131, 3300164, 3313321, 3321419, 3330198, 3349984, 3350182, 3356114, 3356417, 3356917, 3356994, 3357085, 3357087, 3357580, 3357591];}",
    ":authority":"www.google.com",
    ":method": "GET",
    ":scheme":"https",
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "en-US,en;q=0.9,vi;q=0.8",
    "referer": "https://www.google.com/",
    "sec-ch-ua": '"Google Chrome";v="107", "Chromium";v="107", "Not=A?Brand";v="24"',
    "sec-ch-ua-arch": "arm",
    "sec-ch-ua-bitness": "64",
    "sec-ch-ua-full-version": "107.0.5304.110",
    "sec-ch-ua-full-version-list": '"Google Chrome";v="107.0.5304.110", "Chromium";v="107.0.5304.110", "Not=A?Brand";v="24.0.0.0"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-model": "",
    "sec-ch-ua-platform": "macOS",
    "sec-ch-ua-platform-version": "13.0.0",
    "sec-ch-ua-wow64": "?0",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": 1,
    }
# query = "dầu gội clear bạc hà 630g"

freeDelivery = False
url = 'https://www.google.com/search'
# REQUEST

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def request_search(query):
    product = []
    if(query != None):
        params = {
        "q": query["Word"],
        "sa": "X",
        "rlz": "1C1CHBF_enVN1029VN1029",
        "biw": 1512,
        "bih": 437,
        "tbm": "shop",
        "sxsrf": "ALiCzsa8gLJVCD1qxwyk3YNL2LAw99f74A:1668365587705",
        "ei": 'Ez1xY8OsKp28juMP_fiZiAk',
        "ved": "0ahUKEwiDgteR6qv7AhUdnmMGHX18BpEQ4dUDCAg",
        "uact": 5,
        "oq": query["Word"],
        "gs_lcp": "Cgtwcm9kdWN0cy1jYxADMgcIABAeELADMgkIABAIEB4QsAMyCQgAEAgQHhCwAzIJCAAQHhCwAxAYSgQIQRgBUABYAGDaBmgBcAB4AIABAIgBAJIBAJgBAMgBBLgBAcABAQ",
        "sclient": "products-cc",
    }
        response = requests.get(url, params=params)
        soup = BeautifulSoup(response.text, 'html.parser')
        # PARSING RESPONSE
        for element in soup.select('div.u30d4'):
            img_container = element.select_one('.eUQRje')
            content_container = element.select_one('.P8xhZc')
            if img_container is not None and content_container is not None:
                product_title = content_container.select_one('.rgHvZc').text
                product_link = "https://www.google.com" + \
                    content_container.select_one('.rgHvZc').select_one('a')['href']
                product_origin = ""
                product_price = ""
                product_price_text = ""
                product_reviews = ""
                product_source = img_container.select_one('img')["src"]
                for element_sub_content in content_container.select('div.dD8iuc'):
                    if element_sub_content.get("class").count("d1BlKc") > 0:
                        product_reviews = unicodedata.normalize(
                            "NFKD", element_sub_content.select_one('div.DApVsf').text).strip()
                    else:
                        for element_price in element_sub_content.contents:
                            if isinstance(element_price, str):
                                product_origin = element_price.strip()
                            elif not isinstance(element_price, str) and element_price.get("class").count("HRLxBb") > 0:
                                # product_price = int(re.sub(r"\.", '', re.sub(
                                #     r" .*?$", '', unicodedata.normalize("NFKD", element_price.text))))
                                product_price_text = unicodedata.normalize(
                                    "NFKD", element_price.text)
                try:
                    product_shipping = unicodedata.normalize(
                        "NFKD", content_container.select_one('span.dD8iuc').text).strip()
                except:
                    product_shipping = ""

                product.append({
                    'title': product_title,
                    'link': product_link,
                    # 'price': product_price,
                    "origin": product_origin,
                    'price-text': product_price_text,
                    'thumbnail': product_source,
                    'shipping': product_shipping
                })
        return (product)

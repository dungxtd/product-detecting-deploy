import requests
import json
from bs4 import BeautifulSoup
# from outputs import output_json
import re
import unicodedata

# CONFIGURATION
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
}
# query = "dầu gội clear bạc hà 630g"

freeDelivery = False
url = 'https://www.google.com/search'
# REQUEST


def request_search(query):
    params = {
        "q": query,
        "sa": "X",
        "rlz": "1C1CHBF_enVN1029VN1029",
        "biw": 1920,
        "bih": 594,
        "tbm": "shop",
        "sxsrf": "ALiCzsbyAmZEMDMgOsuuXxBnHYzF-XNYBA:1667137473581",
        "ei": 'wX9eY9TgIsTG-Qa-8ajgCA',
        "ved": "0ahUKEwiUxZ2Hi4j7AhVEY94KHb44CowQ4dUDCAY",
        "uact": 5,
        "oq": query,
        "gs_lcp": "Cgtwcm9kdWN0cy1jYxADMggIABCABBCwAzILCK4BEMoDELADECcyCwiuARDKAxCwAxAnMgsIrgEQygMQsAMQJzILCK4BEMoDELADECdKBAhBGAFQAFgAYMZAaAJwAHgAgAEAiAEAkgEAmAEAyAEFwAEB",
        "sclient": "products-cc"
    }
    response = requests.get(url, params=params)

    soup = BeautifulSoup(response.text, 'html.parser')

    # PARSING RESPONSE
    res_dict = {}
    product = []
    res_data = []

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
                            product_price = int(re.sub(r"\.", '', re.sub(
                                r" .*?$", '', unicodedata.normalize("NFKD", element_price.text))))
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
                'price': product_price,
                "origin": product_origin,
                'price-text': product_price_text,
                'source': product_source,
                'shipping': product_shipping
            })

    res_dict.update({"res_data": product})
    return (res_dict)

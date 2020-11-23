# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 00:05:13 2020

@author: anant
"""

import os
import pdb
import requests
import re
import sys
from bs4 import BeautifulSoup as bs
from re import search
import pandas as pd


BASE_PATH = "C:\\Users\\akhilg\\Documents\\CollegeDocuments\\BDMA\\CentralSuperlec\\Coursework\\DM\\Assignments\\Final Project"
NO_PROD = 4000//20 # Whatever you wish you for just divide by 20 since there 20 products per page
COUNTRY = "Germany"
PRODUCT_TAG = "product"


# URL Links for Spain
# BASE_URL = "https://es-en.openfoodfacts.org/nutrition-grade/"
# PROD_URL = "https://es-en.openfoodfacts.org/"

# Links for Germany
BASE_URL = "https://de-en.openfoodfacts.org/nutrition-grade/"
PROD_URL = "https://de-en.openfoodfacts.org/"

# Links for United Kingdom
# BASE_URL = "https://uk.openfoodfacts.org/nutrition-grade/"
# PROD_URL = "https://uk.openfoodfacts.org/"


def scrap_htmls():
    grades = ["a", "b", "c", "d", "e"]

    for grade in grades:
        target_folder = BASE_PATH + "\\data\\{}\\grade_{}".format(COUNTRY,grade)
        if not os.path.exists(target_folder):
            print("\n\n GRADE-{} Folder created successfully".format(grade.upper()))
            os.makedirs(target_folder,mode=0o777)
            os.chdir(target_folder)

        for i in range(1,NO_PROD):
            page_request = requests.get(BASE_URL + grade + "/" + str(i))
            page_text = page_request.text.encode('utf=8')
            print("Page GET requset call for Grade {0} Page {1} returned a status = {2} ".format(grade,i,page_request.status_code))

            with open(target_folder + "\\grade_{}_{}.html".format(grade,str(i)), "wb") as target:
                target.write(page_text)
            sys.stdout.flush()



def locate_products():
    prod_links = []

    grades = ["a", "b", "c", "d", "e"]

    for grade in grades:
        target_folder = BASE_PATH + "\\data\\{}\\grade_{}".format(COUNTRY,grade)
        os.chdir(target_folder)

        for i in range(1,NO_PROD):
            # file_html = open(BASE_PATH + "\\data\\grade_{0}.html".format(grade),"rb")
            with open(target_folder + "\\grade_{}_{}.html"\
                      .format(grade,str(i)), "rb") as file_html:

                plain_text =  file_html.read()
                soup = bs(plain_text,"lxml")
                # pdb.set_trace()
                all_links = soup.find_all("a",
                          attrs={"href":re.compile("^/{}/".format(PRODUCT_TAG))})
                for link in all_links:
                    prod_links.append(link["href"])
                print("For grade = {} and round = {} the len(prod_links) = {}".format(grade, i, len(prod_links)))

    return prod_links


def get_macros(link_products):

    table_id = "nutrition_data_table"
    column_names = []

    data = []
    for i in link_products:
        prod_tup = {}
        page = requests.get(PROD_URL+ "/" +i, allow_redirects=True)

        if not page.status_code == 200:
            print("Some error occurred loading the page. Status code: " + str(page.status_code))
        else:
            print("\nThe scrapper for product information has been invoked.\n")
            soup = bs(page.text, "lxml")

            # Locating the table, header and body of the target table
            table = soup.find("table",{"id":table_id})
            headers = soup.find("thead").findAll("th")
            body = table.find("tbody").findAll("tr")


            product_name = soup.find("h1", {"property":"food:name"})
            # print("-"*40)
            # print("Product Name is = ", product_name.text)
            # extracting columns for our data. Using column named '#' as index
            prod_tup["product"] = product_name.text

            # import pdb;pdb.set_trace()

            # extracting all the information across the tuples of the table
            for row in body:
                try:
                    tup = row.find_all('td')
                    criteria = re.sub(r'[^A-Za-z]', '', tup[0].text.strip()).strip().title()
                    if criteria not in column_names:
                        column_names.append(criteria)
                    # print("crtiteria is = ", criteria)
                    prod_tup[criteria] = tup[1].text.strip()
                    # prod_tup[criteria] = int(re.sub(r'[kcal*]|[,]|[g]','', tup[1].text.strip()).strip())
                except Exception as e:
                    print("Exception while building the block is = ", e)
            # print("Columns names tuple is = ", column_names)
            # print("Product tuple is = ", prod_tup)
            data.append(prod_tup)

    df = pd.DataFrame(data=data)
    # print(df)
    df.to_csv(BASE_PATH +"\\data\\product_{}.csv".format(COUNTRY))
    return(df)

if __name__ == "__main__" :
    #scrap_htmls()
    link_products = locate_products()
    # link_products = ["product/2500300000017/yogur-de-cabra-natural-medianito",
                     # "product/20076740/bio-garbanzo-organic-perdrosillano"]
    get_macros(link_products)

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


BASE_PATH = "C:\\Users\\akhilg\\Documents\\CollegeDocuments\\BDMA\\CentralSuperlec\\Coursework\\DM\\Assignments\\Final Project"
BASE_URL = "https://es-en.openfoodfacts.org/nutrition-grade/"
PROD_URL = "https://es-en.openfoodfacts.org/"



def scrap_htmls():
    grades = ["a", "b", "c", "d", "e"]    
    
    for grade in grades:
        target_folder = BASE_PATH + "\\data\\grade_{}".format(grade)
        if not os.path.exists(target_folder):
            print("\n\n GRADE-{} Folder created successfully".format(grade.upper()))
            os.makedirs(target_folder,mode=0o777)
            os.chdir(target_folder)
            
        for i in range(1,21):
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
        target_folder = BASE_PATH + "\\data\\grade_{}".format(grade)
        
        for i in range(1,21):
            # file_html = open(BASE_PATH + "\\data\\grade_{0}.html".format(grade),"rb")
            with open(target_folder + "\\grade_{}_{}.html"\
                      .format(grade,str(i)), "rb") as file_html:
    
                plain_text =  file_html.read()
                soup = bs(plain_text,"lxml")
                
                all_links = soup.find_all('a', attrs={'href':re.compile("^/product/")})
                print(len(all_links))
                assert False
                # pdb.set_trace()
                # for link in all_links:
                #     #print("type of link is = ", type(link['href']))
                #     if search("product", link['href']):
                #         print("\nLINkkkk",link.attrs['href'])
                #         prod_links.append(link.get('href'))
            
            
    # link_texts = [a if a.contains("products) for a in all_links]
            
    # print("\nLink is  = ", all_hrefs[:-5])
    # print(len(all_hrefs))

if __name__ == "__main__" :
    # scrap_htmls()
    locate_products()
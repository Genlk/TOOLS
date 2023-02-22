import os
import time

# 打开浏览器驱动
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from download_url import extract_pics,doload_all_urls

import shutil
import time

import requests
import os
import cv2
from tqdm import tqdm

def extract_pics(mov_dir = ""):
    save_path = "D:/temp/query_images/"

    for mov in tqdm(os.listdir(mov_dir)):
        if "测试" in mov:
            continue
        if int(mov[:8]) <=20230217:
            continue
        if mov.split(".")[-1] not in ["mp4","MOV"]:
            continue
        if os.path.exists(os.path.join(save_path,mov.rstrip(".mp4").rstrip(".MOV")+ "_0.jpg")):
            print("\n Already exist file %s"%mov)
            continue
        cap = cv2.VideoCapture(os.path.join(mov_dir, mov).replace("\\","/"))
        time.sleep(5)
        if cap.isOpened():
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cnt = 0
            while True:
                ret_val, frame = cap.read()
                if not ret_val:
                    break
                try:
                    if cnt%(5*fps)==0:
                        cv2.imwrite(os.path.join(save_path,mov.rstrip(".mp4").rstrip(".MOV")+ "_" +str(cnt)+".jpg"), frame)
                except Exception as e:
                    print(e)
                cnt +=1
        cap.release()


def check_if_already_scraber(txt_name, txt_dir = "D:/temp/query_txt/",pic_save_dir = "",img_dir = "D:/temp/scraber/"):
    if not os.path.exists(os.path.join(pic_save_dir,txt_name.replace(".txt",".jpg"))):
        return False
    if not os.path.isdir(os.path.join(pic_save_dir,txt_name.rstrip(".txt"))):
        return False
    # with open(os.path.join(txt_dir,txt_name),"r",encoding="utf-8") as f:
    #     lines = f.readlines()
    # if not len(lines) == len(os.listdir(os.path.join(pic_save_dir,txt_name.rstrip(".txt")))):
    #     print("\nWarning : %s: scraber num is %d, txt nums is %d"%(txt_name, len(os.listdir(os.path.join(pic_save_dir,txt_name.rstrip(".txt")))),len(os.listdir(os.path.join(pic_save_dir,txt_name.rstrip(".txt"))))))
    return True

def doload_all_urls(txt_dir = "D:/temp/query_txt/",pic_save_dir = ""):
    txts = os.listdir(txt_dir)
    for txt_name in tqdm(txts):
        if check_if_already_scraber(txt_name):
            continue
        with open(os.path.join(txt_dir,txt_name),"r",encoding="utf-8") as f:
            lines = f.readlines()
        shutil.copy(
            os.path.join("D:/temp/query_images", txt_name.replace(".txt",".jpg")),
            os.path.join(pic_save_dir,txt_name.replace(".txt",".jpg"))
        )
        for index,url in enumerate(lines):
            url = url.rstrip("\n")
            response = requests.get(url)
            temp_pic_save_dir = os.path.join(pic_save_dir,txt_name.rstrip(".txt"))
            if not os.path.exists(os.path.join(temp_pic_save_dir)):
                os.mkdir(temp_pic_save_dir)
            pic_name = os.path.join(temp_pic_save_dir,"img_" +  str(index)+".jpeg")
            if response.status_code == 200:
                with open(pic_name, 'wb') as f:
                    f.write(response.content)
def doload_urls():
    with open("D:/temp/scraber/cathy.txt","r",encoding="utf-8") as f:
        lines = f.readlines()
    for index,url in enumerate(lines):
        url = url.rstrip("\n")
        response = requests.get(url)
        pic_name = "D:/temp/scraber/img_" +  str(index)+".jpeg"
        if response.status_code == 200:
            with open(pic_name, 'wb') as f:
                f.write(response.content)


def spyder():
    options = webdriver.ChromeOptions()
    image_query_dir = "D:/temp/query_images/"
    for index, img_name in enumerate(os.listdir(image_query_dir)[6:]):
        print("Temp process index is %d, image name is %s"%(index, img_name))
        img_path = image_query_dir +  img_name
        save_txt_dir = "D:/temp/query_txt/" + img_name.replace("jpg","txt")
        if os.path.exists(save_txt_dir):
            continue
        browser = webdriver.Chrome(options=options)
        browser.get("https://image.baidu.com")
        browser.implicitly_wait(10)
        browser.maximize_window()

        #找到按钮
        upload_button = browser.find_element(by=By.ID, value="sttb")
        upload_button.click()
        browser.implicitly_wait(1)
        choose_button = browser.find_element(value="uploadImg")
        upload_img_button = browser.find_element(value="stfile")
        upload_img_button.send_keys(img_path)
        browser.implicitly_wait(10)
        for i in range(5):
            browser.execute_script("window.scrollBy(0,50000)")
            time.sleep(10)
        time.sleep(5)
        imgs = browser.find_elements(by=By.TAG_NAME, value="img")
        for img in imgs:
            print(img.get_attribute("src"))
        with open(save_txt_dir,"w",encoding="utf-8") as f:
            for img in imgs:
                try:
                    f.writelines(img.get_attribute("src")+"\n")
                except Exception as e:
                    print(e)
                    continue
        browser.quit()


extract_pics()
spyder()
doload_all_urls()

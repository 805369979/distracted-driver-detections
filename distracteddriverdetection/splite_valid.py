import os
import pandas as pd
import numpy as np
import shutil

dir = "./img/"

driver_imgs_list_csv = os.path.join(dir, "driver_imgs_list.csv")
valid_subjects = ['p052','p056','p061','p064','p066','p072','p075','p081']

if not os.path.exists(dir + "valid"):
    os.mkdir(dir + "valid")
    for i in range(10):
        print("sdfdsfdsfdsfds")
        os.mkdir(dir + "valid/c%d"%i)

df = pd.read_csv(driver_imgs_list_csv)

for valid_subject in valid_subjects:
    df_valid = df[(df["subject"]==valid_subject)]
    for index, row in df_valid.iterrows():
        subpath = row["classname"] + "/" + row["img"]
        print(os.path.join(dir,"train",subpath))
        print(os.path.join(dir,"valid",subpath))
        print("========================================")
        # if os.path.exists(os.path.join(dir,"train",subpath)):
        #     shutil.move(os.path.join(dir,"train",subpath), os.path.join(dir,"valid",subpath),)
        # else:
        #     print("cannot move {} : {}".format(row["subject"],subpath))


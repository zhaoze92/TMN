# coding: utf-8

def genSample():
    labels = ["news", "zhidao", "baike", "website", "image", "video", "jingyan", "weather", "book", "music", "tieba", "app", "dict", "caipiao", "weibo", "express", "transport", "fanyi", "navigation", "map", "product", "stock"]
    path = "/home/zhaoze/mi-files/data/new_multi_label/rawData_new_22cluster/new/newSample/mlp.txt"
    pathSave = "/home/zhaoze/my-projects/34.TMN/TMN/data/multiCluster/mlp_data.txt"
    fw = open(pathSave, "w")
    fw.truncate()

    fr = open(path)
    line = fr.readline()
    count = 0
    while line:
        count += 1
        if(count%10000 == 0):
            print("done:"+str(count))
        msg = line.strip().split("\t")
        if(len(msg) != 24):
            continue
        label = 0
        countMax = 0
        for i in range(1, 23):
            if(int(msg[i]) > countMax):
                countMax = int(msg[i])
                label = i
        fw.write(msg[23]+"\t"+labels[label-1]+"\n")
        line = fr.readline()
    fw.close()
    print("done")

genSample()


import os
dataset_path = "/app/dataset"
class_name = []
class_path = []
for d in os.listdir(dataset_path):
            full_d_path = os.path.join(dataset_path, d)
            if os.path.isdir(full_d_path):
                class_name.append(d)
                class_path.append(full_d_path)

from collections import defaultdict
M = len(class_name)
num_file = defaultdict(int)
for path, name in zip(class_path, class_name):
    cnt = 0
    for f in os.listdir(path):
        if f.endswith(".wav"):
            cnt +=1
    num_file[name] = cnt

#答えとなる単語の数
print(M)
#各単語の音声ファイルの数
print(num_file)
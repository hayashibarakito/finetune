import numpy as np
from pathlib import Path

dir = "/home/fine/ex1/FT/FT/memory_bank(b7)/target(3)/"

q_names = []
querys = []
#クエリ画像のロード
for feature_path2 in Path( dir + "query/" ).glob("*.npy"):
    #print(feature_path2.stem)
    querys.append(np.load(feature_path2))
    q_names.append(feature_path2.stem)
querys = np.array(querys)
print(querys.shape)
#print(querys[0].shape)

#検索候補のロード
features = []
img_paths = []
for feature_path in Path( dir + "search/" ).glob("*.npy"):
    #print(feature_path.stem)
    features.append(np.load(feature_path))
    img_paths.append(feature_path.stem + ".jpg")
features = np.array(features)
#print(img_paths)

lists = [0, 10, 20, 30, 40, 50, 60 ,70 ,80, 90]
# Run search
for j in range(10, 20):
    query = querys[j]
    print(q_names[j])
    dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
    ids = np.argsort(dists)[:1]  # Top n results
    for i in ids:
        print(img_paths[i])
    print()

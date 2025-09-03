import os, glob, random, pandas as pd

RAW_GLOB  = r"D:\Data_Science_project-data\acouslic-ai-train-set\images\stacked-fetal-ultrasound\*.mha"
RAND_SEED = 42   # 必须与预处理脚本相同
OUT_CSV   = r"D:\TransUnet\case_to_uuid.csv"

names = glob.glob(RAW_GLOB)
random.seed(RAND_SEED)
random.shuffle(names)

rows = []
for order, p in enumerate(names):
    uuid = os.path.splitext(os.path.basename(p))[0]  # 文件名就是 uuid
    rows.append({"case_id": order, "uuid": uuid, "mha_path": p})

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"[Saved] {OUT_CSV}, total={len(df)}, missing_uuid={(df['uuid']=='' ).sum()}")

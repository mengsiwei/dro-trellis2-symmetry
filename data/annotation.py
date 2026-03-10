import objaverse
import pandas as pd

# 加载 80 万个物体的基础元数据
uids = objaverse.load_uids()
# 获取部分物体的详细标注（包含名称、标签等）
annotations = objaverse.load_annotations(uids[:20]) # 示例前1000个

# 转化为 DataFrame 方便查看
df = pd.DataFrame.from_dict(annotations, orient='index')
print(df[['name', 'tags']].head())
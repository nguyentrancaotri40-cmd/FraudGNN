# scripts/utils/build_graph.py
from src.utils.config import load_config
from src.data.load_data import load_dataset
from src.data.split import split_dataframe
from src.data.preprocess import FraudPreprocessor
from src.graph.build_graph import build_transaction_graph, save_graph

cfg = load_config('configs/paysim.yaml')
df = load_dataset(cfg)
train_df, val_df, test_df = split_dataframe(df, cfg)
pre = FraudPreprocessor(cfg)
x_train, y_train, t_train = pre.fit_transform(train_df)
graph = build_transaction_graph(x_train, y_train, t_train, cfg)
save_graph(graph, 'data/graphs/train_graph.pkl')
print(graph)
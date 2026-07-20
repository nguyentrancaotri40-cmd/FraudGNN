# scripts/utils/preprocess.py
import argparse
from src.utils.config import load_config
from src.data.load_data import load_dataset
from src.data.split import split_dataframe
from src.data.preprocess import FraudPreprocessor

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/paysim.yaml')
args = parser.parse_args()

cfg = load_config(args.config)
df = load_dataset(cfg)
train_df, val_df, test_df = split_dataframe(df, cfg)
pre = FraudPreprocessor(cfg)
x_train, y_train, t_train = pre.fit_transform(train_df)
x_val, y_val, t_val = pre.transform(val_df)
x_test, y_test, t_test = pre.transform(test_df)
pre.save('data/processed/preprocessor.joblib')

print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
print(f"Train fraud ratio: {y_train.mean():.4f}")
print(f"Val fraud ratio: {y_val.mean():.4f}")
print(f"Test fraud ratio: {y_test.mean():.4f}")
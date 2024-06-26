import argparse

parser = argparse.ArgumentParser(description='Arguments for Human Behavior Predition')

# args for device
parser.add_argument('--device', type=str, default='cuda', help='Device name')
# args for seed
parser.add_argument('--seed', type=int, default=233, help='Random seed')

# args for training
parser.add_argument('--lr', type=float, default=0.0004, help='Learning rate')
parser.add_argument('--epoch', type=int, default=2000, help='Epoch')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--train_radio', type=float, default=0.7, help='Train radio')
parser.add_argument('--vali_radio', type=float, default=0.1, help='Validation radio')
parser.add_argument('--test_radio', type=float, default=0.2, help='Test radio')

# args for model
parser.add_argument('--model_name', type=str, default='GRU', help='Model Name: MLP, CNN, GRU, LSTM, BiLSTM, Transformer, ST_Transformer')
parser.add_argument('--input_dim', type=int, default=75, help='Input dimension')
parser.add_argument('--output_dim', type=int, default=75, help='Output dimension')
parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
parser.add_argument('--hidden_dim_fc', type=int, default=128, help='Hidden dimension')
parser.add_argument('--num_layers', type=int, default=4, help='Number of layers, default=4')

parser.add_argument('--hidden_dim_trans', type=int, default=128, help='Hidden dimension, default=128')
parser.add_argument('--num_heads', type=int, default=8, help='Number of heads, default=8')




configs = parser.parse_args()

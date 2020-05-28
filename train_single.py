from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.arguments import get_args
from src.trainers.GATtrainer import _train_gat_trans
from src.trainers.TransformerTrainer import _train_transformer

if __name__ == "__main__":
  args = get_args()
  global step

  if args.enc_type == 'transformer' and args.dec_type == "transformer":
    _train_transformer(args)

  elif ((args.enc_type == "gat") and (args.dec_type == "transformer")):
    _train_gat_trans(args)

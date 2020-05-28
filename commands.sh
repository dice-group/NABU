python preprocess.py \
  --train_src 'data/processed_data/eng/train_src' \
  --train_tgt 'data/processed_data/eng/train_tgt' \
  --eval_src 'data/processed_data/eng/eval_src' \
  --eval_tgt 'data/processed_data/eng/eval_tgt' \
  --test_src 'data/processed_data/eng/test_src' \
  --spl_sym 'data/processed_data/special_symbols' \
  --model gat --lang eng --sentencepiece True \
  --vocab_size 16000 --sentencepiece_model 'bpe'

python train_single.py \
  --train_path 'data/processed_graphs/eng/gat/train' \
  --eval_path 'data/processed_graphs/eng/gat/eval' \
  --test_path 'data/processed_graphs/eng/gat/test' \
  --src_vocab 'vocabs/gat/eng/src_vocab' \
  --tgt_vocab 'vocabs/gat/eng/train_vocab.model' \
  --batch_size 32 --enc_type gat --dec_type transformer --model gat --vocab_size 16000 \
  --emb_dim 256 --hidden_size 256 --filter_size 256 --beam_size 5 \
  --beam_alpha 0.1 --enc_layers 6 --dec_layers 6 --num_heads 4 --sentencepiece True \
  --steps 10000 --eval_steps 1000 --checkpoint 1000 --alpha 0.2 --dropout 0.2 \
  --reg_scale 0.0 --decay True --decay_steps 5000 --lang eng --debug_mode False \
  --eval 'data/processed_data/eng/eval_src' --eval_ref 'data/processed_data/eng/eval_tgt'

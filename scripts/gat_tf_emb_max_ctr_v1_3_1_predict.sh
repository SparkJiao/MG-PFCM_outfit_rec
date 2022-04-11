#for tuple_len in 4 7; do
for tuple_len in 4 5 6 7 12; do
  python predict.py dataset.max_tuple_num=$tuple_len -cp conf -cn gat_tf_emb_max_ctr_v1_3_1
done;
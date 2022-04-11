#for tuple_len in 5 6 8 10; do
#  python predict.py dataset.max_tuple_num=$tuple_len -cp conf/gat_pooling -cn gat_pooling_emb_max_ctr_v1_3_1
#done;

#for tuple_len in 4 7; do
#  python predict.py dataset.max_tuple_num=$tuple_len -cp conf/gat_pooling -cn gat_pooling_emb_max_ctr_v1_3_1
#done;

for tuple_len in 4 5 6 7 12; do
  python predict.py dataset.max_tuple_num=$tuple_len -cp conf/gat_pooling -cn gat_pooling_emb_max_ctr_v1_3_1
done;
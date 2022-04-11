

#for tuple_len in 5 6 8 10; do
#  python predict.py dataset.max_tuple_num=$tuple_len model.item_use_img=True model.item_use_text=False gpu=A100 -cp conf/item_ab -cn gat_tf_emb_max_ctr_v1_3_1
#done;
#
#for tuple_len in 5 6 8 10; do
#  python predict.py dataset.max_tuple_num=$tuple_len model.item_use_img=False model.item_use_text=True gpu=T4 -cp conf/item_ab -cn gat_tf_emb_max_ctr_v1_3_1
#done;


#for tuple_len in 4 7; do
for tuple_len in 4 5 6 7 12; do
  python predict.py dataset.max_tuple_num=$tuple_len model.item_use_img=True model.item_use_text=False gpu=A100 -cp conf/item_ab -cn gat_tf_emb_max_ctr_v1_3_1
done;

#for tuple_len in 4 7; do
for tuple_len in 4 5 6 7 12; do
  python predict.py dataset.max_tuple_num=$tuple_len model.item_use_img=False model.item_use_text=True gpu=T4 -cp conf/item_ab -cn gat_tf_emb_max_ctr_v1_3_1
done;

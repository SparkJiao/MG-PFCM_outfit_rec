

#for gnn_layer in 3 4 5; do
#  python trainer.py model.gnn.num_layers=${gnn_layer} \
#    per_gpu_train_batch_size=8 per_gpu_eval_batch_size=8 gradient_accumulation_steps=3 \
#    output_dir=experiments/gat_tf_fix_emb_max.v3.1.wd0.1.n5.A100.g${gnn_layer} -cn gat_tf_emb_max_v1_3_1
#done;

#for tf_layer in 2 3 4; do
#  python trainer.py model.transformer.encoder_layers=${tf_layer} \
#    per_gpu_train_batch_size=8 per_gpu_eval_batch_size=8 gradient_accumulation_steps=3 \
#    output_dir=experiments/gat_tf_fix_emb_max.v3.1.wd0.1.n5.A100.tf${tf_layer} -cn gat_tf_emb_max_v1_3_1
#done;



#python trainer.py model.gnn.num_layers=6 per_gpu_train_batch_size=8 per_gpu_eval_batch_size=8 gradient_accumulation_steps=3 output_dir=experiments/gat_tf_fix_emb_max.v3.1.wd0.1.n5.A100.g6 -cn gat_tf_emb_max_v1_3_1


#python trainer.py model.transformer.encoder_layers=5 per_gpu_train_batch_size=8 per_gpu_eval_batch_size=8 gradient_accumulation_steps=3 output_dir=experiments/gat_tf_fix_emb_max.v3.1.wd0.1.n5.A100.tf5 -cn gat_tf_emb_max_v1_3_1


#for gnn in 3 4; do
#    python trainer.py model.gnn.num_layers=${gnn} \
#    output_dir=experiments/gat_tf_fix_emb_max.ctr.v3.1.wd0.1.n5.2080Ti.g${gnn} -cn gat_tf_emb_max_ctr_v1_3_1
#done;


#for gnn in 5 6; do
#    python trainer.py model.gnn.num_layers=${gnn} \
#    output_dir=experiments/gat_tf_fix_emb_max.ctr.v3.1.wd0.1.n5.2080Ti.g${gnn} -cn gat_tf_emb_max_ctr_v1_3_1
#done;

#for tf_layer in 2 3; do
for tf_layer in 4 5; do
  python trainer.py model.transformer.encoder_layers=${tf_layer} \
    per_gpu_train_batch_size=8 per_gpu_eval_batch_size=8 gradient_accumulation_steps=3 \
    output_dir=experiments/gat_tf_fix_emb_max.ctr.v3.1.wd0.1.n5.A100.tf${tf_layer} -cn gat_tf_emb_max_ctr_v1_3_1
done;

for step in 2000 1000 3000; do
  python trainer.py do_train=False do_eval=True eval_sub_path=checkpoint-${step} eval_num_workers=32 dataset.graph_sampler.max_neighbour_num=6 -cp conf/gp_bpr -cn gat_tf_emb_max_v1
done;
max_neighbour_num=5
#max_neighbour_num=3
seed=42
num_workers=32

#python preprocess/sparsing_subgraph_v1.py --path IQON_pair_remove_edge/subgraphs_v1.0.ii \
#  --output_file IQON_pair_remove_edge/subgraph.ii.v1.0.sparse --max_neighbour_num $max_neighbour_num --seed $seed --num_workers $num_workers
#
#python preprocess/sparsing_subgraph_v1.py --path IQON_pair_remove_edge/subgraphs_v1.0.iia \
#  --output_file IQON_pair_remove_edge/subgraph.iia.v1.0.sparse --max_neighbour_num $max_neighbour_num --seed $seed --num_workers $num_workers
#
#python preprocess/sparsing_subgraph_v1.py --path IQON_pair_remove_edge/subgraphs_v1.0.iui \
#  --output_file IQON_pair_remove_edge/subgraph.iui.v1.0.sparse --max_neighbour_num $max_neighbour_num --seed $seed --num_workers $num_workers
#
#python preprocess/sparsing_subgraph_v1.py --path IQON_pair_remove_edge/subgraphs_v1.0.uia \
#  --output_file IQON_pair_remove_edge/subgraph.uia.v1.0.sparse --max_neighbour_num $max_neighbour_num --seed $seed --num_workers $num_workers
#
#python preprocess/sparsing_subgraph_v1.py --path IQON_pair_remove_edge/subgraphs_v1.0.uiu \
#  --output_file IQON_pair_remove_edge/subgraph.uiu.v1.0.sparse --max_neighbour_num $max_neighbour_num --seed $seed --num_workers $num_workers

#python preprocess/sparsing_subgraph_v1.py --path "IQON_pair_remove_edge/subgraphs/subgraph-iai/*" \
#  --output_file IQON_pair_remove_edge/subgraph.iai.v1.0.sparse --max_neighbour_num $max_neighbour_num --seed $seed --num_workers $num_workers
#
#python preprocess/sparsing_subgraph_v1.py --path "IQON_pair_remove_edge/subgraphs/subgraph-uiaiu/*" \
#  --output_file IQON_pair_remove_edge/subgraph.uiaiu.v1.0.sparse --max_neighbour_num $max_neighbour_num --seed $seed --num_workers $num_workers

# ==========================

python preprocess/sparsing_subgraph_v1.py --path gp-bpr/subgraph.ii \
  --output_file gp-bpr/subgraph.ii.v1.0.sparse --max_neighbour_num $max_neighbour_num --seed $seed --num_workers $num_workers

python preprocess/sparsing_subgraph_v1.py --path gp-bpr/subgraph.iia \
  --output_file gp-bpr/subgraph.iia.v1.0.sparse --max_neighbour_num $max_neighbour_num --seed $seed --num_workers $num_workers

python preprocess/sparsing_subgraph_v1.py --path gp-bpr/subgraph.iui \
  --output_file gp-bpr/subgraph.iui.v1.0.sparse --max_neighbour_num $max_neighbour_num --seed $seed --num_workers $num_workers

python preprocess/sparsing_subgraph_v1.py --path gp-bpr/subgraph.uia \
  --output_file gp-bpr/subgraph.uia.v1.0.sparse --max_neighbour_num $max_neighbour_num --seed $seed --num_workers $num_workers

python preprocess/sparsing_subgraph_v1.py --path gp-bpr/subgraph.uiu \
  --output_file gp-bpr/subgraph.uiu.v1.0.sparse --max_neighbour_num $max_neighbour_num --seed $seed --num_workers $num_workers

python preprocess/sparsing_subgraph_v1.py --path "gp-bpr/subgraph-iai/*" \
  --output_file gp-bpr/subgraph.iai.v1.0.sparse --max_neighbour_num $max_neighbour_num --seed $seed --num_workers $num_workers

python preprocess/sparsing_subgraph_v1.py --path "gp-bpr/subgraph-uiaiu/*" \
  --output_file gp-bpr/subgraph.uiaiu.v1.0.sparse --max_neighbour_num $max_neighbour_num --seed $seed --num_workers $num_workers

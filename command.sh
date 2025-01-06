

cd /data/sjb_workspace/rl_generalization/dmcontrol-generalization-benchmark-main
conda activate svea_new
export DISPLAY=:1

CUDA_VISIBLE_DEVICES=5 python src/my_train_all.py --domain_name ball_in_cup --task_name catch --algorithm sgsac79 --seed 1

CUDA_VISIBLE_DEVICES=0 python src/my_train_all.py --domain_name walker --task_name walk --algorithm sgsac79 --seed 2
CUDA_VISIBLE_DEVICES=1 python src/my_train_all.py --domain_name finger --task_name spin --algorithm sgsac79 --seed 2
CUDA_VISIBLE_DEVICES=2 python src/my_train_all.py --domain_name cartpole --task_name swingup --algorithm sgsac79 --seed 2
CUDA_VISIBLE_DEVICES=3 python src/my_train_all.py --domain_name ball_in_cup --task_name catch --algorithm sgsac79 --seed 2

CUDA_VISIBLE_DEVICES=3 python src/my_train_all.py --domain_name cartpole --task_name swingup --algorithm sgsac74 --seed 2
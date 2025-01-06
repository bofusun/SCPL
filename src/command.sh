Xvfb :1 -screen 0 1024x768x24 &
cd /data/sjb_workspace/rl_generalization/dmcontrol-generalization-benchmark-main
conda activate svea_new
export DISPLAY=:1

CUDA_VISIBLE_DEVICES=0 python src/my_train_all.py --algorithm sgsac79 --domain_name walker --task_name stand --seed 0
CUDA_VISIBLE_DEVICES=0 python src/my_train_all.py --algorithm sgsac79 --domain_name walker --task_name stand --seed 1
CUDA_VISIBLE_DEVICES=1 python src/my_train_all.py --algorithm sgsac79 --domain_name walker --task_name stand --seed 2

CUDA_VISIBLE_DEVICES=1 python src/my_train_all.py --algorithm sgsac79 --domain_name walker --task_name walk --seed 0
CUDA_VISIBLE_DEVICES=2 python src/my_train_all.py --algorithm sgsac79 --domain_name walker --task_name walk --seed 1
CUDA_VISIBLE_DEVICES=2 python src/my_train_all.py --algorithm sgsac79 --domain_name walker --task_name walk --seed 2
export NCCL_P2P_DISABLE=1
python setup.py develop --no_cuda_ext
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=4317 basicsr/train.py -opt options/train/GoPro.yml --launcher pytorch

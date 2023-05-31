# FCOS: Fully Convolutional One-Stage Object Detection

## Training

The following command line will train FCOS_imprv_R_50_FPN_1x on 8 GPUs with Synchronous Stochastic Gradient Descent (SGD):

    python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port=$((RANDOM + 10000)) \
        tools/train_net.py \
        --config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml \
        DATALOADER.NUM_WORKERS 2 \
        OUTPUT_DIR training_dir/fcos_imprv_R_50_FPN_1x

## Inference
The inference command line on coco minival split:

    python tools/test_net.py \
        --config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml \
        MODEL.WEIGHT FCOS_imprv_R_50_FPN_1x.pth \
        TEST.IMS_PER_BATCH 4    


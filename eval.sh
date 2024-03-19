for pth in '11'
do
    PUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/dn_detr.sh \
        --model_name dn_detr \
        --use_dn \
        --batch_size 10\
        --num_workers 0 \
        --output_dir "./logs/eval" \
        --Task 1 \
        --CL_Limited 0 \
        --start_epoch 0 \
        --start_task 0 \
        --Total_Classes 90 \
        --Branch_Incremental \
        --verbose \
        --eval \
        --all_data \
        --pretrained_model "./logs/upperbound/checkpoints/cp_01_01_$pth.pth" \
        --coco_path "../COCODIR/" \
        --test_file_list coco \
        --orgcocopath
done
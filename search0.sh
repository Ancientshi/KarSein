SEED=42
NETNAME=KARSEIN
DATASET=ml-1m
TASK_LIST=('CTR')

EPOCHS=6
BATCH_SIZE_LIST=(512 1024 4096)
BitWIDTH_LIST=('32 32' '64 64' '64 32' '128 64' '64 32 32')
VecWIDTH_LIST=('6 6' '12 12' '6 12' '12 6' '6 6 6')
GRID_LIST=(10 5 3)
K_LIST=(1 2 3)
LR_LIST=(0.002 0.003 0.004)
EMB_DIM_LIST=(16 32)
REG_LIST=(0.01)
Pairwise_Multiplication='0 1 2'

for TASK in ${TASK_LIST[@]}
do
    for BATCH_SIZE in ${BATCH_SIZE_LIST[@]}
    do
        for BitWIDTH in "${BitWIDTH_LIST[@]}"
        do
            for VecWIDTH in "${VecWIDTH_LIST[@]}"
            do
                for GRID in ${GRID_LIST[@]}
                do
                    for K in ${K_LIST[@]}
                    do
                        for LR in ${LR_LIST[@]}
                        do
                            for EMB_DIM in ${EMB_DIM_LIST[@]}
                            do
                                for REG in ${REG_LIST[@]}
                                do
                                    python main.py \
                                        --seed $SEED \
                                        --lr $LR \
                                        --reg $REG \
                                        --emb_dim $EMB_DIM \
                                        --epochs $EPOCHS \
                                        --netname $NETNAME \
                                        --bit_width $BitWIDTH \
                                        --vec_width $VecWIDTH \
                                        --pairwise_multiplication $Pairwise_Multiplication \
                                        --use_bit_wise 1 \
                                        --use_vec_wise 1 \
                                        --dataset $DATASET \
                                        --grid $GRID \
                                        --k $K \
                                        --task $TASK \
                                        --batch_size $BATCH_SIZE \
                                        --device_index 0 \
                                        --note 'None'
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

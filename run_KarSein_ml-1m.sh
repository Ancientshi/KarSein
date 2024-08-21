
SEED=42
NETNAME=KARSEIN
DATASET=ml-1m
TASK_LIST=('CTR')

EPOCHS=6
BATCH_SIZE=512
BitWIDTH='32 32'
VecWIDTH='12 12'
GRID=10
K=1
LR=0.003
Pairwise_Multiplication='0 1 2'
EMB_DIM=16
REG=0.001

for TASK in ${TASK_LIST[@]}
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
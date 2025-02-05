source activate 1

DATASET='shiny_blender'
EXPS=('car' 'toaster' 'ball'  'coffee' 'helmet' 'teapot' )
for EXP in "${EXPS[@]}"; do
    echo "scene: $EXP"

    python run.py \
    --mode train \
    --expname "$EXP" \
    --dataset_path ./datasets/"$DATASET"/"$EXP" \
    --output_dir ./results/"$DATASET" \
    --dataset_type blender \
    --config config/shiny_blender.py \
    --no_reload \
    --prefix "$EXP" \
    --suffix 0 \
    --geometry_searching True \
    --coarse_training True

    python run.py \
    --mode train \
    --expname "$EXP" \
    --dataset_path ./datasets/"$DATASET"/"$EXP" \
    --output_dir ./results/"$DATASET" \
    --dataset_type blender \
    --config config/shiny_blender.py \
    --no_reload \
    --prefix "$EXP" \
    --suffix 0 \
    --fine_training True

done

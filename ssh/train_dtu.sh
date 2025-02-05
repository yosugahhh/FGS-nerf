source activate 1

DATASET='dtu'
EXPS=('scan24')
      #  'scan37' 'scan65' 'scan69' 'scan83' 'scan97' 'scan105' 'scan106' 'scan110' 'scan114' 'scan118' 'scan122''scan40' 'scan55' 'scan63'
for EXP in "${EXPS[@]}"; do
    echo "scene: $EXP"

     python run.py \
     --mode train \
     --expname "$EXP" \
     --dataset_path ./datasets/"$DATASET"/"$DATASET"_"$EXP" \
     --output_dir ./results/"$DATASET" \
     --dataset_type dtu \
     --config config/dtu.py \
     --no_reload \
     --prefix "$DATASET" \
     --suffix "$EXP" \
     --geometry_searching True \
     --coarse_training True

    python run.py \
    --mode train \
    --expname "$EXP" \
    --dataset_path ./datasets/"$DATASET"/"$DATASET"_"$EXP" \
    --output_dir ./results/"$DATASET" \
    --dataset_type dtu \
    --config config/dtu.py \
    --no_reload \
    --prefix "$DATASET" \
    --suffix "$EXP" \
    --fine_training True
done
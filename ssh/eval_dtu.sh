source activate 1

DATASET='dtu'
EXPS=( 'scan24' 'scan37' )
#'scan40' 'scan55' 'scan63'  'scan69' 'scan83' 'scan97' 'scan105' 'scan114'  'scan106' 'scan65' 'scan110' 'scan118' 'scan122'
for EXP in "${EXPS[@]}"; do
    echo "scene: $EXP"

    python run.py \
    --mode eval \
    --expname "$EXP" \
    --dataset_path ./datasets/"$DATASET"/"$DATASET"_"$EXP" \
    --output_dir ./results/"$DATASET" \
    --dataset_type dtu \
    --config config/dtu.py \
    --no_reload \
    --prefix "$DATASET" \
    --suffix "$EXP" \

done
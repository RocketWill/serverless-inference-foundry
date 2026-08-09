# GPU MEMORY 1289MiB
WEIGHTS_DIR="/path/to/weights/dir"
WEIGHTS_FILE="weights file"
CUDA_VISIBLE_DEVICES=0

nuctl deploy \
--project-name lkdi \
--namespace nuclio \
--platform local \
--path ${PWD} \
--volume ${PWD}/main.py:/opt/nuclio/main.py \
--volume ${PWD}/infer.py:/opt/nuclio/infer.py \
--volume ${PWD}/tools.py:/opt/nuclio/tools.py \
--volume ${WEIGHTS_DIR}:/opt/nuclio/weights \
--env WEIGHTS_FILE=${WEIGHTS_FILE} \
--env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
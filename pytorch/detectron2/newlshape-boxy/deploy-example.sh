# GPU MEMORY 2117MiB
WEIGHTS_DIR="/path/to/weights/dir"
DET_WEIGHTS_FILE="weights file"
KPT_WEIGHTS_FILE="weights file"
CUDA_VISIBLE_DEVICES=0

nuctl deploy \
--project-name lkdi \
--namespace nuclio \
--platform local \
--path ${PWD} \
--volume ${PWD}/main.py:/opt/nuclio/main.py \
--volume ${PWD}/d2_infer.py:/opt/nuclio/d2_infer.py \
--volume ${PWD}/yolov5_infer.py:/opt/nuclio/yolov5_infer.py \
--volume ${PWD}/two_stage_infer.py:/opt/nuclio/two_stage_infer.py \
--volume ${PWD}/tools.py:/opt/nuclio/tools.py \
--volume ${PWD}/configs:/opt/nuclio/configs \
--volume ${WEIGHTS_DIR}:/opt/nuclio/weights \
--env DET_WEIGHTS_FILE=${DET_WEIGHTS_FILE} \
--env KPT_WEIGHTS_FILE=${KPT_WEIGHTS_FILE} \
--env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \

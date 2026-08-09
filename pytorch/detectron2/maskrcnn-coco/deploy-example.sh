# GPU MEMORY 3565MiB
CUDA_VISIBLE_DEVICES=0

nuctl deploy \
--project-name lkdi \
--namespace nuclio \
--platform local \
--path ${PWD} \
--volume ${PWD}/main.py:/opt/nuclio/main.py \
--volume ${PWD}/infer.py:/opt/nuclio/infer.py \
--volume ${PWD}/utils.py:/opt/nuclio/utils.py \
--env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \

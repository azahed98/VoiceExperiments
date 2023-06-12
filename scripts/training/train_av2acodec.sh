HOME_DIR=.
CONFIGS_DIR=${HOME_DIR}/configs

PIPELINE_CFG=${CONFIGS_DIR}/pipeline_configs/av2acodec.yaml
TRAINING_CFG=${CONFIGS_DIR}/training_configs/encodec_vctk.yaml

python3 ${HOME_DIR}/VoiceExperiments/train.py --pipeline_config=${PIPELINE_CFG} --training_config=${TRAINING_CFG} --compile
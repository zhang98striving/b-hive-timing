#!/bin/bash

test_version="test_offline"

export LXUSERNAME=$(whoami)
export TESTDIRECTORY="${B_HIVE_DIR}/tests"

# Abort on errors
set -e
trap 'echo "An error occurred. Exiting..."; exit 1' ERR

if [ -z "$LXUSERNAME" ]; then
    echo "LXUSERNAME is not set. Please set it in your environment or export manually."
    exit 1
fi
if [ -z "$TESTDIRECTORY" ]; then
    echo "TESTDIRECTORY is not set. Please set it in your environment or export manually."
    exit 1
fi

# Ensure data directory exists
mkdir -p "$TESTDIRECTORY/data"
DEST_FILE="$TESTDIRECTORY/data/ntuple_merged_0.root"

# Copy or download the file
if [ ! -f $TESTDIRECTORY/data/ntuple_merged_0.root ]; then
    echo "Copying test file to: $DEST_FILE"
    # Define the source file path
    SOURCE_FILE=/eos/cms/store/group/phys_btag/ParticleTransformer/merged/ntuple_merged_0.root
    if [ -f "$SOURCE_FILE" ]; then
        echo "File exists locally."
        cp $SOURCE_FILE $DEST_FILE
    else
        echo "File not found locally. Attempting remote download..."
        scp -r $LXUSERNAME@lxplus.cern.ch:$SOURCE_FILE $DEST_FILE
        if [ $? -eq 0 ]; then
            echo "File downloaded successfully."
        else
            echo "Failed to download the file. Aborting."
            exit 1
        fi
    fi
fi

# Create filelist
echo "$DEST_FILE" > "$TESTDIRECTORY/data/filelist.txt"

for task in DatasetConstructorTask TrainingTask InferenceTask ROCCurveTask; do
    for config in part_run3 part_fp16_run3 offline_run3 UParT_v0_run3; do
        path="$DATA_PATH/$task/$config/$test_version"
        if [ -d "$path" ]; then
            rm -r "$path"
        fi
    done
done
echo "Begining test..."

printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+      Test 1:  part_run3 + ParticleNet_InPro + epoch_lin_decay + Adam                        +\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

time law run ROCCurveTask \
        --config part_run3 \
        --training-version $test_version \
        --dataset-version $test_version \
        --filelist $TESTDIRECTORY/data/filelist.txt \
        --test-dataset-version $test_version \
        --test-filelist $TESTDIRECTORY/data/filelist.txt  \
        --model-name ParticleNet_InPro \
        --epochs 1 \
        --batch-size 512 \
        --lr-scheduler epoch_lin_decay \
        --lr-decay-factor 0.1 \
        --optimizer Adam \
        --betas 0.95,0.999 
        
printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+      Test 2:  part_fp16_run3 + ParticleTransformer + batch_cosine_warmup + AdamW            +\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

time law run ROCCurveTask \
        --config part_fp16_run3 \
        --training-version $test_version \
        --dataset-version $test_version \
        --filelist $TESTDIRECTORY/data/filelist.txt \
        --test-dataset-version $test_version \
        --test-filelist $TESTDIRECTORY/data/filelist.txt  \
        --model-name ParticleTransformer \
        --epochs 2 \
        --batch-size 512 \
        --lr-scheduler batch_cosine_warmup \
        --lr-decay-factor 0.1 \
        --optimizer AdamW \
        --betas 0.9,0.999

printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+      Test 3:  offline_run3 + DeepJet + epoch_lin_decay + RAdam                              +\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

time law run ROCCurveTask \
        --config offline_run3 \
        --training-version $test_version \
        --dataset-version $test_version \
        --filelist $TESTDIRECTORY/data/filelist.txt \
        --test-dataset-version $test_version \
        --test-filelist $TESTDIRECTORY/data/filelist.txt  \
        --model-name DeepJet \
        --epochs 2 \
        --batch-size 512 \
        --lr-scheduler epoch_lin_decay \
        --lr-decay-factor 0.01 \
        --optimizer RAdam \
        --betas 0.95,0.999 

printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+      Test 4:  UParT_v0_run3 + UParT_v0 + batch_lin_decay + AdamW                            +\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

time law run ROCCurveTask \
        --config UParT_v0_run3 \
        --training-version $test_version \
        --dataset-version $test_version \
        --filelist $TESTDIRECTORY/data/filelist.txt \
        --test-dataset-version $test_version \
        --test-filelist $TESTDIRECTORY/data/filelist.txt  \
        --model-name UParT_v0 \
        --epochs 1 \
        --batch-size 512 \
        --lr-scheduler batch_lin_decay \
        --lr-decay-factor 0.01 \
        --optimizer Adam \
        --betas 0.9,0.999 

printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "++                                GREAT SUCCESS!!!                                           ++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"
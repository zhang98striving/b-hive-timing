#!/bin/bash

test_version="test_hlt"

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
mkdir -p "$TESTDIRECTORY"
mkdir -p "$TESTDIRECTORY/data"
mkdir -p "$TESTDIRECTORY/data/HLT"
DEST_FILE="$TESTDIRECTORY/data/HLT/hlt_test_TT.root"

# Copy or download the file
if [ ! -f $DEST_FILE ]; then
    echo "Copying test file to: $DEST_FILE"
    # Define the source file path
    SOURCE_FILE=/eos/cms/store/group/phys_btag/HLT/Run3/2023_08_22/small_test.root 
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
FILELIST="$TESTDIRECTORY/data/HLT/filelist.txt"
echo "$DEST_FILE" > "$FILELIST"

echo "Deleting old HLT tests."
for task in DatasetConstructorTask TrainingTask InferenceTask ROCCurveTask; do
    for config in hlt_run3; do
        path="$DATA_PATH/$task/$config/$test_version"
        if [ -d "$path" ]; then
            rm -r "$path"
        fi
    done
done

echo "Begining testing..."

printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+      Test 1:  hlt_run3 + DeepJet + epoch_lin_decay + AdamW + attack (pgd)                             +\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

time law run ROCCurveTask \
        --config hlt_run3 \
        --training-version $test_version \
        --dataset-version $test_version \
        --filelist $FILELIST \
        --test-dataset-version $test_version \
        --test-filelist $FILELIST  \
        --model-name DeepJet \
        --epochs 2 \
        --batch-size 512 \
        --attack pgd \
        --attack-magnitude 0.1 \
        --test-attack pgd \
        --test-attack-magnitude 0.1 \
        --DatasetConstructorTask-chunk-size 10000 \
        --lr-scheduler epoch_lin_decay \
        --lr-decay-factor 0.1 \
        --optimizer AdamW \
        --betas 0.95,0.999 

printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+      Test 2:  hlt_run3 + DeepJetTransformer + batch_lin_decay + RAdam + attack (jetfool)              +\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

time law run ROCCurveTask \
        --config hlt_run3 \
        --training-version $test_version \
        --dataset-version $test_version \
        --filelist $FILELIST \
        --test-dataset-version $test_version \
        --test-filelist $FILELIST  \
        --model-name DeepJetTransformer \
        --epochs 2 \
        --batch-size 512 \
        --attack jetfool \
        --attack-magnitude 0.1 \
        --test-attack jetfool \
        --test-attack-magnitude 0.1 \
        --DatasetConstructorTask-chunk-size 10000 \
        --lr-scheduler batch_lin_decay \
        --lr-decay-factor 0.1 \
        --optimizer RAdam \
        --betas 0.95,0.999 

printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+      Test 3:  hlt_run3 + ParticleNet_InPro + batch_lin_decay + Adam + attack (minimizer)              +\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

time law run ROCCurveTask \
        --config hlt_run3 \
        --training-version $test_version \
        --dataset-version $test_version \
        --filelist $FILELIST \
        --test-dataset-version $test_version \
        --test-filelist $FILELIST  \
        --model-name ParticleNet_InPro \
        --epochs 2 \
        --batch-size 512 \
        --attack minimizer \
        --attack-magnitude 0.1 \
        --test-attack minimizer \
        --test-attack-magnitude 0.1 \
        --DatasetConstructorTask-chunk-size 10000 \
        --lr-scheduler batch_lin_decay \
        --lr-decay-factor 0.1 \
        --optimizer Adam \
        --betas 0.95,0.999 
        
printf "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "++                                GREAT SUCCESS!!!                                           ++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
printf "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n"

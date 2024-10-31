#!/bin/bash
test_version=$1

set -e
# check if LXUSERNAME and TESTDIRECTORY are set
if [ -z "$LXUSERNAME" ]; then
    echo "LXUSERNAME is not set"
fi
if [ -z "$TESTDIRECTORY" ]; then
    echo "TESTDIRECTORY is not set"
fi
mkdir -p $TESTDIRECTORY/data

# download data
if [ ! -f $TESTDIRECTORY/data/ntuple_merged_0.root ]; then
    scp -r $LXUSERNAME@lxplus.cern.ch:/eos/cms/store/group/phys_btag/ParT_2024/merged_mc/ntuple_merged_0.root $TESTDIRECTORY/data/ntuple_merged_0.root
fi
#
# creatte test filelist
echo "$TESTDIRECTORY/data/ntuple_merged_0.root" > $TESTDIRECTORY/data/filelist.txt
law run DatasetConstructorTask --dataset-version $test_version --filelist $TESTDIRECTORY/data/filelist.txt --coffea-worker 1 --config offline_run3
law run ROCCurveTask --training-version $test_version --dataset-version $test_version --config offline_run3 --model-name DeepJet --epochs 1 --test-dataset-version $test_version
law run ROCCurveTask --training-version $test_version --dataset-version $test_version --config offline_run3 --model-name DeepJet --epochs 1 --test-dataset-version $test_version --test-attack pgd --test-attack-iterations 1

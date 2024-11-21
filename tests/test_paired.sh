
#!/bin/bash
test_version="test_paired"

# set so script aborts when something fails
set -e

if [ -z "$LXUSERNAME" ]; then
    echo "LXUSERNAME is not set. Please set it in your environment."
    exit 1
fi
if [ -z "$TESTDIRECTORY" ]; then
    echo "TESTDIRECTORY is not set. Please set it in your environment."
    exit 1
fi
mkdir -p $TESTDIRECTORY/data

# download data
if [ ! -f $TESTDIRECTORY/data/PAIReD_LL_CC_BB/paired_test.root ]; then
    mkdir -p $TESTDIRECTORY/data/PAIReD_LL_CC_BB/
    scp -r $LXUSERNAME@lxplus.cern.ch:/eos/cms/store/group/phys_btag/b-hive/test_files/PAIReD_LL_CC_BB/paired_test.root $TESTDIRECTORY/data/PAIReD_LL_CC_BB/paired_test.root
fi
echo "begin teardown of old tests"
echo "a" | law run ROCCurveTask --dataset-version $test_version --training-version $test_version --config PAIReD_ParT_cls --model-name PAIReDTagger --epochs 2 --batch-size 512 --attack pgd --attack-magnitude 0.02 --attack-iterations 2 --test-dataset-version $test_version --remove-output -1
#
# create test filelist
echo "$TESTDIRECTORY/data/PAIReD_LL_CC_BB/paired_test.root" > $TESTDIRECTORY/data/PAIReD_LL_CC_BB/filelist_paired.txt
law run DatasetConstructorTask --dataset-version $test_version --filelist $TESTDIRECTORY/data/PAIReD_LL_CC_BB/filelist_paired.txt --coffea-worker 4 --config PAIReD_ParT_cls
law run ROCCurveTask --dataset-version $test_version --training-version $test_version --config PAIReD_ParT_cls --model-name PAIReDTagger --epochs 2 --batch-size 512 --attack pgd --attack-magnitude 0.02 --attack-iterations 2 --test-dataset-version $test_version

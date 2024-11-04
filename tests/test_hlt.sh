test_version="test_hlt"

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
if [ ! -f $TESTDIRECTORY/data/hlt_test_TT.root ]; then
    scp -r $LXUSERNAME@lxplus.cern.ch:/eos/cms/store/group/phys_btag/HLT/Run3/2023_08_22/small_test.root $TESTDIRECTORY/data/hlt_test_TT.root
fi

echo "begin teardown of old tests"
echo "a" | law run ROCCurveTask --training-version $test_version --dataset-version $test_version --config hlt_run3 --model-name DeepJetHLT --epochs 1 --test-dataset-version $test_version --remove-output -1
echo "a" | law run ROCCurveTask --training-version $test_version --dataset-version $test_version --config hlt_run3 --model-name DeepJetHLT --epochs 1 --test-dataset-version $test_version --test-attack pgd --test-attack-iterations 1 --remove-output -1
#
# creatte test filelist
echo "$TESTDIRECTORY/data/hlt_test_TT.root" > $TESTDIRECTORY/data/hlt_test.txt
law run DatasetConstructorTask --dataset-version $test_version --filelist $TESTDIRECTORY/data/hlt_test.txt --coffea-worker 4 --config hlt_run3 --chunk-size 1000
law run ROCCurveTask --training-version $test_version --dataset-version $test_version --config hlt_run3 --model-name DeepJetHLT --epochs 1 --test-dataset-version $test_version
law run ROCCurveTask --training-version $test_version --dataset-version $test_version --config hlt_run3 --model-name DeepJetHLT --epochs 1 --test-dataset-version $test_version --test-attack pgd --test-attack-iterations 1

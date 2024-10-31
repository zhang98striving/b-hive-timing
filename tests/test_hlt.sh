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
if [ ! -f $TESTDIRECTORY/data/hlt_test_TT.root ]; then
    scp -r $LXUSERNAME@lxplus.cern.ch:/eos/cms/store/group/phys_btag/HLT/Run3/2023_08_22/small_test.root $TESTDIRECTORY/data/hlt_test_TT.root
fi
#
# creatte test filelist
echo "$TESTDIRECTORY/data/hlt_test_TT.root" > $TESTDIRECTORY/data/hlt_test.txt
law run DatasetConstructorTask --dataset-version $test_version --filelist $TESTDIRECTORY/data/hlt_test.txt --coffea-worker 1 --config hlt_run3 --chunk-size 1000
law run ROCCurveTask --training-version $test_version --dataset-version $test_version --config hlt_run3 --model-name DeepJet --epochs 1 --test-dataset-version $test_version
law run ROCCurveTask --training-version $test_version --dataset-version $test_version --config hlt_run3

file_path = 'test_files_formal500_01.txt'

with open(file_path, 'w') as f:
    for i in range(81,100):
        print('root://eoscms.cern.ch///eos/cms/store/group/cmst3/group/deepjet/Run3/DJ_Run3/Puppi3_test/ntuple_ttbar/ntuple_ttbar_semilep_puppi_122X_%d.root'%i, file = f)

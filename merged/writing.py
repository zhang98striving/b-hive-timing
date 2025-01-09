for k in range(20):

    file_path1 = 'training_files%d.txt'%k

    with open(file_path1, 'w') as f:

        for i in range(k*50,(k+1)*50):
            print('root://eoscms.cern.ch///eos/cms/store/group/phys_btag/ParticleTransformer/merged/ntuple_merged_%d.root'%i, file = f)

    file_path2 = 'test_files%d.txt'%k

    with open(file_path2, 'w') as f:

        for i in range(1134,1184):
            print('root://eoscms.cern.ch///eos/cms/store/group/phys_btag/ParticleTransformer/merged/ntuple_merged_%d.root'%i, file = f)



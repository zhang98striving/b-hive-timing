for k in range(10):
	
	file_path1 = 'merged_%d.sh'%k
	with open(file_path1, 'w') as f:
		
		print('#!/bin/bash\n', file = f)

		print('cd /home/home1/institut_3a/zhang/', file = f)
		print('source ~/.bashrc\n', file = f)
		
		print('ulimit -u 32768\n', file = f)

		print('mamba activate b_hive\n', file = f)
		
		print('cd /home/home1/institut_3a/zhang/Documents/b-hive', file = f)
		print('source setup.sh\n', file = f)
		
		print('law index\n', file = f)
		
		print('law run DatasetConstructorTask --dataset-version pt30_%d  --filelist /home/home1/institut_3a/zhang/Documents/b-hive/merged/training_files%d.txt --coffea-worker 32 --config offline_run3 --chunk-size 100000\n'%(k,k), file = f)
		
		print('law run TrainingTask --training-version pt30_training%d --dataset-version pt30_%d --config offline_run3 --model-name DeepJet --epochs 5 --batch-size 512\n'%(k,k), file = f)		
		
		print('law run ROCCurveTask --training-version pt30_training%d --dataset-version pt30_%d --test-filelist /home/home1/institut_3a/zhang/Documents/b-hive/merged/test_files%d.txt --test-dataset-version pt30_test%d --config offline_run3 --model-name DeepJet --epochs 5 --batch-size 512'%(k,k,k,k), file = f)

		
	file_path2 = 'condor%d.sub'%k
	with open(file_path2, 'w') as f:
				
		print('executable = /home/home1/institut_3a/zhang/Documents/b-hive/merged_%d.sh\n'%k, file = f)
		

		print('use_x509userproxy = true\n', file = f)
		

		print('should_transfer_files = YES\n', file = f)
		

		print('error  = err.$(ClusterId).$(ProcId)', file = f)
		print('output = out.$(ClusterId).$(ProcId)', file = f)
		print('log    = log.$(ClusterId).$(ProcId)\n', file = f)
		
		
		
		print('request_memory    = 100000', file = f)
		print('request_GPUs      = 1', file = f)
		print('request_cpus = 3\n', file = f)
		
		
		

		print('initialdir = /net/scratch_cms3a/zhang/b-hive/logs\n', file = f)
		


		print('requirements = (OpSys == "LINUX") && (Arch == "X86_64")\n', file = f)
		

		print('queue', file = f)
		


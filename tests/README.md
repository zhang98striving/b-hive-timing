# Tests

There are currently 4 tests to be checked before commiting any changes to the repository. They also provide nice examples of using b-hive itself. In case of successful finish of test the message "GEAT SUCCESS" will appear.

1) Test_offline.sh
   There are 4 tests inside.
   - Test 1:  part_run3 + ParticleNet_InPro + epoch_lin_decay + Adam 
   - Test 2:  part_fp16_run3 + DeepJet + batch_cosine_warmup + AdamW + attack (pgd)         
   - Test 3:  offline_run3 + ParticleTransformer + epoch_lin_decay + RAdam + attack (jetfool) + test_attack (minimizer)  
   - Test 4:  UParT_v0_run3 + UParT_v0 + batch_lin_decay + AdamW   --  attacks are not yet implemented for UParT   
   
2) Test_HLT.sh
   - Test 1:  hlt_run3 + DeepJet + epoch_lin_decay + AdamW + attack (pgd)
   - Test 2:  hlt_run3 + DeepJetTransformer + batch_lin_decay + RAdam + attack (jetfool)
   - Test 3:  hlt_run3 + ParticleNet_InPro + batch_lin_decay + Adam + attack (minimizer)
  
3) Test_paired.sh
   - Test 1:  PAIReD_ParT_cls + PAIReDTagger + epoch_lin_decay + AdamW + attack (pgd)
  
4) Test_MoD.sh (to be implemented correctly)
   - Test 1:  mod_offline_run3 + MoDJet + batch_lin_decay + AdamW + attack (pgd) + test_attack (pgd)  
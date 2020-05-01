#load the library for inference on mixture models
library(depmixS4)
#load the library for parallel computing
library(parallel)


sourses = c('BTC', 'BTC+c', 'SEC', 'SEC+c', 'company_cased', 'company_type_detector', 'company_uncased', 'compound_detector', 
            'Crowd2003', 'Crowd2003+c', 'core_web_md', 'core_web_md+c', 'crunchbase_cased', 'crunchbase_uncased', 'date_detector', 
            'doc_history', 'doc_majority_cased', 'doc_majority_uncased', 'full_name_detector', 'geo_cased', 'geo_uncased', 
            'infrequent_compound_detector', 'infrequent_nnp_detector', 'infrequent_proper2_detector', 'infrequent_proper_detector', 
            'legal_detector', 'misc_detector', 'money_detector', 'multitoken_company_cased', 'multitoken_company_uncased', 
            'multitoken_crunchbase_cased', 'multitoken_crunchbase_uncased', 'multitoken_geo_cased', 'multitoken_geo_uncased', 
            'multitoken_product_cased', 'multitoken_product_uncased', 'multitoken_wiki_cased', 'multitoken_wiki_small_cased', 
            'multitoken_wiki_small_uncased', 'multitoken_wiki_uncased', 'nnp_detector', 'number_detector', 'product_cased', 
            'product_uncased', 'proper2_detector', 'proper_detector', 'snips', 'time_detector', 'wiki_cased', 'wiki_small_cased', 
            'wiki_small_uncased', 'wiki_uncased')

classes = c('B-CARDINAL', 'I-CARDINAL', 'L-CARDINAL', 'U-CARDINAL', 'B-COMPANY', 'I-COMPANY', 'L-COMPANY', 'U-COMPANY', 'B-DATE',
            'I-DATE', 'L-DATE', 'U-DATE', 'B-EVENT', 'I-EVENT', 'L-EVENT', 'U-EVENT', 'B-FAC', 'I-FAC', 'L-FAC', 'U-FAC', 'B-GPE', 
            'I-GPE', 'L-GPE', 'U-GPE', 'B-LANGUAGE', 'I-LANGUAGE', 'L-LANGUAGE', 'U-LANGUAGE', 'B-LAW', 'I-LAW', 'L-LAW', 'U-LAW', 
            'B-LOC', 'I-LOC', 'L-LOC', 'U-LOC', 'B-MONEY', 'I-MONEY', 'L-MONEY', 'U-MONEY', 'B-NORP', 'I-NORP', 'L-NORP', 'U-NORP', 
            'B-ORDINAL', 'I-ORDINAL', 'L-ORDINAL', 'U-ORDINAL', 'B-ORG', 'I-ORG', 'L-ORG', 'U-ORG', 'B-PERCENT', 'I-PERCENT', 
            'L-PERCENT', 'U-PERCENT', 'B-PERSON', 'I-PERSON', 'L-PERSON', 'U-PERSON', 'B-PRODUCT', 'I-PRODUCT', 'L-PRODUCT', 
            'U-PRODUCT', 'B-QUANTITY', 'I-QUANTITY', 'L-QUANTITY', 'U-QUANTITY', 'B-TIME', 'I-TIME', 'L-TIME', 'U-TIME', 
            'B-WORK_OF_ART', 'I-WORK_OF_ART', 'L-WORK_OF_ART', 'U-WORK_OF_ART','O')


#input data
input = "/data/crowd.csv"
#input tockens
tokens = "/data/crowd_tokens.csv"
#labels with two columns to resolve the label matching problem
match.lab = "/data/crowdsourced.agg"
#parameters of the model
param = read.csv("param.csv",header = T)
size = param[1,2]
states = param[2,2]
draw = param[3,2]
nseq = param[4,2]
rely = param[5,2]
#specify your number of cores
n.cores = detectCores(all.tests = FALSE, logical = TRUE) - 1
if(is.na(n.cores))
  n.cores = 4

#a function implementing confusion matrix Mixtures of Multinomoals
CM = function(id)
{
  
  if (!dir.exists(paste0(getwd(),"/results/MixtureModelsCrowd/"))){
    dir.create(paste0(getwd(),"/results/MixtureModelsCrowd/"),recursive = T)
  }
  if(file.exists(paste0(getwd(),"/results/MixtureModelsCrowd/resprobunmatched",id,".csv")))
  {
    print("A")
    return("A")
  }
  print(id)
  X = read.table(input,skip = size*(id-1),nrows = size,sep = ",")#[-1]
  X[X==0]=states
  nsours = dim(X)[2]
  matches = read.table(match.lab,skip = size*(id-1),nrows = size,sep = ",")
  matches[matches==0]=77
  
  instart = rep(0.0001,states)
  set.seed(id)
  resp = NULL
  ns = NULL
  for(i in 1:nsours)
  {
    
    if(length(levels(as.factor(X[,i])))==1)
    {  
      ns = c(ns,i)
      next
    }
    
    if(i==rely)
    {
      freq = table(X[,i])
      instart[as.integer(names(freq))]  =  instart[as.integer(names(freq))]+ freq
    }
  }
  
  X[,ns]=NULL
  
  rely = which(names(X)=="V12")
  if(dim(X)[2]==0)
    X$V1 = c(rep(states,size-1),1)
  
  nsours = dim(X)[2]
  
  
  instart = instart/sum(instart)
  
  
  model = depmixS4::mix(response = lapply(FUN = function(X) as.formula(paste0(X,"~1")),X = names(X)), family = lapply(function(X)multinomial(),X=1:nsours),instart = instart,nstates = states,data = X)
  mod = fit(model,verbose =T,emcontrol=em.control(maxit = 25))
  res = posterior(mod)$state 
  
  if(length(rely)==0)
  {
    X$rely = states
    rely = dim(X)[2]
  }
  resmap = array(0,states)
  
  for(resv in 1:states)
  {
    tmpres = res == resv
    smax = -10000000
    tmplab = states
    for(ann in 1:states)
    {
      tmpann =  X[,rely] == ann
      tmpsum = (cor(tmpres, tmpann))
      if(is.na(tmpsum))
        next
      if(tmpsum>=smax)
      {
        smax = tmpsum
        tmplab = ann
      }
      if(smax<0.1)
      {  
        tmplab = states
      }
    }
    resmap[resv] = tmplab
  }
  
  
  rprob = posterior(mod)[,2:(states+1)]
  rprob.new = rprob*0
  write.csv( rprob , paste0(getwd(),"/results/MixtureModelsCrowd/resprobunmatched",id,".csv"),row.names = F)
  write.csv(res, paste0(getwd(),"/results/MixtureModelsCrowd/resunmatched",id,".csv"),row.names = F)
  
  res = resmap[res]
  for(i in 1:length(resmap))
  {
    idss = which(resmap == i)
    if(length(idss)==1)
      rprob.new[,i] = rprob[,idss]
    if(length(idss)>1)
      rprob.new[,i] = rowSums(rprob[,idss]) 
    
  }
  
  
  write.csv(cbind(classes[X[,rely]],classes[res]),paste0(getwd(),"/results/MixtureModelsCrowd/check",id,".csv"))
  rprob.new = rprob.new[,c(states,1:(states-1))]
  names(rprob.new) = paste0("T",0:(states-1))
  write.csv(rprob.new, paste0(getwd(),"/results/MixtureModelsCrowd/resprob",id,".csv"),row.names = F)
  res[res==states]=0
  write.csv(res, paste0(getwd(),"/results/MixtureModelsCrowd/res",id,".csv"),row.names = F)
  
  
  res = posterior(mod)$state 
  
  X[,rely] = matches$V1
  resmap = array(0,states)
  
  for(resv in 1:states)
  {
    tmpres = res == resv
    smax = -10000000
    tmplab = states
    for(ann in 1:states)
    {
      tmpann =  X[,rely] == ann
      tmpsum = (cor(tmpres, tmpann))
      if(is.na(tmpsum))
        next
      if(tmpsum>=smax)
      {
        smax = tmpsum
        tmplab = ann
      }
      if(smax<0.1)
      {  
        tmplab = states
      }
    }
    resmap[resv] = tmplab
  }
  
  
  rprob = posterior(mod)[,2:(states+1)]
  rprob.new = rprob*0
  write.csv( rprob , paste0(getwd(),"/results/MixtureModelsCrowd/resprobunmatchedMV",id,".csv"),row.names = F)
  
  
  
  write.csv(res, paste0(getwd(),"/results/MixtureModelsCrowd/resunmatchedMV",id,".csv"),row.names = F)
  
  res = resmap[res]
  
  for(i in 1:length(resmap))
  {
    idss = which(resmap == i)
    if(length(idss)==1)
      rprob.new[,i] = rprob[,idss]
    if(length(idss)>1)
      rprob.new[,i] = rowSums(rprob[,idss]) 
    
  }
  
  
  write.csv(cbind(classes[X[,rely]],classes[res]),paste0(getwd(),"/results/MixtureModelsCrowd/checkMV",id,".csv"))
  
  rprob.new = rprob.new[,c(states,1:(states-1))]
  
  names(rprob.new) = paste0("T",0:(states-1))
  
  write.csv(rprob.new, paste0(getwd(),"/results/MixtureModelsCrowd/resprobMV",id,".csv"),row.names = F)
  
  res[res==states]=0
  
  write.csv(res, paste0(getwd(),"/results/MixtureModelsCrowd/resMV",id,".csv"),row.names = F)
  
  
  res = posterior(mod)$state 
  
  X[,rely] = matches$V2
  resmap = array(0,states)
  
  for(resv in 1:states)
  {
    tmpres = res == resv
    smax = -10000000
    tmplab = states
    for(ann in 1:states)
    {
      tmpann =  X[,rely] == ann
      tmpsum = (cor(tmpres, tmpann))
      if(is.na(tmpsum))
        next
      if(tmpsum>=smax)
      {
        smax = tmpsum
        tmplab = ann
      }
      if(smax<0.1)
      {  
        tmplab = states
      }
    }
    resmap[resv] = tmplab
  }
  
  
  rprob = posterior(mod)[,2:(states+1)]
  rprob.new = rprob*0
  write.csv( rprob , paste0(getwd(),"/results/MixtureModelsCrowd/resprobunmatchedDMM",id,".csv"),row.names = F)
  
  
  
  write.csv(res, paste0(getwd(),"/results/MixtureModelsCrowd/resunmatchedDMM",id,".csv"),row.names = F)
  
  res = resmap[res]
  
  for(i in 1:length(resmap))
  {
    idss = which(resmap == i)
    if(length(idss)==1)
      rprob.new[,i] = rprob[,idss]
    if(length(idss)>1)
      rprob.new[,i] = rowSums(rprob[,idss]) 
    
  }
  
  
  write.csv(cbind(classes[X[,rely]],classes[res]),paste0(getwd(),"/results/MixtureModelsCrowd/checkDMM",id,".csv"))
  
  rprob.new = rprob.new[,c(states,1:(states-1))]
  
  names(rprob.new) = paste0("T",0:(states-1))
  
  write.csv(rprob.new, paste0(getwd(),"/results/MixtureModelsCrowd/resprobDMM",id,".csv"),row.names = F)
  
  res[res==states]=0
  
  write.csv(res, paste0(getwd(),"/results/MixtureModelsCrowd/resDMM",id,".csv"),row.names = F)
  
  #clean up the thread
  rm(X,freq, instart,mod, model,rprob,rprob.new)
  gc()
  return(res)
}




#a function implementing confusion matrix Dependent Mixtures of Multinomoals
DCM = function(id)
{
  
  if (!dir.exists(paste0(getwd(),"/results/MixtureModelsCrowdHM/"))){
    dir.create(paste0(getwd(),"/results/MixtureModelsCrowdHM/"),recursive = T)
  }
  if(file.exists(paste0(getwd(),"/results/MixtureModelsCrowdHM/resprobunmatched",id,".csv")))
  {
    print("A")
    return("A")
  }
  print(id)
  X = read.table(input,skip = size*(id-1),nrows = size,sep = ",")#[-1]
  X[X==0]=states
  nsours = dim(X)[2]
  matches = read.table(match.lab,skip = size*(id-1),nrows = size,sep = ",")
  matches[matches==0]=77
  
  instart = rep(0.0001,states)
  set.seed(id)
  resp = NULL
  ns = NULL
  for(i in 1:nsours)
  {
    
    if(length(levels(as.factor(X[,i])))==1)
    {  
      ns = c(ns,i)
      next
    }
    
    if(i==rely)
    {
      freq = table(X[,i])
      instart[as.integer(names(freq))]  =  instart[as.integer(names(freq))]+ freq
    }
  }
  
  X[,ns]=NULL
  
  rely = which(names(X)=="V12")
  if(dim(X)[2]==0)
    X$V1 = c(rep(states,size-1),1)
  
  nsours = dim(X)[2]
  
  
  instart = instart/sum(instart)
  
  
  model = depmixS4::depmix(response = lapply(FUN = function(X) as.formula(paste0(X,"~1")),X = names(X)), family = lapply(function(X)multinomial(),X=1:nsours),instart = instart,nstates = states,data = X)
  mod = fit(model,verbose =T,emcontrol=em.control(maxit = 25))
  res = posterior(mod)$state 
  
  if(length(rely)==0)
  {
    X$rely = states
    rely = dim(X)[2]
  }
  resmap = array(0,states)
  
  for(resv in 1:states)
  {
    tmpres = res == resv
    smax = -10000000
    tmplab = states
    for(ann in 1:states)
    {
      tmpann =  X[,rely] == ann
      tmpsum = (cor(tmpres, tmpann))
      if(is.na(tmpsum))
        next
      if(tmpsum>=smax)
      {
        smax = tmpsum
        tmplab = ann
      }
      if(smax<0.1)
      {  
        tmplab = states
      }
    }
    resmap[resv] = tmplab
  }
  
  
  rprob = posterior(mod)[,2:(states+1)]
  rprob.new = rprob*0
  write.csv( rprob , paste0(getwd(),"/results/MixtureModelsCrowdHM/resprobunmatched",id,".csv"),row.names = F)
  write.csv(res, paste0(getwd(),"/results/MixtureModelsCrowdHM/resunmatched",id,".csv"),row.names = F)
  
  res = resmap[res]
  for(i in 1:length(resmap))
  {
    idss = which(resmap == i)
    if(length(idss)==1)
      rprob.new[,i] = rprob[,idss]
    if(length(idss)>1)
      rprob.new[,i] = rowSums(rprob[,idss]) 
    
  }
  
  
  write.csv(cbind(classes[X[,rely]],classes[res]),paste0(getwd(),"/results/MixtureModelsCrowdHM/check",id,".csv"))
  rprob.new = rprob.new[,c(states,1:(states-1))]
  names(rprob.new) = paste0("T",0:(states-1))
  write.csv(rprob.new, paste0(getwd(),"/results/MixtureModelsCrowdHM/resprob",id,".csv"),row.names = F)
  res[res==states]=0
  write.csv(res, paste0(getwd(),"/results/MixtureModelsCrowdHM/res",id,".csv"),row.names = F)
  
  
  res = posterior(mod)$state 
  
  X[,rely] = matches$V1
  resmap = array(0,states)
  
  for(resv in 1:states)
  {
    tmpres = res == resv
    smax = -10000000
    tmplab = states
    for(ann in 1:states)
    {
      tmpann =  X[,rely] == ann
      tmpsum = (cor(tmpres, tmpann))
      if(is.na(tmpsum))
        next
      if(tmpsum>=smax)
      {
        smax = tmpsum
        tmplab = ann
      }
      if(smax<0.1)
      {  
        tmplab = states
      }
    }
    resmap[resv] = tmplab
  }
  
  
  rprob = posterior(mod)[,2:(states+1)]
  rprob.new = rprob*0
  write.csv( rprob , paste0(getwd(),"/results/MixtureModelsCrowdHM/resprobunmatchedHMMV",id,".csv"),row.names = F)
  
  
  
  write.csv(res, paste0(getwd(),"/results/MixtureModelsCrowdHM/resunmatchedHMMV",id,".csv"),row.names = F)
  
  res = resmap[res]
  
  for(i in 1:length(resmap))
  {
    idss = which(resmap == i)
    if(length(idss)==1)
      rprob.new[,i] = rprob[,idss]
    if(length(idss)>1)
      rprob.new[,i] = rowSums(rprob[,idss]) 
    
  }
  
  
  write.csv(cbind(classes[X[,rely]],classes[res]),paste0(getwd(),"/results/MixtureModelsCrowdHM/checkHMMV",id,".csv"))
  
  rprob.new = rprob.new[,c(states,1:(states-1))]
  
  names(rprob.new) = paste0("T",0:(states-1))
  
  write.csv(rprob.new, paste0(getwd(),"/results/MixtureModelsCrowdHM/resprobHMMV",id,".csv"),row.names = F)
  
  res[res==states]=0
  
  write.csv(res, paste0(getwd(),"/results/MixtureModelsCrowdHM/resHMMV",id,".csv"),row.names = F)
  
  
  res = posterior(mod)$state 
  
  X[,rely] = matches$V2
  resmap = array(0,states)
  
  for(resv in 1:states)
  {
    tmpres = res == resv
    smax = -10000000
    tmplab = states
    for(ann in 1:states)
    {
      tmpann =  X[,rely] == ann
      tmpsum = (cor(tmpres, tmpann))
      if(is.na(tmpsum))
        next
      if(tmpsum>=smax)
      {
        smax = tmpsum
        tmplab = ann
      }
      if(smax<0.1)
      {  
        tmplab = states
      }
    }
    resmap[resv] = tmplab
  }
  
  
  rprob = posterior(mod)[,2:(states+1)]
  rprob.new = rprob*0
  write.csv( rprob , paste0(getwd(),"/results/MixtureModelsCrowdHM/resprobunmatchedHMDMM",id,".csv"),row.names = F)
  
  
  
  write.csv(res, paste0(getwd(),"/results/MixtureModelsCrowdHM/resunmatchedHMDMM",id,".csv"),row.names = F)
  
  res = resmap[res]
  
  for(i in 1:length(resmap))
  {
    idss = which(resmap == i)
    if(length(idss)==1)
      rprob.new[,i] = rprob[,idss]
    if(length(idss)>1)
      rprob.new[,i] = rowSums(rprob[,idss]) 
    
  }
  
  
  write.csv(cbind(classes[X[,rely]],classes[res]),paste0(getwd(),"/results/MixtureModelsCrowdHM/checkHMDMM",id,".csv"))
  
  rprob.new = rprob.new[,c(states,1:(states-1))]
  
  names(rprob.new) = paste0("T",0:(states-1))
  
  write.csv(rprob.new, paste0(getwd(),"/results/MixtureModelsCrowdHM/resprobHMDMM",id,".csv"),row.names = F)
  
  res[res==states]=0
  
  write.csv(res, paste0(getwd(),"/results/MixtureModelsCrowdHM/resHMDMM",id,".csv"),row.names = F)
  
  #clean up the thread
  rm(X,freq, instart,mod, model,rprob,rprob.new)
  gc()
  return(res)
}

#a function implementing sequantial confusion matrix Mixtures of Multinomoals
SEQ = function(id)
{
  
  
  if (!dir.exists(paste0(getwd(),"/results/ARMixtureModelsCrowd/"))){
    dir.create(paste0(getwd(),"/results/ARMixtureModelsCrowd/"),recursive = T)
  }
  if(file.exists(paste0(getwd(),"/results/ARMixtureModelsCrowd/res",id,".csv")))
  {
    print("A")
    return("A")
  }
  print(id)
  X = read.table(input,skip = size*(id-1),nrows = size,sep = ",")#[-1]
  X[X==0]=states
  nsours = dim(X)[2]
  
  matches = read.table(match.lab,skip = size*(id-1),nrows = size,sep = ",")
  matches[matches==0]=77
  
  instart = rep(0.0001,states)
  set.seed(id)
  resp = NULL
  ns = NULL
  for(i in 1:nsours)
  {
    
    if(length(levels(as.factor(X[,i])))==1)
    {  
      ns = c(ns,i)
      next
    }
    
    if(i==rely)
    {
      freq = table(X[,i])
      instart[as.integer(names(freq))]  =  instart[as.integer(names(freq))]+ freq
    }
    
    X[[paste0("L",names(X)[i])]]=as.integer(c(states,(X[1:(size-1),i]))==X[,i])
    
  }
  
  X[,ns]=NULL
  
  
  rely = which(names(X)=="V12")
  if(dim(X)[2]==0)
    X$V1 = c(rep(states,size-1),1)
  
  nsours = dim(X)[2]/2
  
  
  instart = instart/sum(instart)
  model = depmixS4::mix(response = lapply(FUN = function(X) as.formula(paste0(X,"~1","+L",X)),X = names(X)[1:nsours]), family = lapply(function(X)multinomial(),X=1:nsours),instart = instart,nstates = states,data = X)
  
  mod = fit(model,verbose =T,emcontrol=em.control(maxit = 25))
  
  res = posterior(mod)$state
  if(length(rely)==0)
  {
    X$rely = states
    rely = dim(X)[2]
  }
  resmap = array(0,states)
  
  for(resv in 1:states)
  {
    tmpres = res == resv
    smax = -10000000
    tmplab = states
    for(ann in 1:states)
    {
      tmpann =  X[,rely] == ann
      tmpsum = (cor(tmpres, tmpann))
      if(is.na(tmpsum))
        next
      if(tmpsum>=smax)
      {
        smax = tmpsum
        tmplab = ann
      }
      if(smax<0.1)
      {  
        tmplab = states
      }
    }
    resmap[resv] = tmplab
  }
  
  
  rprob = posterior(mod)[,2:(states+1)]
  rprob.new = rprob*0
  write.csv( rprob , paste0(getwd(),"/results/ARMixtureModelsCrowd/resprobunmatched",id,".csv"),row.names = F)
  write.csv(res, paste0(getwd(),"/results/ARMixtureModelsCrowd/resunmatched",id,".csv"),row.names = F)
  
  res = resmap[res]
  for(i in 1:length(resmap))
  {
    idss = which(resmap == i)
    if(length(idss)==1)
      rprob.new[,i] = rprob[,idss]
    if(length(idss)>1)
      rprob.new[,i] = rowSums(rprob[,idss]) 
    
  }
  
  
  write.csv(cbind(classes[X[,rely]],classes[res]),paste0(getwd(),"/results/ARMixtureModelsCrowd/check",id,".csv"))
  rprob.new = rprob.new[,c(states,1:(states-1))]
  names(rprob.new) = paste0("T",0:(states-1))
  write.csv(rprob.new, paste0(getwd(),"/results/ARMixtureModelsCrowd/resprob",id,".csv"),row.names = F)
  res[res==states]=0
  write.csv(res, paste0(getwd(),"/results/ARMixtureModelsCrowd/res",id,".csv"),row.names = F)
  
  res = posterior(mod)$state 
  
  X[,rely] = matches$V1
  resmap = array(0,states)
  
  for(resv in 1:states)
  {
    tmpres = res == resv
    smax = -10000000
    tmplab = states
    for(ann in 1:states)
    {
      tmpann =  X[,rely] == ann
      tmpsum = (cor(tmpres, tmpann))
      if(is.na(tmpsum))
        next
      if(tmpsum>=smax)
      {
        smax = tmpsum
        tmplab = ann
      }
      if(smax<0.1)
      {  
        tmplab = states
      }
    }
    resmap[resv] = tmplab
  }
  
  
  rprob = posterior(mod)[,2:(states+1)]
  rprob.new = rprob*0
  write.csv( rprob , paste0(getwd(),"/results/ARMixtureModelsCrowd/resprobunmatchedARMV",id,".csv"),row.names = F)
  
  
  
  write.csv(res, paste0(getwd(),"/results/ARMixtureModelsCrowd/resunmatchedARMV",id,".csv"),row.names = F)
  
  res = resmap[res]
  
  for(i in 1:length(resmap))
  {
    idss = which(resmap == i)
    if(length(idss)==1)
      rprob.new[,i] = rprob[,idss]
    if(length(idss)>1)
      rprob.new[,i] = rowSums(rprob[,idss]) 
    
  }
  
  
  write.csv(cbind(classes[X[,rely]],classes[res]),paste0(getwd(),"/results/ARMixtureModelsCrowd/checkARMV",id,".csv"))
  
  rprob.new = rprob.new[,c(states,1:(states-1))]
  
  names(rprob.new) = paste0("T",0:(states-1))
  
  write.csv(rprob.new, paste0(getwd(),"/results/ARMixtureModelsCrowd/resprobARMV",id,".csv"),row.names = F)
  
  res[res==states]=0
  
  write.csv(res, paste0(getwd(),"/results/ARMixtureModelsCrowd/resARMV",id,".csv"),row.names = F)
  
  
  res = posterior(mod)$state 
  
  X[,rely] = matches$V2
  resmap = array(0,states)
  
  for(resv in 1:states)
  {
    tmpres = res == resv
    smax = -10000000
    tmplab = states
    for(ann in 1:states)
    {
      tmpann =  X[,rely] == ann
      tmpsum = (cor(tmpres, tmpann))
      if(is.na(tmpsum))
        next
      if(tmpsum>=smax)
      {
        smax = tmpsum
        tmplab = ann
      }
      if(smax<0.1)
      {  
        tmplab = states
      }
    }
    resmap[resv] = tmplab
  }
  
  
  rprob = posterior(mod)[,2:(states+1)]
  rprob.new = rprob*0
  write.csv( rprob , paste0(getwd(),"/results/ARMixtureModelsCrowd/resprobunmatchedARDMM",id,".csv"),row.names = F)
  
  
  
  write.csv(res, paste0(getwd(),"/results/ARMixtureModelsCrowd/resunmatchedARDMM",id,".csv"),row.names = F)
  
  res = resmap[res]
  
  for(i in 1:length(resmap))
  {
    idss = which(resmap == i)
    if(length(idss)==1)
      rprob.new[,i] = rprob[,idss]
    if(length(idss)>1)
      rprob.new[,i] = rowSums(rprob[,idss]) 
    
  }
  
  
  write.csv(cbind(classes[X[,rely]],classes[res]),paste0(getwd(),"/results/ARMixtureModelsCrowd/checkARDMM",id,".csv"))
  
  rprob.new = rprob.new[,c(states,1:(states-1))]
  
  names(rprob.new) = paste0("T",0:(states-1))
  
  write.csv(rprob.new, paste0(getwd(),"/results/ARMixtureModelsCrowd/resprobARDMM",id,".csv"),row.names = F)
  
  res[res==states]=0
  
  write.csv(res, paste0(getwd(),"/results/ARMixtureModelsCrowd/resARDMM",id,".csv"),row.names = F)
  
  
  #clean up the thread
  rm(X,freq, instart,mod, model,rprob,rprob.new)
  gc()
  return(res)
}


#a function implementing confusion vector Mixtures of Multinomoals
CV = function(id)
{
  
  if (!dir.exists(paste0(getwd(),"/results/CVMixtureModelsCrowd/"))){
    dir.create(paste0(getwd(),"/results/CVMixtureModelsCrowd/"),recursive = T)
  }
  if(file.exists(paste0(getwd(),"/results/CVMixtureModelsCrowd/res",id,".csv")))
  {
    print("A")
    return("A")
  }
  print(id)
  X = read.table(input,skip = size*(id-1),nrows = size,sep = ",")#[-1]
  X[X==0]=states
  nsours = dim(X)[2]
  matches = read.table(match.lab,skip = size*(id-1),nrows = size,sep = ",")
  matches[matches==0]=77
  instart = rep(0.0001,states)
  set.seed(id+1000)
  resp = NULL
  ns = NULL
  for(i in 1:nsours)
  {
    
    if(length(levels(as.factor(X[,i])))==1)
    {  
      ns = c(ns,i)
      next
    }
    
    if(i==rely)
    {
      freq = table(X[,i])
      instart[as.integer(names(freq))]  =  instart[as.integer(names(freq))]+ freq
    }
  }
  
  X[,ns]=NULL
  
  rely = which(names(X)=="V12")
  if(dim(X)[2]==0)
    X$V1 = c(rep(states,size-1),1)
  nsours = dim(X)[2]
  
  model = depmixS4::mix(response = lapply(FUN = function(X) as.formula(paste0(X,"~1")),X = names(X)), family = lapply(function(X)multinomial(),X=1:nsours),instart = instart,nstates = states,data = X)
  
  #specify the CV constraints
  par = getpars(model)
  par = as.integer(par)
  lp = length(par)
  k = states +1
  for(j in 1:(states))
  {
    for(i in 1:nsours)
    {
      tlen = length(levels(as.factor(X[,i])))
      if(k-tlen>lp)
        break
      pi = sum(X[,i]==j)/size
      if(pi==0)
      {
        par[(k):(k+tlen - 1)] = k+tlen
      }else
      {
        thi = which(levels(as.factor(X[,i])) == j)-1
        par[(k):(k+tlen-1)] = k+tlen
        par[k+thi] = k+thi
        #print(c(j,  par[k+thi], par[(k):(k+tlen-1)]))
      }
      k = k + tlen
    }
  }
  
  is.fixed = par[1:lp]
  is.fixed[1:states] = 1
  mod = fit(model,verbose =T,emcontrol=em.control(maxit = 25),equals = is.fixed)
  
  res = posterior(mod)$state 
  
  res = posterior(mod)$state
  if(length(rely)==0)
  {
    X$rely = states
    rely = dim(X)[2]
  }
  resmap = array(0,states)
  
  for(resv in 1:states)
  {
    tmpres = res == resv
    smax = -10000000
    tmplab = states
    for(ann in 1:states)
    {
      tmpann =  X[,rely] == ann
      tmpsum = (cor(tmpres, tmpann))
      if(is.na(tmpsum))
        next
      if(tmpsum>=smax)
      {
        smax = tmpsum
        tmplab = ann
      }
      if(smax<0.1)
      {  
        tmplab = states
      }
    }
    resmap[resv] = tmplab
  }
  
  
  rprob = posterior(mod)[,2:(states+1)]
  rprob.new = rprob*0
  write.csv( rprob , paste0(getwd(),"/results/CVMixtureModelsCrowd/resprobunmatched",id,".csv"),row.names = F)
  write.csv(res, paste0(getwd(),"/results/CVMixtureModelsCrowd/resunmatched",id,".csv"),row.names = F)
  
  res = resmap[res]
  
  for(i in 1:length(resmap))
  {
    idss = which(resmap == i)
    if(length(idss)==1)
      rprob.new[,i] = rprob[,idss]
    if(length(idss)>1)
      rprob.new[,i] = rowSums(rprob[,idss]) 
    
  }
  
  
  write.csv(cbind(classes[X[,rely]],classes[res]),paste0(getwd(),"/results/CVMixtureModelsCrowd/check",id,".csv"))
  rprob.new = rprob.new[,c(states,1:(states-1))]
  names(rprob.new) = paste0("T",0:(states-1))
  write.csv(rprob.new, paste0(getwd(),"/results/CVMixtureModelsCrowd/resprob",id,".csv"),row.names = F)
  res[res==states]=0
  write.csv(res, paste0(getwd(),"/results/CVMixtureModelsCrowd/res",id,".csv"),row.names = F)
  
  res = posterior(mod)$state 
  
  X[,rely] = matches$V1
  resmap = array(0,states)
  
  for(resv in 1:states)
  {
    tmpres = res == resv
    smax = -10000000
    tmplab = states
    for(ann in 1:states)
    {
      tmpann =  X[,rely] == ann
      tmpsum = (cor(tmpres, tmpann))
      if(is.na(tmpsum))
        next
      if(tmpsum>=smax)
      {
        smax = tmpsum
        tmplab = ann
      }
      if(smax<0.1)
      {  
        tmplab = states
      }
    }
    resmap[resv] = tmplab
  }
  
  
  rprob = posterior(mod)[,2:(states+1)]
  rprob.new = rprob*0
  write.csv( rprob , paste0(getwd(),"/results/CVMixtureModelsCrowd/resprobunmatchedMV",id,".csv"),row.names = F)
  
  
  
  write.csv(res, paste0(getwd(),"/results/CVMixtureModelsCrowd/resunmatchedMV",id,".csv"),row.names = F)
  
  res = resmap[res]
  
  for(i in 1:length(resmap))
  {
    idss = which(resmap == i)
    if(length(idss)==1)
      rprob.new[,i] = rprob[,idss]
    if(length(idss)>1)
      rprob.new[,i] = rowSums(rprob[,idss]) 
    
  }
  
  
  write.csv(cbind(classes[X[,rely]],classes[res]),paste0(getwd(),"/results/CVMixtureModelsCrowd/checkMV",id,".csv"))
  
  rprob.new = rprob.new[,c(states,1:(states-1))]
  
  names(rprob.new) = paste0("T",0:(states-1))
  
  write.csv(rprob.new, paste0(getwd(),"/results/CVMixtureModelsCrowd/resprobMV",id,".csv"),row.names = F)
  
  res[res==states]=0
  
  write.csv(res, paste0(getwd(),"/results/CVMixtureModelsCrowd/resMV",id,".csv"),row.names = F)
  
  
  res = posterior(mod)$state 
  
  X[,rely] = matches$V2
  resmap = array(0,states)
  
  for(resv in 1:states)
  {
    tmpres = res == resv
    smax = -10000000
    tmplab = states
    for(ann in 1:states)
    {
      tmpann =  X[,rely] == ann
      tmpsum = (cor(tmpres, tmpann))
      if(is.na(tmpsum))
        next
      if(tmpsum>=smax)
      {
        smax = tmpsum
        tmplab = ann
      }
      if(smax<0.1)
      {  
        tmplab = states
      }
    }
    resmap[resv] = tmplab
  }
  
  
  rprob = posterior(mod)[,2:(states+1)]
  rprob.new = rprob*0
  write.csv( rprob , paste0(getwd(),"/results/CVMixtureModelsCrowd/resprobunmatchedDMM",id,".csv"),row.names = F)
  
  
  
  write.csv(res, paste0(getwd(),"/results/CVMixtureModelsCrowd/resunmatchedDMM",id,".csv"),row.names = F)
  
  res = resmap[res]
  
  for(i in 1:length(resmap))
  {
    idss = which(resmap == i)
    if(length(idss)==1)
      rprob.new[,i] = rprob[,idss]
    if(length(idss)>1)
      rprob.new[,i] = rowSums(rprob[,idss]) 
    
  }
  
  
  write.csv(cbind(classes[X[,rely]],classes[res]),paste0(getwd(),"/results/CVMixtureModelsCrowd/checkDMM",id,".csv"))
  
  rprob.new = rprob.new[,c(states,1:(states-1))]
  
  names(rprob.new) = paste0("T",0:(states-1))
  
  write.csv(rprob.new, paste0(getwd(),"/results/CVMixtureModelsCrowd/resprobDMM",id,".csv"),row.names = F)
  
  res[res==states]=0
  
  write.csv(res, paste0(getwd(),"/results/CVMixtureModelsCrowd/resDMM",id,".csv"),row.names = F)
  
  
  #clean up the thread
  rm(X,freq, instart,mod, model,rprob,rprob.new)
  gc()
  return(res)
}


#a function implementing accuracy model Mixtures of Multinomoals
ACC= function(id)
{
  
  
  
  if (!dir.exists(paste0(getwd(),"/results/ACCMixtureModelsCrowd/"))){
    dir.create(paste0(getwd(),"/results/ACCMixtureModelsCrowd/"),recursive = T)
  }
  
  if(file.exists(paste0(getwd(),"/results/ACCMixtureModelsCrowd/res",id,".csv")))
  {
    print("A")
    return("A")
  }
  print(id)
  X = read.table(input,skip = size*(id-1),nrows = size,sep = ",")#[-1]
  X[X==0]=states
  nsours = dim(X)[2]
  matches = read.table(match.lab,skip = size*(id-1),nrows = size,sep = ",")
  matches[matches==0]=77
  instart = rep(0.0001,states)
  set.seed(id)
  resp = NULL
  ns = NULL
  for(i in 1:nsours)
  {
    
    if(length(levels(as.factor(X[,i])))==1)
    {  
      ns = c(ns,i)
      next
    }
    
    if(i==rely)
    {
      freq = table(X[,i])
      instart[as.integer(names(freq))]  =  instart[as.integer(names(freq))]+ freq
    }
  }
  
  X[,ns]=NULL
  
  rely = which(names(X)=="V12")
  
  nsours = dim(X)[2]
  if(dim(X)[2]==0)
    X$V1 = c(rep(states,size-1),1)
  
  model = depmixS4::mix(response = lapply(FUN = function(X) as.formula(paste0(X,"~1")),X = names(X)), family = lapply(function(X)multinomial(),X=1:nsours),instart = instart,nstates = states,data = X)
  
  
  #specify the accuracy model constraints
  par = getpars(model)
  par = as.integer(par)
  lp = length(par)
  k = states +1
  for(j in 1:(states))
  {
    for(i in 1:nsours)
    {
      
      
      tlen = length(levels(as.factor(X[,i])))
      if(k-tlen>lp)
        break
      pi = sum(X[,i]==j)/size
      if(pi==0)
      {
        par[(k):(k+tlen - 1)] = k+tlen
      }else
      {
        thi = which(levels(as.factor(X[,i])) == j)-1
        par[(k):(k+tlen-1)] = k+thi
        par[k+thi] =  tlen*(i^2)
      }
      
      k = k + tlen
      
    }
  }
  
  
  is.fixed = par[1:lp]
  is.fixed[1:states] = 1
  mod = fit(model,verbose =T,emcontrol=em.control(maxit = 25),equals = is.fixed)
  
  res = posterior(mod)$state 
  if(length(rely)==0)
  {
    X$rely = states
    rely = dim(X)[2]
  }
  resmap = array(0,states)
  
  for(resv in 1:states)
  {
    tmpres = res == resv
    smax = -10000000
    tmplab = resv
    for(ann in 1:states)
    {
      tmpann =  X[,rely] == ann
      tmpsum = (cor(tmpres, tmpann))
      if(is.na(tmpsum))
        next
      if(tmpsum>=smax)
      {
        smax = tmpsum
        tmplab = ann
      }
    }
    resmap[resv] = tmplab
  }
  
  
  rprob = posterior(mod)[,2:(states+1)]
  rprob.new = rprob*0
  write.csv( rprob , paste0(getwd(),"/results/ACCMixtureModelsCrowd/resprobunmatched",id,".csv"),row.names = F)
  
  
  write.csv(res, paste0(getwd(),"/results/ACCMixtureModelsCrowd/resunmatched",id,".csv"),row.names = F)
  res = resmap[res]
  
  
  for(i in 1:length(resmap))
  {
    idss = which(resmap == i)
    if(length(idss)==1)
      rprob.new[,i] = rprob[,idss]
    if(length(idss)>1)
      rprob.new[,i] = rowSums(rprob[,idss]) 
    
  }
  
  
  write.csv(cbind(classes[X[,rely]],classes[res]),paste0(getwd(),"/results/ACCMixtureModelsCrowd/check",id,".csv"))
  rprob.new = rprob.new[,c(states,1:(states-1))]
  names(rprob.new) = paste0("T",0:(states-1))
  write.csv(rprob.new, paste0(getwd(),"/results/ACCMixtureModelsCrowd/resprob",id,".csv"),row.names = F)
  res[res==states]=0
  write.csv(res, paste0(getwd(),"/results/ACCMixtureModelsCrowd/res",id,".csv"),row.names = F)
  
  res = posterior(mod)$state 
  
  X[,rely] = matches$V1
  resmap = array(0,states)
  
  for(resv in 1:states)
  {
    tmpres = res == resv
    smax = -10000000
    tmplab = states
    for(ann in 1:states)
    {
      tmpann =  X[,rely] == ann
      tmpsum = (cor(tmpres, tmpann))
      if(is.na(tmpsum))
        next
      if(tmpsum>=smax)
      {
        smax = tmpsum
        tmplab = ann
      }
      if(smax<0.1)
      {  
        tmplab = states
      }
    }
    resmap[resv] = tmplab
  }
  
  
  rprob = posterior(mod)[,2:(states+1)]
  rprob.new = rprob*0
  write.csv( rprob , paste0(getwd(),"/results/ACCMixtureModelsCrowd/resprobunmatchedMV",id,".csv"),row.names = F)
  
  
  
  write.csv(res, paste0(getwd(),"/results/ACCMixtureModelsCrowd/resunmatchedMV",id,".csv"),row.names = F)
  
  res = resmap[res]
  
  for(i in 1:length(resmap))
  {
    idss = which(resmap == i)
    if(length(idss)==1)
      rprob.new[,i] = rprob[,idss]
    if(length(idss)>1)
      rprob.new[,i] = rowSums(rprob[,idss]) 
    
  }
  
  
  write.csv(cbind(classes[X[,rely]],classes[res]),paste0(getwd(),"/results/ACCMixtureModelsCrowd/checkMV",id,".csv"))
  
  rprob.new = rprob.new[,c(states,1:(states-1))]
  
  names(rprob.new) = paste0("T",0:(states-1))
  
  write.csv(rprob.new, paste0(getwd(),"/results/ACCMixtureModelsCrowd/resprobMV",id,".csv"),row.names = F)
  
  res[res==states]=0
  
  write.csv(res, paste0(getwd(),"/results/ACCMixtureModelsCrowd/resMV",id,".csv"),row.names = F)
  
  
  res = posterior(mod)$state 
  
  X[,rely] = matches$V2
  resmap = array(0,states)
  
  for(resv in 1:states)
  {
    tmpres = res == resv
    smax = -10000000
    tmplab = states
    for(ann in 1:states)
    {
      tmpann =  X[,rely] == ann
      tmpsum = (cor(tmpres, tmpann))
      if(is.na(tmpsum))
        next
      if(tmpsum>=smax)
      {
        smax = tmpsum
        tmplab = ann
      }
      if(smax<0.1)
      {  
        tmplab = states
      }
    }
    resmap[resv] = tmplab
  }
  
  
  rprob = posterior(mod)[,2:(states+1)]
  rprob.new = rprob*0
  write.csv( rprob , paste0(getwd(),"/results/ACCMixtureModelsCrowd/resprobunmatchedDMM",id,".csv"),row.names = F)
  
  
  
  write.csv(res, paste0(getwd(),"/results/ACCMixtureModelsCrowd/resunmatchedDMM",id,".csv"),row.names = F)
  
  res = resmap[res]
  
  for(i in 1:length(resmap))
  {
    idss = which(resmap == i)
    if(length(idss)==1)
      rprob.new[,i] = rprob[,idss]
    if(length(idss)>1)
      rprob.new[,i] = rowSums(rprob[,idss]) 
    
  }
  
  
  write.csv(cbind(classes[X[,rely]],classes[res]),paste0(getwd(),"/results/ACCMixtureModelsCrowd/checkDMM",id,".csv"))
  
  rprob.new = rprob.new[,c(states,1:(states-1))]
  
  names(rprob.new) = paste0("T",0:(states-1))
  
  write.csv(rprob.new, paste0(getwd(),"/results/ACCMixtureModelsCrowd/resprobDMM",id,".csv"),row.names = F)
  
  res[res==states]=0
  
  write.csv(res, paste0(getwd(),"/results/ACCMixtureModelsCrowd/resDMM",id,".csv"),row.names = F)
  
  
  #clean up the thread
  rm(X,freq, instart,mod, model,rprob,rprob.new)
  gc()
  return(res)
}


#run the models
results = unlist(mclapply(X=sample.int(nseq,nseq,replace = F),FUN = CM,mc.preschedule = T,mc.cleanup = T,mc.cores = n.cores))
print(results)
results = unlist(mclapply(X=sample.int(nseq,nseq,replace = F),FUN = DCM,mc.preschedule = T,mc.cleanup = T,mc.cores = n.cores))
print(results)
results = unlist(mclapply(X=sample.int(nseq,nseq,replace = F),FUN = SEQ,mc.preschedule = T,mc.cleanup = T,mc.cores = n.cores))
print(results)
results = unlist(mclapply(X=sample.int(nseq,nseq,replace = F),FUN = ACC,mc.preschedule = T,mc.cleanup = T,mc.cores = n.cores))
print(results)
results = unlist(mclapply(X=sample.int(nseq,nseq,replace = F),FUN = CV,mc.preschedule = T,mc.cleanup = T,mc.cores = n.cores))
print(results)





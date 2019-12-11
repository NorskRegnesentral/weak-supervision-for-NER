#load the library for inference on mixture models
library(depmixS4)
#load the library for parallel computing
library(parallel)

#a list of sourses
sourses = c('BTC', 'BTC+c', 'SEC', 'SEC+c', 'company_cased', 'company_type_detector', 'company_uncased', 'compound_detector', 
            'conll2003', 'conll2003+c', 'core_web_md', 'core_web_md+c', 'crunchbase_cased', 'crunchbase_uncased', 'date_detector', 
            'doc_history', 'doc_majority_cased', 'doc_majority_uncased', 'full_name_detector', 'geo_cased', 'geo_uncased', 
            'infrequent_compound_detector', 'infrequent_nnp_detector', 'infrequent_proper2_detector', 'infrequent_proper_detector', 
            'legal_detector', 'misc_detector', 'money_detector', 'multitoken_company_cased', 'multitoken_company_uncased', 
            'multitoken_crunchbase_cased', 'multitoken_crunchbase_uncased', 'multitoken_geo_cased', 'multitoken_geo_uncased', 
            'multitoken_product_cased', 'multitoken_product_uncased', 'multitoken_wiki_cased', 'multitoken_wiki_small_cased', 
            'multitoken_wiki_small_uncased', 'multitoken_wiki_uncased', 'nnp_detector', 'number_detector', 'product_cased', 
            'product_uncased', 'proper2_detector', 'proper_detector', 'snips', 'time_detector', 'wiki_cased', 'wiki_small_cased', 
            'wiki_small_uncased', 'wiki_uncased')

#a list of classes
classes = c('B-CARDINAL', 'I-CARDINAL', 'L-CARDINAL', 'U-CARDINAL', 'B-COMPANY', 'I-COMPANY', 'L-COMPANY', 'U-COMPANY', 'B-DATE',
            'I-DATE', 'L-DATE', 'U-DATE', 'B-EVENT', 'I-EVENT', 'L-EVENT', 'U-EVENT', 'B-FAC', 'I-FAC', 'L-FAC', 'U-FAC', 'B-GPE', 
            'I-GPE', 'L-GPE', 'U-GPE', 'B-LANGUAGE', 'I-LANGUAGE', 'L-LANGUAGE', 'U-LANGUAGE', 'B-LAW', 'I-LAW', 'L-LAW', 'U-LAW', 
            'B-LOC', 'I-LOC', 'L-LOC', 'U-LOC', 'B-MONEY', 'I-MONEY', 'L-MONEY', 'U-MONEY', 'B-NORP', 'I-NORP', 'L-NORP', 'U-NORP', 
            'B-ORDINAL', 'I-ORDINAL', 'L-ORDINAL', 'U-ORDINAL', 'B-ORG', 'I-ORG', 'L-ORG', 'U-ORG', 'B-PERCENT', 'I-PERCENT', 
            'L-PERCENT', 'U-PERCENT', 'B-PERSON', 'I-PERSON', 'L-PERSON', 'U-PERSON', 'B-PRODUCT', 'I-PRODUCT', 'L-PRODUCT', 
            'U-PRODUCT', 'B-QUANTITY', 'I-QUANTITY', 'L-QUANTITY', 'U-QUANTITY', 'B-TIME', 'I-TIME', 'L-TIME', 'U-TIME', 
            'B-WORK_OF_ART', 'I-WORK_OF_ART', 'L-WORK_OF_ART', 'U-WORK_OF_ART','O')


#specify the working directory and read in the files
setwd("weak_supervision")
input = "/weak_supervision/data/conll2003.csv"
tokens = "/conll2003_tokens.csv"

#read in the parameters of the model
param = read.csv("parammix.csv",header = T)

size = param[1,2] #batch size
states = param[2,2] #number of latent states of the Mixtures of Multinomials
nseq = param[3,2] #number of batches in the whole training data
rely = param[4,2] #id of the most reliable sourse


#a function implementing confusion matrix Mixtures of Multinomoals
CM = function(id)
{
  
  if (!dir.exists(paste0(getwd(),"/results/MixtureModelsConll/"))){
    dir.create(paste0(getwd(),"/results/MixtureModelsConll/"),recursive = T)
  }
  if(file.exists(paste0(getwd(),"/results/MixtureModelsConll/resprobunmatched",id,".csv")))
  {
    print("A")
    return("A")
  }
  print(id)
  X = read.table(input,skip = size*(id-1),nrows = size,sep = ",")#[-1]
  X[X==0]=states
  nsours = dim(X)[2]
  
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
  write.csv( rprob , paste0(getwd(),"/results/MixtureModelsConll/resprobunmatched",id,".csv"),row.names = F)
  write.csv(res, paste0(getwd(),"/results/MixtureModelsConll/resunmatched",id,".csv"),row.names = F)
  
  res = resmap[res]
  for(i in 1:length(resmap))
  {
    idss = which(resmap == i)
    if(length(idss)==1)
      rprob.new[,i] = rprob[,idss]
    if(length(idss)>1)
      rprob.new[,i] = rowSums(rprob[,idss]) 
    
  }
  
  
  write.csv(cbind(classes[X[,rely]],classes[res]),paste0(getwd(),"/results/MixtureModelsConll/check",id,".csv"))
  rprob.new = rprob.new[,c(states,1:(states-1))]
  names(rprob.new) = paste0("T",0:(states-1))
  write.csv(rprob.new, paste0(getwd(),"/results/MixtureModelsConll/resprob",id,".csv"),row.names = F)
  res[res==states]=0
  write.csv(res, paste0(getwd(),"/results/MixtureModelsConll/res",id,".csv"),row.names = F)
  
  #clean up the thread
  rm(X,freq, instart,mod, model,rprob,rprob.new)
  gc()
  return(res)
}

#a function implementing sequantial confusion matrix Mixtures of Multinomoals
SEQ = function(id)
{
  
  
  if (!dir.exists(paste0(getwd(),"/results/ARMixtureModelsConll/"))){
    dir.create(paste0(getwd(),"/results/ARMixtureModelsConll/"),recursive = T)
  }
  if(file.exists(paste0(getwd(),"/results/ARMixtureModelsConll/res",id,".csv")))
  {
    print("A")
    return("A")
  }
  print(id)
  X = read.table(input,skip = size*(id-1),nrows = size,sep = ",")#[-1]
  X[X==0]=states
  nsours = dim(X)[2]
  
  
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
  write.csv( rprob , paste0(getwd(),"/results/ARMixtureModelsConll/resprobunmatched",id,".csv"),row.names = F)
  write.csv(res, paste0(getwd(),"/results/ARMixtureModelsConll/resunmatched",id,".csv"),row.names = F)
  
  res = resmap[res]
  for(i in 1:length(resmap))
  {
    idss = which(resmap == i)
    if(length(idss)==1)
      rprob.new[,i] = rprob[,idss]
    if(length(idss)>1)
      rprob.new[,i] = rowSums(rprob[,idss]) 
    
  }
  
  
  write.csv(cbind(classes[X[,rely]],classes[res]),paste0(getwd(),"/results/ARMixtureModelsConll/check",id,".csv"))
  rprob.new = rprob.new[,c(states,1:(states-1))]
  names(rprob.new) = paste0("T",0:(states-1))
  write.csv(rprob.new, paste0(getwd(),"/results/ARMixtureModelsConll/resprob",id,".csv"),row.names = F)
  res[res==states]=0
  write.csv(res, paste0(getwd(),"/results/ARMixtureModelsConll/res",id,".csv"),row.names = F)
  #clean up the thread
  rm(X,freq, instart,mod, model,rprob,rprob.new)
  gc()
  return(res)
}


#a function implementing confusion vector Mixtures of Multinomoals
CV = function(id)
{
  
  if (!dir.exists(paste0(getwd(),"/results/CVMixtureModelsConll/"))){
    dir.create(paste0(getwd(),"/results/CVMixtureModelsConll/"),recursive = T)
  }
  if(file.exists(paste0(getwd(),"/results/CVMixtureModelsConll/res",id,".csv")))
  {
    print("A")
    return("A")
  }
  print(id)
  X = read.table(input,skip = size*(id-1),nrows = size,sep = ",")#[-1]
  X[X==0]=states
  nsours = dim(X)[2]
  
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
      }
      k = k + tlen
    }
  }
  
  is.fixed = par[1:lp]
  is.fixed[1:states] = 1
  mod = fit(model,verbose =T,emcontrol=em.control(maxit = 25))
  
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
  write.csv( rprob , paste0(getwd(),"/results/CVMixtureModelsConll/resprobunmatched",id,".csv"),row.names = F)
  write.csv(res, paste0(getwd(),"/results/CVMixtureModelsConll/resunmatched",id,".csv"),row.names = F)
  
  res = resmap[res]
  
  for(i in 1:length(resmap))
  {
    idss = which(resmap == i)
    if(length(idss)==1)
      rprob.new[,i] = rprob[,idss]
    if(length(idss)>1)
      rprob.new[,i] = rowSums(rprob[,idss]) 
    
  }
  
  
  write.csv(cbind(classes[X[,rely]],classes[res]),paste0(getwd(),"/results/CVMixtureModelsConll/check",id,".csv"))
  rprob.new = rprob.new[,c(states,1:(states-1))]
  names(rprob.new) = paste0("T",0:(states-1))
  write.csv(rprob.new, paste0(getwd(),"/results/CVMixtureModelsConll/resprob",id,".csv"),row.names = F)
  res[res==states]=0
  write.csv(res, paste0(getwd(),"/results/CVMixtureModelsConll/res",id,".csv"),row.names = F)
  
  #clean up the thread
  rm(X,freq, instart,mod, model,rprob,rprob.new)
  gc()
  return(res)
}


#a function implementing sequantial confusion matrix Mixtures of Multinomoals
SEQ = function(id)
{
  
  
  if (!dir.exists(paste0(getwd(),"/results/ARMixtureModelsConll/"))){
    dir.create(paste0(getwd(),"/results/ARMixtureModelsConll/"),recursive = T)
  }
  if(file.exists(paste0(getwd(),"/results/ARMixtureModelsConll/res",id,".csv")))
  {
    print("A")
    return("A")
  }
  print(id)
  X = read.table(input,skip = size*(id-1),nrows = size,sep = ",")#[-1]
  X[X==0]=states
  nsours = dim(X)[2]
  
  
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
  write.csv( rprob , paste0(getwd(),"/results/ARMixtureModelsConll/resprobunmatched",id,".csv"),row.names = F)
  write.csv(res, paste0(getwd(),"/results/ARMixtureModelsConll/resunmatched",id,".csv"),row.names = F)
  
  res = resmap[res]
  for(i in 1:length(resmap))
  {
    idss = which(resmap == i)
    if(length(idss)==1)
      rprob.new[,i] = rprob[,idss]
    if(length(idss)>1)
      rprob.new[,i] = rowSums(rprob[,idss]) 
    
  }
  
  
  write.csv(cbind(classes[X[,rely]],classes[res]),paste0(getwd(),"/results/ARMixtureModelsConll/check",id,".csv"))
  rprob.new = rprob.new[,c(states,1:(states-1))]
  names(rprob.new) = paste0("T",0:(states-1))
  write.csv(rprob.new, paste0(getwd(),"/results/ARMixtureModelsConll/resprob",id,".csv"),row.names = F)
  res[res==states]=0
  write.csv(res, paste0(getwd(),"/results/ARMixtureModelsConll/res",id,".csv"),row.names = F)
  #clean up the thread
  rm(X,freq, instart,mod, model,rprob,rprob.new)
  gc()
  return(res)
}

#a function implementing accuracy model Mixtures of Multinomoals
ACC= function(id)
{
  
  
  
  if (!dir.exists(paste0(getwd(),"/results/ACCMixtureModelsConll/"))){
    dir.create(paste0(getwd(),"/results/ACCMixtureModelsConll/"),recursive = T)
  }
  
  if(file.exists(paste0(getwd(),"/results/ACCMixtureModelsConll/res",id,".csv")))
  {
    print("A")
    return("A")
  }
  print(id)
  X = read.table(input,skip = size*(id-1),nrows = size,sep = ",")#[-1]
  X[X==0]=states
  nsours = dim(X)[2]
  
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
  mod = fit(model,verbose =T,emcontrol=em.control(maxit = 25))#,equal = is.fixed)
  
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
  write.csv( rprob , paste0(getwd(),"/results/ACCMixtureModelsConll/resprobunmatched",id,".csv"),row.names = F)
  
  
  write.csv(res, paste0(getwd(),"/results/ACCMixtureModelsConll/resunmatched",id,".csv"),row.names = F)
  res = resmap[res]
  
  
  for(i in 1:length(resmap))
  {
    idss = which(resmap == i)
    if(length(idss)==1)
      rprob.new[,i] = rprob[,idss]
    if(length(idss)>1)
      rprob.new[,i] = rowSums(rprob[,idss]) 
    
  }
  
  
  write.csv(cbind(classes[X[,rely]],classes[res]),paste0(getwd(),"/results/ACCMixtureModelsConll/check",id,".csv"))
  rprob.new = rprob.new[,c(states,1:(states-1))]
  names(rprob.new) = paste0("T",0:(states-1))
  write.csv(rprob.new, paste0(getwd(),"/results/ACCMixtureModelsConll/resprob",id,".csv"),row.names = F)
  res[res==states]=0
  write.csv(res, paste0(getwd(),"/results/ACCMixtureModelsConll/res",id,".csv"),row.names = F)
  
  #clean up the thread
  rm(X,freq, instart,mod, model,rprob,rprob.new)
  gc()
  return(res)
}



#run the models
results = unlist(mclapply(X=sample.int(nseq,nseq,replace = F),FUN = CM,mc.preschedule = T,mc.cleanup = T,mc.cores = 20))
print(results)
results = unlist(mclapply(X=sample.int(nseq,nseq,replace = F),FUN = SEQ,mc.preschedule = T,mc.cleanup = T,mc.cores = 20))
print(results)
results = unlist(mclapply(X=sample.int(nseq,nseq,replace = F),FUN = CV,mc.preschedule = T,mc.cleanup = T,mc.cores = 20))
print(results)
results = unlist(mclapply(X=sample.int(nseq,nseq,replace = F),FUN = ACC,mc.preschedule = T,mc.cleanup = T,mc.cores = 20))
print(results)
gc()




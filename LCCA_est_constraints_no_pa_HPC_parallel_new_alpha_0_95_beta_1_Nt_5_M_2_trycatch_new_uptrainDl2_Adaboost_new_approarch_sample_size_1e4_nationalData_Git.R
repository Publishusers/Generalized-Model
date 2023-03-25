#this file is used to verify the algorithm of the paper
#this file only deal with LCCA soft splitting
#this file improve the prior_update for each iteration
#library(sf)

library(sp)
library(dplyr)
library(stringr)
library(tidyr)
library(tidyverse)
library(data.table)
#library(rgdal)
library(tictoc)
library(numbers)
#library(rgeos)
library(pracma)

#add machine learning library
#library(ggplot2)
#library(tidymodels)
library(caret)
library(carat)
#library(AmesHousing)
library(rsample)
library(rpart)
library(rpart.plot)
library(ROSE)
library(imbalance)

#add MNL model package
library(mlogit)
library(lmtest)
library(randomForest)
library(rattle)
library(recipes)

#soft spling package
library(SplitSoftening)
library(party)
library(mcmc)
# library(CVXR)

library(optimx)
library(DEoptim) #for global optimization

library(markovchain)
library(maxLik)
library(optimParallel)
library(scriptName)
library(parallel)
library(doParallel)
library(matrixStats)
library(smotefamily)


setwd("/public1/home/sc60401")


load("train_test_dataf_SMOTE_no_DL_1e4_nationalData.RData")

up_train <- train_dataf
up_test <- test_dataf

ds <- 1e4

load("pr_optim_new_av.RData")

#set cluster with 2 nodes. doSNOW is used for nested parallel computing
cl <- makeCluster(4, type = "FORK")

setDefaultCluster(cl=cl)
registerDoParallel(cl)



P <- 10 #the number of possible LCCAs
# k <- 2 #choose the number of attributes to form LCCA at each iteration, toge

#get the entropy at the root
FreqC <- as.data.frame(prop.table(table(up_train$Class))) #percentage of each class
#to balance the data, we upsample the original data to let each number of class equal
Entr <- sum(-log2(FreqC$Freq)*FreqC$Freq) #entropy at the terminal node



av <- colnames(up_train)[which(colnames(up_train) == c("TRVLCMIN","TRPMILES", "WHYFROM", "R_AGE",
                                                       "EDUC","HHSIZE","HHFAMINC","LIF_CYC","HHVEHCNT","WRKCOUNT","DRVRCNT"))]


up_train <- up_train %>% 
  mutate(PL = rep(1,nrow(up_train))) %>% #equal to PL*up_train$LR
  mutate(PR = rep(1,nrow(up_train))) %>% #equal to PR*up_train$LR
  mutate(LR = rep(1,nrow(up_train))) %>% #the probability of each observation progressed to the current terminal node for hard splitting, should be updated via each loop of MCMC
  mutate(SP = rep(1,nrow(up_train))) #selection probability for each observation
 
SP <<- NULL 

#add the probability matrix for each observation, alternative and terminal node
up_train <- up_train %>% 
  mutate(Prv = list(matrix(rep(1,5),ncol = 1))) 
  
#par: the historical used parameter vectors. iv: the historical used variable vector
#Prvt: store the historical probability for each terminal node: each element is a vector containing a list of 
#probability at the internal node
up_trainf <- list(up_train, "nt"=c(1), "id" = c(1),"tdv"=c(0), "entr" = c(Entr), "iv" = NULL, "par"= NULL,"Tree" = "T1","LL" = 0,"Prvt" = data.frame("Pr1"=rep(1,nrow(up_train)))) #"iv" is the used variables in acessors of each terminal node, at root, no variable used


#Prv: current probabiltiy vector for each alternative and each observation
MIF <- function(par,vn,up_trainf,Prv){
  
  #check if the length of par is k + 1
  if(length(par) != length(vn)+1){
    print("the length of par does not match")
    return(NULL)
  }
  
  #extract up_train
  up_train <- up_trainf[[1]]
  
  # print("hello1")
  #get the terminal node id, id = 0 for root
  # idx <- up_trainf$id
  
  #select the data by the given variable vector
  data <- as.data.frame(up_train[,which(colnames(up_train) %in% vn)])
  
  # print("hello2")
  
  #add the intercept parameter
  data <- data %>% 
    mutate(I = rep(1,nrow(data)))
  # print("hello3")
  #convert to matrix
  dataM <- as.matrix(data)
  # print("hello4")
  #set threshold, the length of beta is k+1, convert vector to a matrix
  beta <- par
  beta <- as.matrix(beta, nrow = length(beta))
  # print("hello5")
  #probability of selecting right and left child node for each 
  #observation, dataM %*% beta is the logistic regression equation (2)
  PR <- 1/(1 + exp(dataM %*% beta))
  PL <- 1- PR
  # print("hello6")
  #partion each observation to left and right by soft splitting, even for just single variable
  #up_train$LR should be updated at each iteration
  # idxd <- ifelse(idx==0,idx+1,idx) #idx == 0 for root, idx != 0 for other terminal nodes
  # print("hello7")
  # temp <- Reduce(rbind, up_train$Prv)[,idxd] #get the current probability for each observation and alternative
  # print("hello8")
  # PrvM <- matrix(unlist(cbind(up_train$Prv)),ncol = up_trainf$nt, byrow = TRUE)
  Prv <- as.data.frame(Prv)
  Prv <- Prv %>% 
    mutate(id = rep(1:nrow(up_train), each = 5))
  # print("hello9")
  
  Prvl <- aggregate(Prv,by = list(Prv$id),FUN = mean)
  # print("hello10")
  
  up_train$PL <- PL*Prvl$Prv
  up_train$PR <- PR*Prvl$Prv
  
  
  #extract data for fields "Class", "PL" and "PR"
  up_trainC <- up_train[,which(colnames(up_train) %in% c("Class","PL","PR"))]
  # print("hello12")
  
  up_trainC$Class <- as.numeric(as.character(up_trainC$Class))
  # print("hello13")
  up_trainCA <- aggregate(up_trainC, by = list(up_trainC$Class), FUN = sum)
  # print("hello14")
  
  FreqL <- up_trainCA$PL/sum(up_trainCA$PL)
  FreqR <- up_trainCA$PR/sum(up_trainCA$PR)
  
  # print("hello15")
  
  #new entropy
  EntrL <- sum(-log2(FreqL)*FreqL)
  EntrR <- sum(-log2(FreqR)*FreqR)
  # print("hello16")
  
  #weight for left and right nodes, sum(up_trainCA$PL) represents
  #the number of observations assigning to the left child
  wL <- sum(up_trainCA$PL)/(sum(up_trainCA$PL) + sum(up_trainCA$PR))
  wR <- sum(up_trainCA$PR)/(sum(up_trainCA$PL) + sum(up_trainCA$PR))
  # print("hello17")
  
  EntrN <- sum(wL*EntrL,wR*EntrR,na.rm = TRUE)
  # print("hello18")
  
  return(EntrN)
}


#for selected variables at the root
#output: values: P minimum entropies
#sv: selected variables
#parameters: optimized parameters for variable vector
#pM: prior for each variable

prior_root <- function(up_trainf){
  
  if(up_trainf$id != 1)
  {
    print("this is not root")
    return(NULL)
  }
  
  #extract up_train
  up_train <- up_trainf[[1]]
  
  #get the Prv
  Prv <- Reduce(rbind, up_train$Prv)[,up_trainf$id]
  
  #full variable vector
  nkn <- matrix(c(rep(colnames(up_train)[-which(colnames(up_train)%in%c("Class","ID","LR","PL","PR","Prv"))],each = 2),
                  matrix(nk,nrow = 1)),nrow = 2)
  
  # nknt <- nkn[,7:13]
  
  #full LCCA containing the single variable
  vector <- c(1:ncol(nkn))
  # vector <- c(1:ncol(nknt))
  value <- list()
  parameter <- c()
  
  #define a for loop, each of which for one LCCA.
  #then select the minimum entropy
  for(i in vector){
    
    vn <-  unique(nkn[,i]) #the variable names, can be single or LCCA
    # vn <-  unique(nknt[,i]) #the variable names, can be single or LCCA
    lower <- rep(0,length(vn)+1) #lower bound of the parameter, add one intercept
    upper <- rep(20,length(vn)+1) #upper bound of the parameter, add one intercept
    
    print(i)
    # print("Hello")
    tic()
    result <- DEoptim(fn = MIF,lower = lower, upper = upper,vn = vn,
                      up_trainf = up_trainf, Prv = Prv, control = list(itermax = 1)) #optimize the parameter
    toc()
    
    
    value <- c(value,result$optim$bestval)
    # parameter <- rbind(parameter,result$optim$bestmem)
    parameter <- c(parameter,list(result$optim$bestmem))
  }
  
  #get the prior for selecting the LCCA
  #the number of LCCAs is limited to P
  sv <- nkn[,order(unlist(value))[1:P]]
  # sv <- nknt[,order(unlist(value))[1:P]] 
  
  #selected minimum entropy vector for the root node
  values <- value[order(unlist(value))[1:P]]
  parameters <- parameter[order(unlist(value))[1:P]] #the parameter includes one intercept, so the length must be plus one
  MI <- Entr - unlist(values) 
  pM <- MI/sum(MI) #the selection probability at the root
  
  return(list(sv,values,parameters,pM))
}

sv <- pr[[1]]

parameters <- pr[[3]]

# pM <- pr[[4]]
pM <- rep(1/ncol(sv),ncol(sv))

# par_root <- list("sv"=sv, "parameters" = parameters, "pM" = pM)
par_root <- list("sv"=sv, "parameters" = parameters, "pM" = pM)

prior_update <- function(up_trainf,sv,idx){
  
  if(up_trainf$id == 0)
  {
    print("this is root")
    return(NULL)
  }
  
  #get the updated entropy for the current terminal node
  EntrU <- tail(up_trainf$entr[[idx]],1) #EntrU is a vector
  
  #extract up_train
  up_train <- up_trainf[[1]]
  
  #get the Prv
  # Prv <- Reduce(rbind, up_train$Prv)[,up_trainf$id]
  PrvM <- Reduce(rbind, up_trainf[[1]]$Prv)
  Prv <- PrvM[,idx]
  # Entr <- sum(-log2(Freq)*Freq)
  
  #for each possible LCCA, find the optimum parameter vector
  lv <- c(1:ncol(sv))
  value <- list()
  parameter <- c()
  
  for (i in lv) { #for each LCCA, call DEoptim to obtain the optimized parameter
    svs <- unique(sv[,i]) #may be LCCA, may be just one single variable
    
    # print(paste0("here",i))
    
    lower <- rep(0,length(svs)+1)
    upper <- rep(20,length(svs)+1)
    result <- DEoptim(fn = MIF,lower = lower, upper = upper,vn = svs,up_trainf = up_trainf, Prv = Prv, control = list(itermax = 1))
    value <- c(value,result$optim$bestval)
    
    parameter <- c(parameter,list(result$optim$bestmem))
  }
  
  #sort the value, assign probability
  values <- value[order(unlist(value))[1:P]] #entropy
  MIU <- EntrU - unlist(values)
  
  svsu <- sv[,order(unlist(value))[1:P]]
  
  parametersu <- parameter[order(unlist(value))[1:P]]
  
  pMu <- MIU/sum(MIU) 
  
  #update sub-fields: nt, id, tdv, Prv at two childs, entr, adding
  #sequences: left and right
  
  return(list("LCCA"=svsu,"Ent" = values,"par_u" = parametersu,"Pr"=pMu))
}

LL <- function(Coe_MNL, av_mnl, nt,data, PrvM,wv)
{
  #get the updated up_train
  # up_train <- up_trainf[[1]]
  up_train <- data
  # nt <- up_trainf$nt #number of terminal nodes
  vec <- c(1:nt)
  
  #PrvM: 
  PrvM <- as.data.frame(PrvM) %>% mutate(id = rep(1:nrow(up_train),each = 5))
  
  #accumulated probability at each terminal node for each observation
  Prvl <- aggregate(PrvM,by = list(PrvM$id),FUN = mean) 
  Prvlt <- Prvl[,-which(colnames(Prvl)%in%c("Group.1","id"))] #dimension: nrow(up_train) X number of terminal nodes
  Prvlt <- as.data.frame(Prvlt) #transform the vector to data frame
  # #all variables for LCCA and MNL, no LCCA
  # av <- colnames(up_train)[1:10]
  
  #extract the data for the selected attributes and class
  # up_trainD <- up_train[,which(colnames(up_train) %in% c("distance_miles","workrelated","persons_count",
  #                                                        "hincome","worker_count","student_count","driver_license","education","vpp","lpp","bpp","Class"))]
  
  up_trainD <- up_train
  up_trainDR <- up_trainD[rep(1:nrow(up_trainD),each = 5),]
  up_trainDR <- up_trainDR %>%
    mutate(choice = as.numeric(rep(c(1,3,4,5,6),nrow(up_trainD)) == up_trainDR$Class)) %>%
    mutate(travel_mode = rep(c(1,3,4,5,6), nrow(up_trainD)))

  #reshape Coe_MNL into a matrix with 4*(length(av_mnl)+1) rows, nt columns
  Coe_MNLM <- Reshape(Coe_MNL,n=4*(length(av_mnl)+1))
  
  #extract the data corrosponding to av_mnl
  up_trainDR_av <- up_trainDR[,which(colnames(up_trainDR)%in%c(av_mnl,"choice"))]
  
  #add index field
  up_trainDR_av <- up_trainDR_av %>% 
    mutate("m_idx" = rep(c(1:5),nrow(up_trainDR_av)/5))
  up_trainDR_avs <<- up_trainDR_av[which(up_trainDR_av$choice==1),]
  #get the ncol of up_trainDR_avs
  m <- ncol(up_trainDR_avs)
  
  #convert the avaible mnl variables to matrix
  up_trainDR_avsM <- as.matrix(up_trainDR_avs[,-c(m-1,m)])
  
  #add 1 before the first column of up_trainDR_avsM for intercept
  up_trainDR_avsM <- cbind("Inter"=rep(1,nrow(up_trainDR_avsM)),up_trainDR_avsM)
  
  #set the prob matrix for storing
  probMN <- NULL
  utilMaM <- NULL
  probvv <- NULL
  
  #set the prob matrix for whole selection
  prob_SM <- matrix(rep(0,nrow(up_trainDR_avsM)*5), ncol = 5)
  
  # print("test_shangbo1")
  # print("shang10")
  for (i in vec) {
    Coei <- Reshape(Coe_MNLM[,i], n = 4)
    
    #product with up_trainDR_avsM to get the utility function
    utilM <- up_trainDR_avsM %*% t(Coei)
    
    #get the exp and add the first column for alt 5
    utilMa <- cbind(rep(1,nrow(utilM)),exp(utilM))
    
    #check if utilMa contains Inf or NaN
    temp3 <- rowProds(!is.infinite(utilMa))
    
    idx_f <- which(temp3 == 1) #index for finite values
    idx_if <- which(temp3 == 0) #index for infinite values
    
    probM <- zeros(nrow(up_trainDR_avsM), 5) #initialize the probM matrix
    
    if(!isempty(idx_f))
    {
      # utilMaS <- ifelse(length(idx_f)==1,sum(utilMa[idx_f,]),rowSums(utilMa[idx_f,]))
      if(length(idx_f)==1)
      {
        utilMaS <- sum(utilMa[idx_f,])
      }
      else
      {
        utilMaS <- rowSums(utilMa[idx_f,])
      }
      
      probM[idx_f,] <- utilMa[idx_f, ]/utilMaS
    }
    
    if(!isempty(idx_if))
    {
      idxm_if <- apply(matrix(utilMa[idx_if,], ncol = 5), 1, which.max)
      
      lv <- c(1:length(idx_if))
      temp <- foreach(j = lv) %do%
        {
          probM[idx_if[j], idxm_if[j]] <- 1  
        }
      # toc()
    }
    
    
    #get the selected choice
    lv <- c(1:nrow(up_trainDR_avs))
    probv <- unlist(lapply(lv, function(x){probM[x, up_trainDR_avs$m_idx[x]]}))
    probvv <- cbind(probvv, probv)
    
    #factor with the accumulated probabiltiy on that termina node
    probf <- Prvlt[,i]*probv
    
    #factor with the accumulated probability on the ternimal nodes for selection matrix
    prob_SMi <- Prvlt[,i]*probM
    
    #store the prob vector
    probMN <- cbind(probMN, probf)
    
    utilMaM <- cbind(utilMaM, utilMa)
    
    prob_SM <- prob_SM + prob_SMi
    
  }
  
  
  SP <<- rowSums(probMN)#for the desired alternative
  
  utilMaM <<- utilMaM
  probvv <<- probvv
  prob_SM <<- prob_SM
  
  wp <- wv*SP
  idx0 <- which(wp == 0)
  
  if(!isempty(idx0))
  {
    wp[idx0] <- min(wp[wp != min(wp)])
  }
  
  
  wp <<- wp
  
  #get the accumulated LL via wv
  #weight vector is used to weight different samples
  return(sum(log(wp)))
  # return(sum(log(wv*rowSums(probMN))))
  
}

#define gradient function of log lik function
#from the equation (11)
#Coe_MNL: 4*(length(av_mnl)+1) vector
#av_mnl: available MNL variables
#up_trainf: training data
#PrvM: 
logLikGrad <- function(Coe_MNL, av_mnl, nt, data, PrvM,wv){
  
  #get the updated up_train
  # up_train <- up_trainf[[1]]
  up_train <- data
  
  #number of terminal nodes
  # nt <- up_trainf$nt
  vec <- c(1:nt)
  #get P_m for each observation at each terminal node
  PrvM <- as.data.frame(PrvM) %>% mutate(id = rep(1:nrow(up_train),each = 5))
  # #accumulated probability at each terminal node for each observation
  Prvl <- aggregate(PrvM,by = list(PrvM$id),FUN = mean)
  Prvlt <- Prvl[,-which(colnames(Prvl)%in%c("Group.1","id"))] #dimension: nrow(up_train) X number of terminal nodes
  Prvlt <- as.data.frame(Prvlt) #transform the vector to data frame
  #extract the data for the selected attributes and class
  up_trainD <- up_train[,which(colnames(up_train) %in% c("TRVLCMIN","TRPMILES","WHYFROM","R_AGE",
                                                         "EDUC","HHSIZE","HHFAMINC","LIF_CYC","HHVEHCNT","WRKCOUNT","DRVRCNT", "Class"))]
  
  
  
  # up_trainD <- up_train
  up_trainDR <- up_trainD[rep(1:nrow(up_trainD),each = 5),]
  up_trainDR <- up_trainDR %>%
    mutate(choice = as.numeric(rep(c(1,3,4,5,6),nrow(up_trainD)) == up_trainDR$Class)) %>%
    mutate(travel_mode = rep(c(1,3,4,5,6), nrow(up_trainD)))
  
  # up_trainDR <- up_trainDR %>%
  #   mutate(choice = as.numeric(rep(c(1,2,5,6,15),nrow(up_trainD)) == up_trainDR$Class)) %>%
  #   mutate(travel_mode = rep(c(1,2,5,6,15), nrow(up_trainD)))
  
  #calculate the selection probabiltiy for each user at each terminal node
  #reshape Coe_MNL into a matrix with 4*(length(av_mnl)+1) rows, nt columns
  Coe_MNLM <- Reshape(Coe_MNL,n=4*(length(av_mnl)+1))
  
  #extract the data corrosponding to av_mnl
  up_trainDR_av <- up_trainDR[,which(colnames(up_trainDR)%in%c(av_mnl,"choice"))]
  
  #add index field
  up_trainDR_av <- up_trainDR_av %>% 
    mutate("m_idx" = rep(c(1:5),nrow(up_trainDR_av)/5))
  
  up_trainDR_avs <- up_trainDR_av[which(up_trainDR_av$choice==1),]
  
  #get the ncol of up_trainDR_avs
  m <- ncol(up_trainDR_avs)
  
  #convert the avaible mnl variables to matrix
  up_trainDR_avsM <- as.matrix(up_trainDR_avs[,-c(m-1,m)])
  
  #add 1 before the first column of up_trainDR_avsM for intercept
  up_trainDR_avsM <- cbind("Inter"=rep(1,nrow(up_trainDR_avsM)),up_trainDR_avsM)
  
  #set the factored prob matrix for K_n for storing 
  probMN <- NULL
  
  #set the selection probabiltiy vector for K_n
  probKN <- NULL
  
  #set the space for stroing the whole selection matrix
  probMM <- NULL
  
  kvec <- c(2:5)#params for 4 alternatives to be optimized
  kkvec <- c(1,3,4,5,6)
  # kkvec <- c(1,2,5,6,15)
  alt <- kkvec[up_trainDR_avs$m_idx] #selected alternatives
  
  #set the store vector for nominator Dim: 4(length(av_mnl)+1) X nt
  NOM <- NULL
  for (i in vec) {
    Coei <- Reshape(Coe_MNLM[,i], n = 4)
    
    #product with up_trainDR_avsM to get the utility function
    utilM <- up_trainDR_avsM %*% t(Coei)
    
    #get the exp and add the first column for alt 5
    utilMa <- cbind(rep(1,nrow(utilM)),exp(utilM))
    
    #check if utilMa contains Inf or NaN
    temp3 <- rowProds(!is.infinite(utilMa))
    
    idx_f <- which(temp3 == 1) #index for finite values
    idx_if <- which(temp3 == 0) #index for infinite values
    
    probM <- zeros(nrow(up_trainDR_avsM), 5) #initialize the probM matrix
    
    if(!isempty(idx_f))
    {
      # utilMaS <- ifelse(length(idx_f)==1,sum(utilMa[idx_f,]),rowSums(utilMa[idx_f,]))
      if(length(idx_f)==1)
      {
        utilMaS <- sum(utilMa[idx_f,])
      }
      else
      {
        utilMaS <- rowSums(utilMa[idx_f,])
      }
      
      probM[idx_f,] <- utilMa[idx_f, ]/utilMaS
    }
    
    if(!isempty(idx_if))
    {
      idxm_if <- apply(matrix(utilMa[idx_if,], ncol = 5), 1, which.max)
      
      # apply(probM[idx_if,], 1, function(x){
      #   
      #   x[idxm_if] <- 1
      #   return(x)
      #   })
      
      lv <- c(1:length(idx_if))
      # 
      # apply(probM[idx_if,], 1, function(x){
      #   
      #   idxm_if
      #   
      # })
      # tic()
      temp <- foreach(j = lv) %do%
        {
          probM[idx_if[j], idxm_if[j]] <- 1  
        }
      # toc()
    }
    
    #get the probability matrix for each alt and observation
    # probM <- utilMa/rowSums(utilMa)
    
    #get the selected choice
    lv <- c(1:nrow(up_trainDR_avs))
    probv <- unlist(lapply(lv, function(x){probM[x, up_trainDR_avs$m_idx[x]]}))
    
    #factor with the accumulated probabiltiy on that termina node
    probf <- Prvlt[,i]*probv
    
    #store the prob vector, Dim: nrow(up_trainDR_avsM)*nt
    probMN <- cbind(probMN, probf)
    
    #store the selection vector for K_n, Dim: nrow(up_trainDR_avsM)*nt 
    probKN <- cbind(probKN, probv)
    
    #store the whole selection matrix, Dim: nrow(up_trainDR_avsM) * (5*nt)
    probMM <- cbind(probMM, probM)
    
    #set the prob matrix for the i-th terminal node
    probi <- NULL
    
    for (k in kvec) {
      kn <- kkvec[k] #from 1 alt (1,15,6,2)
      
      #get the observations whose selection is kn
      idx_kn <- which(alt == kn)
      idx_k <- which(alt != kn)
      
      Pr_kn_mn <- probv[idx_kn]
      Pr_knk_mn <- probv[idx_k] #Pr_kn for those selection is not for kn
      
      #get the Pr_k, k != k_n
      probMk <- probM[,k]
      Pr_k_mn <- probMk[idx_k] #Pr_k for those selection is not for kn
      
      #get the nominator, Dim: nrow(up_trainDR_avsM) X (length(av_mnl)+1)
      nomv <- rep(0,nrow(up_trainDR_avs))
      nomv[idx_kn] <- Pr_kn_mn*(1-Pr_kn_mn)
      nomv[idx_k] <- -Pr_knk_mn*Pr_k_mn
      
      nomvM <- repmat(matrix(nomv,nrow = length(nomv)),1,length(av_mnl)+1)
      
      #get the product: dim:nrow(up_trainDR_avsM) X  length(av_mnl)+1
      probi <- cbind(probi,up_trainDR_avsM * nomvM * repmat(as.matrix(Prvlt[,i]),1,length(av_mnl)+1))
    }
    
    #reshape col order of probi according to trainingM_init$coefficients
    #order of coefficient: Inter2, ... Inter5, var1_2, ..., var1_5 ... 
    probio <- probi[,Reshape(t(Reshape(c(1:ncol(probi)), n = length(av_mnl)+1)), n = 1)]
    
    #store the prob for the m-th terminal node
    NOM <- cbind(NOM,probio)
  }
  
  #get the decimator of (11)
  dec <- rowSums(probMN)
  
  Nd <- NOM/dec
  
  #replace NaN with 0
  Nd <- replace(Nd, is.nan(Nd),0)
  
  #replace Inf with 0
  Nd <- replace(Nd, is.infinite(Nd),0)
  
  #impose the weight factor
  wv_mat <- repmat(as.matrix(wv), n = 1, m = ncol(Nd))
  
  #get the fisher matrix
  FIM <- colSums(wv_mat*Nd)
  
  # print("logLikGrad is:")
  # print(FIM)
  
  if(anyNA(FIM) || !isempty(which(is.infinite(FIM))))
  {
    print("this is NaN")
    save(Coe_MNL, file = "Coe_MNL_logLikGrad.RData")
    save(nt, file = "nt_logLikGrad.RData")
    save(av_mnl, file = "av_mnl_logLikGrad.RData")
    save(data, file = "data_logLikGrad.RData")
    save(PrvM, file = "PrvM_logLikGrad.RData")
    save(wv, file = "wv_logLikGrad.RData")
    save(FIM, file = "FIM_logLikGrad.RData")
    
  }
  
  return(FIM)
  
}

LL_split <- function(param, av_del, av_MNL,n, data,PrvM,wv){
  # print("param for LL_split is: ")
  # print(param)
  up_train <- data
  # par_split <- param[[1]] #for node splitting
  # par_mnl <- param[[2]] #for MNL model at the two terminal nodes
  par_split <- param[1:(length(av_del)+1)]
  par_mnl <- param[(length(av_del)+2):length(param)]
  
  ##extract the train data for av_del
  data_split <- up_train[,which(colnames(up_train) %in% av_del)]
  data_splitM <- as.matrix(as.data.frame(data_split) %>% mutate(I = rep(1,nrow(data))))
  paruM <- as.matrix(par_split, nrow = length(par_split))
  PR <- 1/(1 + exp(data_splitM %*%  paruM))
  PL <- 1 - PR
  
  Prvm <- PrvM[,n] * rep(PL,each = 5)
  Prvn <- PrvM[,n] * rep(PR,each = 5)

  # PrvM_MNL <- cbind(PrvM[,n]*rep(PL, each = 5), PrvM[,n]*rep(PR, each = 5))
  
  #update PrvM at each terminal node
  nt <- ncol(PrvM) + 1
  
  # print("nt is:")
  # print(nt)
  
  Prv_temp1 <- ifelse(n-1>=1,list(PrvM[,1:(n-1)]),NA)
  Prv_temp2 <- ifelse(n+1<=ncol(PrvM),list(PrvM[,(n+1):ncol(PrvM)]),NA)
  
  temp <- cbind(Prv_temp1[[1]],Prvm,Prvn,Prv_temp2[[1]])
  
  PrvM_MNL <- as.data.frame(temp[,which(colSums(is.na(temp)) == 0)])
  ##extract the train data for av_mnl

  data_MNL <- up_train[, which(colnames(up_train) %in%  av_MNLc)]
  # print("wang5")
  # nt <- 2 #from root
  # print("PrvM_MNL is:")
  # print(PrvM_MNL)
  # print("wv is:")
  # print(wv)
  
  loglik <- LL(par_mnl, av_MNL, nt, data_MNL,PrvM_MNL,wv)
  

  # loglik <- 1
  # print("wang6")
  
  print(loglik)
  # 
  # if(is.nan(loglik)||is.infinite(loglik))
  # {
  #   print("loglik is infinite")
  #   save(par_mnl, file = "par_mnl.RData")
  #   save(nt, file = "nt.RData")
  #   save(av_MNL, file = "av_MNL.RData")
  #   save(data_MNL, file = "data_MNL.RData")
  #   save(PrvM_MNL, file = "PrvM_MNL.RData")
  #   save(wv, file = "wv.RData")
  #   save(av_del, file = "av_del.RData")
  #   return()
  # }
  # return(ifelse(is.infinite(loglik), -4000,loglik))
  return(loglik)
  
}

LL_split_grad <- function(param, av_del, av_MNL,n, data,PrvM,wv){
  # print("param for LL_split_grad is: ")
  # print(param)
  
  up_train <- data
  
  par_split <- param[1:(length(av_del)+1)]
  par_mnl <- param[(length(av_del)+2):length(param)]
  
  data_split <- up_train[,which(colnames(up_train) %in% av_del)]
  data_splitM <- as.matrix(as.data.frame(data_split) %>% mutate(I = rep(1,nrow(data))))
  paruM <- as.matrix(par_split, nrow = length(par_split))
  
  data_split_util <- exp(data_splitM %*%  paruM) #it might have Inf
  
  PR <- 1/(1 + data_split_util)
  PL <- 1 - PR
  
  Prvm <- PrvM[,n] * rep(PL,each = 5) #J+1
  Prvn <- PrvM[,n] * rep(PR,each = 5) #J
  
  nt <- ncol(PrvM) + 1
  Prv_temp1 <- ifelse(n-1>=1,list(PrvM[,1:(n-1)]),NA)
  Prv_temp2 <- ifelse(n+1<=ncol(PrvM),list(PrvM[,(n+1):ncol(PrvM)]),NA)
  
  temp <- cbind(Prv_temp1[[1]],Prvm,Prvn,Prv_temp2[[1]])
  
  PrvM_MNL <- as.data.frame(temp[,which(colSums(is.na(temp)) == 0)])
  PrvM_MNL <- PrvM_MNL %>% mutate(id = rep(1:nrow(up_train),each = 5))
  
  Prvl_MNL <- aggregate(PrvM_MNL,by = list(PrvM_MNL$id),FUN = mean)
  Prvlt_MNL <- Prvl_MNL[,-which(colnames(Prvl_MNL)%in%c("Group.1","id"))] #dimension: nrow(up_train) X number of terminal nodes
  Prvlt_MNL <- as.data.frame(Prvlt_MNL) #transform the vector to data frame
  
  
  ##extract the train data for av_mnl
  # av_MNLc <- c(av_MNL, "Class")
  # data_MNL <- up_train[, which(colnames(up_train) %in%  av_MNLc)]
  up_trainD <- up_train
  up_trainDR <- up_trainD[rep(1:nrow(up_trainD),each = 5),]
  
  up_trainDR <- up_trainDR %>%
    mutate(choice = as.numeric(rep(c(1,3,4,5,6),nrow(up_trainD)) == up_trainDR$Class)) %>%
    mutate(travel_mode = rep(c(1,3,4,5,6), nrow(up_trainD)))
  
  up_trainDR_av <- up_trainDR[,which(colnames(up_trainDR)%in%c(av_MNL,"choice"))]
  
  up_trainDR_av <- up_trainDR_av %>%
    mutate("m_idx" = rep(c(1:5),nrow(up_trainDR_av)/5))
  # print("shang7")
  up_trainDR_avs <- up_trainDR_av[which(up_trainDR_av$choice==1),] #select the corret choice
  # print("shang8")
  #get the ncol of up_trainDR_avs
  m <- ncol(up_trainDR_avs)
  
  #convert the avaible mnl variables to matrix
  up_trainDR_avsM <- as.matrix(up_trainDR_avs[,-c(m-1,m)])
  # print("shang9")
  #add 1 before the first column of up_trainDR_avsM for intercept
  up_trainDR_avsM <- cbind("Inter"=rep(1,nrow(up_trainDR_avsM)),up_trainDR_avsM)
  
  vec <- c(1:nt)
  par_mnlM <- Reshape(par_mnl,n=4*(length(av_MNL)+1))
  
  probMN <- NULL
  
  probvv <- NULL
  #utlity functions for each alternative at each terminal node
  # utilMaS <- NULL
  # utilMaM <- NULL
  # utilMavv <- NULL
  # utilMaD <- NULL
  
  for (i in vec) { #for each terminal node
    Coei <- Reshape(par_mnlM[,i], n = 4)
    
    utilM <- up_trainDR_avsM %*%  t(Coei)
    
    utilMa <- cbind(rep(1,nrow(utilM)), exp(utilM))
    
    #check if utilMa contains Inf or NaN
    temp3 <- rowProds(!is.infinite(utilMa))
    
    idx_f <- which(temp3 == 1) #index for finite values
    idx_if <- which(temp3 == 0) #index for infinite values
    
    probM <- zeros(nrow(up_trainDR_avsM), 5) #initialize the probM matrix
    
    if(!isempty(idx_f))
    {
      # utilMaS <- ifelse(length(idx_f)==1,sum(utilMa[idx_f,]),rowSums(utilMa[idx_f,]))
      if(length(idx_f)==1)
      {
        utilMaS <- sum(utilMa[idx_f,])
      }
      else
      {
        utilMaS <- rowSums(utilMa[idx_f,])
      }
      
      probM[idx_f,] <- utilMa[idx_f, ]/utilMaS
    }
    
    if(!isempty(idx_if))
    {
      idxm_if <- apply(matrix(utilMa[idx_if,], ncol = 5), 1, which.max)
      
      # apply(probM[idx_if,], 1, function(x){
      #   
      #   x[idxm_if] <- 1
      #   return(x)
      #   })
      
      lv <- c(1:length(idx_if))
      # 
      # apply(probM[idx_if,], 1, function(x){
      #   
      #   idxm_if
      #   
      # })
      # tic()
      temp <- foreach(j = lv) %do%
        {
          probM[idx_if[j], idxm_if[j]] <- 1  
        }
      # toc()
    }
    
    # probM <- utilMa/rowSums(utilMa)
    
    # utilMaS <- cbind(utilMaS, rowSums(utilMa))
    # utilMaM <- cbind(utilMaM, utilMa)
    
    #construct the utility difference matrix, for 1,15,6,2 alternatives all observations
    # utilMaD <- cbind(utilMaD, -utilMa[,c(2:ncol(utilMa))])
    
    lv <- c(1:nrow(up_trainDR_avs))
    probv <- unlist(lapply(lv, function(x){probM[x,up_trainDR_avs$m_idx[x]]})) 
    # utilMav <- unlist(lapply(lv, function(x){utilMa[x,up_trainDR_avs$m_idx[x]]})) #get the exp utility for desired choice for each user
    
    
    # utilMavv <- cbind(utilMavv, utilMav)
    
    probvv <- cbind(probvv, probv)
    
    probf <- Prvlt_MNL[,i]*probv
    
    probMN <- cbind(probMN, probf)
    
    ##get exp(V_i_j) and exp(V_m_j) for each terminal node
  }
  
  up_train_MNL <- up_train[,-which(colnames(up_train) %in%  av_del)]
  PrvM_MNL_MNL <- PrvM_MNL[,-which(colnames(PrvM_MNL) %in% "id")]
  
  FIM_beta <- logLikGrad(par_mnl, av_MNL, nt, up_train_MNL,PrvM_MNL_MNL,wv)
  
  #get the denominator for first derivative of LL to beta_kmj
  SP <- rowSums(probMN)
  
  #determine alpha_gr
  pd <- probvv[,(n + 1)] - probvv[,n]
  
  # PrvM_MNL <- PrvM %>% mutate(id = rep(1:nrow(up_train),each = 5))
  # Prvl_MNL <- aggregate(PrvM_MNL,by = list(PrvM_MNL$id),FUN = mean)
  # Prvlt_MNL <- Prvl[,-which(colnames(Prvl_MNL)%in%c("Group.1","id"))] #dimension: nrow(up_train) X number of terminal nodes
  # Prvlt_MNL <- as.data.frame(Prvlt_MNL) #transform the vector to data 
  
  PrvMn_df <- as.data.frame(PrvM[,n]) %>% mutate(id = rep(1:nrow(up_train),each = 5))
  PrvMn_df <- aggregate(PrvMn_df,by = list(PrvMn_df$id),FUN = mean)
  PrvMn_df <- PrvMn_df[,-which(colnames(PrvMn_df)%in%c("Group.1","id"))]
  PrvMn_dfM <- repmat(as.matrix(PrvMn_df), n = 1, m = length(av_del)+1)
  prod_PrvMx <- PrvMn_dfM*data_splitM
  
  dr <- data_split_util/(1 + data_split_util)^2
  
  idx_if2 <- which(is.infinite(data_split_util))
  
  if(!isempty(idx_if2))
  {
    dr[idx_if2] <- 0
  }
  
  dpS <- dr*pd/SP 
  dpS[c(which(is.nan(dpS)), which(is.infinite(dpS)))] <- 0
  
  pdpS <- prod_PrvMx*repmat(dpS, m = length(av_del)+1, n = 1)
  
  #pose the weighting factor
  wv_mat <- repmat(as.matrix(wv), m = ncol(pdpS), n = 1)
  
  #get the first derivative of LL to alpha
  alpha_gr <- as.vector(colSums(wv_mat*pdpS))
  
  #get the fisher matrix
  FIM <- c(alpha_gr, FIM_beta)
  
  
  return(FIM)
  
}

#get the attributes for node splitting by 
#determining the gradient to the available variables
#m: the number of selected attributes for node splitting

Get_av_del <- function(parmnl, av_mnl, PrvM, data, wv, m){
  nt <- ncol(PrvM)
  up_train <- data
  
  parM <- Reshape(parmnl, n = 4)
  AV <- c(1:length(av_mnl)) #for different attributes
  avv <- c(1:4) #number of alternatives: 1,15,6,2
  
  grv <- as.matrix(rep(0, length(AV)))
  
  PrvM_id <- as.data.frame(PrvM) %>% mutate(id = rep(1:nrow(up_train),each = 5))
  # print("shang2")
  #accumulated probability at each terminal node for each observation
  Prvl <- aggregate(PrvM_id,by = list(PrvM_id$id),FUN = mean)
  Prvlt <- Prvl[,-which(colnames(Prvl)%in%c("Group.1","id"))] #dimension: nrow(up_train) X number of terminal nodes
  Prvlt <- as.data.frame(Prvlt) #tr
  
  # prob_SM
  parMu <- rbind(rep(0,ncol(Prvlt)*(length(av_mnl)+1)), parM)
  
  #utilMaM: size: N X 5*ncol(Prvlt)
  utilS <- NULL
  sum_exp <- NULL
  nom <- matrix(rep(0, nrow(up_train)*length(av_mnl)), nrow = nrow(up_train))
  vec <- c(1:nt)
  
  #get utilMaM, probvv and up_trainDR_avs
  LL(parmnl, av_mnl,nt,up_train,PrvM,wv)
  
  for (i in vec) {
    # print("test1")
    utilMaMi <- utilMaM[,((i-1)*5+1):(i*5)]
    # print("test2")
    utilSi <- as.data.frame(rowSums(utilMaMi))
    probvvi <- as.data.frame(probvv[,i])
    # utilS <- cbind(utilS,rowSums(utilMaM[,((i-1)*5+1):(i*5)])) #the sum of utility function for each terminal node
    parMui <- parMu[,((i-1)*(length(av_mnl)+1)+1):(i*(length(av_mnl)+1))]
    
    sum_expi <- parMui[up_trainDR_avs$m_idx,c(2:ncol(parMui))]* utilSi[,rep(1,ncol(parMui)-1)]
    sum_beta_expi <- t(t(parMui[,c(2:ncol(parMui))]) %*% t(utilMaMi))
    
    sdi <- 1 - sum_beta_expi / sum_expi #get the difference
    
    nomi <- Prvlt[,rep(1, (ncol(parMui)-1))] * parMui[up_trainDR_avs$m_idx,c(2:ncol(parMui))] * probvvi[,rep(1,(ncol(parMui)-1))]
    
    nom <- nom + nomi*sdi
    
  }
  
  denom <- as.data.frame((rowSums(Prvlt * probvv))^2)
  temp4 <- nom / denom[,rep(1,length(av_mnl))]
  wv_mat <- repmat(as.matrix(wv), m = ncol(temp4), n = 1)
  
  gr_x <- colSums(wv_mat*temp4,na.rm = TRUE)
  
  # av_del <- av_mnl[which.min(abs(gr_x))]
  
  idx_gr <- order(abs(gr_x)) #ascent order'
  
  av_del <- av_mnl[idx_gr[1:m]]
  
  return(av_del)
  
}

#build the matching table for tree space
MT <- data.frame(c(1:26))
colnames(MT) <- "id"

#code terminal nodes
MTl <- list()
MTl[1] <- list(c("0"))
MTl[2] <- list(c("00","01"))
MTl[3] <- list(c("000","001","01"))
MTl[4] <- list(c("00","010","011"))
MTl[5] <- list(c("000","001","010","011"))
MTl[6] <- list(c("0000","0001","001","01"))
MTl[7] <- list(c("000","0010","0011","01"))
MTl[8] <- list(c("0000","0001","001","010","011"))
MTl[9] <- list(c("000","0010","0011","010","011"))
MTl[10] <- list(c("000","001","0100","0101","011"))
MTl[11] <- list(c("000","001","010","0110","0111"))
MTl[12] <- list(c("00","0100","0101","011"))
MTl[13] <- list(c("00","010","0110","0111"))
MTl[14] <- list(c("0000","0001","0010","0011","01"))
MTl[15] <- list(c("0000","0001","0010","0011","010","011"))
MTl[16] <- list(c("0000","0001","001","0100","0101","011"))
MTl[17] <- list(c("0000","0001","001","010","0110","0111"))
MTl[18] <- list(c("000","001","0100","0101","0110","0111"))
MTl[19] <- list(c("000","0010","0011","010","0110","0111"))
MTl[20] <- list(c("000","0010","0011","0100","0101","011"))
MTl[21] <- list(c("0000","0001","0010","0011","010","0110","0111"))
MTl[22] <- list(c("0000","0001","001","0100","0101","0110","0111"))
MTl[23] <- list(c("000","0010","0011","0100","0101","0110","0111"))
MTl[24] <- list(c("0000","0001","0010","0011","0100","0101","011"))
MTl[25] <- list(c("0000","0001","0010","0011","0100","0101","0110","0111"))
MTl[26] <- list(c("00","0100","0101","0110","0111"))

MT <- MT %>% mutate("ts" = MTl) #MTl is the tree structure

# up_trainf <- dataul[[length(dataul)]]

#debugen
v1n <- data.frame("i","n")
v1sidx <- data.frame("i","idx")
v2sn <-  matrix(ncol = 3,nrow = 1)
v3idxa <- data.frame("i","idxa")
PrvMl <- data.frame("i","PrvM")
PrvMul <- data.frame("i","PrvM")

functionNames <- c("LL","logLikGrad")
clusterExport(cl=cl, varlist = c("functionNames", functionNames),envir=environment())
print("test_wang")

# datal <- list()
# dataul <- list()
##MH algorithms
#begin from no boosting
#nbatch: the maximum number of iteration
#up_trainf: data frame for attributes and selection outcome
#up_test: test data frame, used to evaluate PrvM when training the model
#par_root: parameters at the root
#alpha, beta: parameter for pr splitting
#gpv: the random integer vector
#av: the available variable vector
#wv: weighing vector

GMT_MH <- function(up_trainf,nbatch,par_root, alpha, beta, MT,gpv,av,wv,K,maxit)
{
  #at each iteration, terminal nodes may have 
  #grow, prune, 
  #after each operation, up_trainf should be updated
  up_train_r <- up_trainf #data frame at root
  nt_r <- up_trainf$nt #number of terminal node
  tdv_r <- up_trainf$tdv #depth of terminal node
  entr_r <- up_trainf$entr #entropy of terminal nodes
  iv_r <- up_trainf$iv #variables used in internal nodes
  Prv_r <- up_train$Prv
  
  MTl <- MT[,2] #get the tree code
  
  #get the parameters at root
  sv <- par_root$sv #selected LCCA
  parameters_i <- par_root$parameters #parameters for 10 LCCA
  pM_i <- par_root$pM #probability for each LCCA, each probability
  
  #update tree parameters
  # nt_i <- nt_r
  # tdv_i <- tdv_r
  # entr_i <- entr_r
  # iv_i <- iv_r
  
  
  #get the transition matrix and stationary prob of 26 trees
  P1_1 <- 1 - alpha*(1+0)^(-beta)
  P1_2 <- 1- P1_1 #1 tree to 2 tree
  PV1 <- c(P1_1,P1_2, rep(0,26-2))
  
  P2_1 <- 0.5 #prune
  
  P2_3 <- 0.5*0.5*alpha*(1+1)^(-beta) #grow
  P2_4 <- 0.5*0.5*alpha*(1+1)^(-beta) #grow
  P2_2 <- 0.5 - (P2_3 + P2_4) #no change
  PV2 <- c(P2_1, P2_2,P2_3,P2_4, rep(0,26-4))
  
  P3_5 <- 0.5*1/3*alpha*(1+1)^(-beta) #grow
  P3_2 <- 0.5 #prune
  P3_6 <- 0.5*1/3*alpha*(1+2)^(-beta) #grow
  P3_7 <- 0.5*1/3*alpha*(1+2)^(-beta) #grow
  P3_3 <- 0.5-P3_5-P3_6-P3_7 #no change
  PV3 <- c(0,P3_2,P3_3,0,P3_5,P3_6,P3_7,rep(0,26-7))
  
  P4_5 <- 0.5*1/3*alpha*(1+1)^(-beta) #grow
  P4_12 <- 0.5*1/3*alpha*(1+2)^(-beta) #grow
  P4_13 <- 0.5*1/3*alpha*(1+2)^(-beta) #grow
  P4_2 <- 0.5
  P4_4 <- 0.5-P4_5-P4_12-P4_13 #grow
  PV4 <- c(0, P4_2, 0, P4_4, P4_5, 0,0,0,0,0,0,P4_12,P4_13,0, rep(0,26-14))
  
  P5_8 <- 0.5*1/4*alpha*(1+2)^(-beta) #grow
  P5_9 <- 0.5*1/4*alpha*(1+2)^(-beta) #grow
  P5_10 <- 0.5*1/4*alpha*(1+2)^(-beta) #grow
  P5_11 <- 0.5*1/4*alpha*(1+2)^(-beta) #grow
  P5_4 <- 0.5*0.5
  P5_3 <- 0.5*0.5
  P5_5 <- 0.5 - P5_8 - P5_9 - P5_10 - P5_11 #no change
  PV5 <- c(rep(0,2),P5_3,P5_4,P5_5, 0, 0, P5_8, P5_9,P5_10,P5_11,rep(0,3), rep(0,26-14))
  
  #from tree 6, it cannot grow fully 
  P6_8 <- 0.5*1/2*alpha*(1+1)^(-beta) #grow
  P6_3 <- 0.5#prune
  P6_14 <- 0.5*1/2*alpha*(1+2)^(-beta) #grow
  # P6_6 <- 0.5*(1 - 1/4*(alpha*(1+3)^(-beta) + alpha*(1+3)^(-beta) + alpha*(1+2)^(-beta) + alpha*(1+1)^(-beta)))
  P6_6 <- 0.5*(1 - 1/2*(alpha*(1+2)^(-beta) + alpha*(1+1)^(-beta)))
  PV6 <- c(0,0,P6_3,0,0,P6_6,0,P6_8,rep(0,5),P6_14, rep(0,26-14))
 
  # P7_9 <- 0.5*1/4*alpha*(1+1)^(-beta)
  P7_9 <- 0.5*1/2*alpha*(1+1)^(-beta)
  P7_3 <- 0.5 #prune
  # P7_14 <- 0.5*1/4*alpha*(1+2)^(-beta) #grow
  P7_14 <- 0.5*1/2*alpha*(1+2)^(-beta)
  # P7_7 <- 0.5*(1 - 1/4*(alpha*(1+2)^(-beta) + alpha*(1+3)^(-beta) + alpha*(1+3)^(-beta) + alpha*(1+1)^(-beta)))
  P7_7 <- 0.5*(1 - 1/2*alpha*(1+1)^(-beta) - 1/2*alpha*(1+2)^(-beta))
  PV7 <- c(rep(0,2),P7_3,rep(0,3),P7_7,0,P7_9,rep(0,4),P7_14, rep(0,26-14))
  
  P8_6 <- 0.5*0.5 #prune
  P8_5 <- 0.5*0.5 #prune
  P8_15 <- 0.5*1/3*alpha*(1+2)^(-beta)
  P8_16 <- 0.5*1/3*alpha*(1+2)^(-beta)
  P8_17 <- 0.5*1/3*alpha*(1+2)^(-beta)
  # P8_8 <- 0.5*(1 - 1/5*(alpha*(1+3)^(-beta) + alpha*(1+3)^(-beta) + alpha*(1+2)^(-beta) + alpha*(1+2)^(-beta) + alpha*(1+2)^(-beta)))
  P8_8 <- 0.5 - 0.5*alpha*(1+2)^(-beta)
  PV8 <- c(rep(0,4),P8_5,P8_6,0,P8_8,rep(0,6), P8_15,P8_16,P8_17, rep(0,26-17))
  
  P9_5  <- 0.5*0.5
  P9_7 <- 0.5*0.5
  # P9_9 <- 0.5*(1 - 1/5*(alpha*(1+2)^(-beta) + alpha*(1+3)^(-beta) + alpha*(1+3)^(-beta) + alpha*(1+2)^(-beta) + alpha*(1+2)^(-beta)))
  P9_15 <- 0.5*1/3*alpha*(1+2)^(-beta)
  P9_19 <- 0.5*1/3*alpha*(1+2)^(-beta)
  P9_20 <- 0.5*1/3*alpha*(1+2)^(-beta)
  P9_9 <- 0.5 - 0.5*alpha*(1+2)^(-beta)
  PV9 <- c(rep(0,4),P9_5,0,P9_7,0,P9_9,rep(0,5),P9_15, 0,0,0,P9_19,P9_20, rep(0,26-20))
  
  P10_5 <- 0.5*0.5
  P10_12 <- 0.5*0.5
  # P10_10 <- 0.5*(1 - 1/5*(alpha*(1+2)^(-beta) + alpha*(1+2)^(-beta) + alpha*(1+3)^(-beta) + alpha*(1+3)^(-beta) + alpha*(1+2)^(-beta)))
  P10_16 <- 0.5*1/3*alpha*(1+2)^(-beta)
  P10_20 <- 0.5*1/3*alpha*(1+2)^(-beta)
  P10_18 <- 0.5*1/3*alpha*(1+2)^(-beta)
  P10_10 <- 0.5 - 0.5*alpha*(1+2)^(-beta)
  PV10 <- c(rep(0,4),P10_5,rep(0,4),P10_10,0,P10_12,0,0,0,P10_16,0,P10_18,0,P10_20,rep(0,26-20))
  
  P11_5 <- 0.5*0.5
  P11_13 <- 0.5*0.5
  P11_17 <- 0.5*1/3*alpha*(1+2)^(-beta)
  P11_18 <- 0.5*1/3*alpha*(1+2)^(-beta)
  P11_19 <- 0.5*1/3*alpha*(1+2)^(-beta)
  
  # P11_11 <- 0.5*(1 - 1/5*(alpha*(1+2)^(-beta) + alpha*(1+2)^(-beta) + alpha*(1+2)^(-beta) + alpha*(1+3)^(-beta) + alpha*(1+3)^(-beta)))
  # P11_11 <- 0.5 - 0.5*alpha*(1+2)^(-beta)
  P11_11 <- 0.5 - 0.5*alpha*(1+2)^(-beta)
  PV11 <- c(rep(0,4),P11_5,rep(0,5),P11_11,0,P11_13,0,0,0,P11_17,P11_18,P11_19, rep(0,26-19))
  
  P12_4 <- 0.5 #prune
  # P12_10 <- 0.5*1/4*alpha*(1+1)^(-beta)
  P12_10 <- 0.5*0.5*alpha*(1+1)^(-beta)
  P12_26 <- 0.5*0.5*alpha*(1+2)^(-beta)
  # P12_12 <- 0.5*(1 - 1/4*(alpha*(1+1)^(-beta) + alpha*(1+3)^(-beta) + alpha*(1+3)^(-beta) + alpha*(1+2)^(-beta)))
  P12_12 <- 0.5 - (0.25*alpha*(1+1)^(-beta) + 0.25*alpha*(1+2)^(-beta))
  
  PV12 <- c(rep(0,3),P12_4,rep(0,5),P12_10,0,P12_12,0,0, rep(0,25-14),P12_26)
  
  P13_4 <- 0.5
  # P13_11 <- 0.5*1/4*alpha*(1+1)^(-beta)
  P13_11 <- 0.5*0.5*alpha*(1+1)^(-beta)
  P13_26 <- 0.5*0.5*alpha*(1+2)^(-beta)
  # P13_13 <- 0.5*(1 - 1/4*(alpha*(1+1)^(-beta) + alpha*(1+2)^(-beta) + alpha*(1+3)^(-beta) + alpha*(1+3)^(-beta)))
  P13_13 <- 0.5 - (0.25*alpha*(1+1)^(-beta) + 0.25*alpha*(1+2)^(-beta))
  PV13 <- c(rep(0,3),P13_4,rep(0,6),P13_11,0,P13_13,0,rep(0,25-14),P13_26)
  
  
  P14_7 <- 0.5*0.5
  P14_6 <- 0.5*0.5
  # P14_14 <- 0.5*(1 - 1/5*(alpha*(1+1)^(-beta) + alpha*(1+3)^(-beta) + alpha*(1+3)^(-beta) + alpha*(1+3)^(-beta)))
  P14_15 <- 0.5*alpha*(1+1)^(-beta)
  P14_14 <- 0.5 - 0.5*alpha*(1+1)^(-beta)
  PV14 <- c(rep(0,5),P14_6,P14_7,rep(0,6),P14_14,P14_15,rep(0,26-15))
  
  
  P15_14 <- 0.5*1/3
  P15_8 <- 0.5*1/3
  P15_9 <- 0.5*1/3
  P15_24 <- 0.5*0.5*alpha*(1+2)^(-beta)
  P15_21 <- 0.5*0.5*alpha*(1+2)^(-beta)
  P15_15 <- 0.5 - 0.5*alpha*(1+2)^(-beta)
  PV15 <- c(rep(0,7),P15_8,P15_9,rep(0,4),P15_14,P15_15,rep(0,5),P15_21,0,0,P15_24,0,0)
  
  P16_10 <- 0.5*0.5
  P16_8 <- 0.5*0.5
  P16_24 <- 0.5*0.5*alpha*(1+2)^(-beta)
  P16_22 <- 0.5*0.5*alpha*(1+2)^(-beta)
  P16_16 <- 0.5 - 0.5*alpha*(1+2)^(-beta)
  PV16 <- c(rep(0,7),P16_8, 0, P16_10,rep(0,5),P16_16,rep(0,5),P16_22,0,P16_24,0,0)
  
  P17_11 <- 0.5*0.5
  P17_8 <- 0.5*0.5
  P17_21 <- 0.5*0.5*alpha*(1+2)^(-beta)
  P17_22 <- 0.5*0.5*alpha*(1+2)^(-beta)
  P17_17 <- 0.5 - 0.5*alpha*(1+2)^(-beta)
  PV17 <- c(rep(0,7),P17_8,0,0,P17_11,rep(0,5),P17_17,rep(0,3),P17_21,P17_22,rep(0,26-22))
  
  P18_11 <- 0.5*1/3
  P18_10 <- 0.5*1/3
  P18_26 <- 0.5*1/3
  P18_22 <- 0.5*0.5*alpha*(1+2)^(-beta)
  P18_23 <- 0.5*0.5*alpha*(1+2)^(-beta)
  P18_18 <- 0.5 - 0.5*alpha*(1+2)^(-beta)
  PV18 <- c(rep(0,9),P18_10,P18_11,rep(0,6),P18_18,rep(0,3),P18_22,P18_23,0,0,P18_26)
  
  P19_11 <- 0.5*0.5
  P19_10 <- 0.5*0.5
  P19_21 <- 0.5*0.5*alpha*(1+2)^(-beta)
  P19_23 <- 0.5*0.5*alpha*(1+2)^(-beta)
  P19_19 <- 0.5 - 0.5*alpha*(1+2)^(-beta)
  PV19 <- c(rep(0,9),P19_10,P19_11,rep(0,7),P19_19,0,P19_21,0,P19_23,rep(0,26-23))
  
  P20_9 <- 0.5*0.5
  P20_10 <- 0.5*0.5
  P20_23 <- 0.5*0.5*alpha*(1+2)^(-beta)
  P20_24 <- 0.5*0.5*alpha*(1+2)^(-beta)
  P20_20 <- 0.5 - 0.5*alpha*(1+2)^(-beta)
  PV20 <- c(rep(0,8), P20_9,P20_10,rep(0,9),P20_20,0,0,P20_23,P20_24,rep(0,26-24))
  
  P21_15 <- 0.5*1/3
  P21_17 <- 0.5*1/3
  P21_19 <- 0.5*1/3
  P21_25 <- 0.5*alpha*(1+2)^(-beta)
  P21_21 <- 0.5 - 0.5*alpha*(1+2)^(-beta)
  PV21 <- c(rep(0,14),P21_15,0,P21_17,0,P21_19,0,P21_21,rep(0,3),P21_25,0)
  
  P22_16 <- 0.5*1/3
  P22_17 <- 0.5*1/3
  P22_18 <- 0.5*1/3
  P22_25 <- 0.5*alpha*(1+2)^(-beta)
  P22_22 <- 0.5 - 0.5*alpha*(1+2)^(-beta)
  PV22 <- c(rep(0,15),P22_16,P22_17,P22_18,rep(0,3),P22_22,0,0,P22_25,0)
  
  P23_18 <- 0.5*1/3
  P23_19 <- 0.5*1/3
  P23_20 <- 0.5*1/3
  P23_25 <- 0.5*alpha*(1+2)^(-beta)
  P23_23 <- 0.5 - 0.5*alpha*(1+2)^(-beta)
  PV23 <- c(rep(0,17),P23_18,P23_19,P23_20,0,0,P23_23,0,P23_25,0)
  
  P24_15 <- 0.5*1/3
  P24_16 <- 0.5*1/3
  P24_20 <- 0.5*1/3
  P24_25 <- 0.5*alpha*(1+2)^(-beta)
  P24_24 <- 0.5 - 0.5*alpha*(1+2)^(-beta)
  PV24 <- c(rep(0,14),P24_15,P24_16,rep(0,3),P24_20,rep(0,3),P24_24,P24_25,0)
  
  P25_21 <- 0.5*1/4
  P25_22 <- 0.5*1/4
  P25_23 <- 0.5*1/4
  P25_24 <- 0.5*1/4
  P25_25 <- 0.5
  PV25 <- c(rep(0,20),P25_21,P25_22,P25_23,P25_24,P25_25,0)
  
  P26_13 <- 0.5*0.5
  P26_12 <- 0.5*0.5
  P26_18 <- 0.5*alpha*(1+1)^(-beta)
  P26_26 <- 0.5 - 0.5*alpha*(1+1)^(-beta)
  PV26 <- c(rep(0,11),P26_12,P26_13,rep(0,4),P26_18,rep(0,7),P26_26)
  
#   save(PV1,PV2,PV3,PV4,PV5,PV6,PV7,PV8,PV9,PV10,PV11,PV12,PV13,PV14,PV15,PV16,PV17,PV18,PV19,
# PV20,PV21,PV22,PV23,PV24,PV25,PV26, file = "/users/shangbo/rproject/PV1.RData")

  #construct Markov chain
  TM <- rbind("[1,]" = PV1,"[2,]" = PV2,"[3,]" = PV3, "[4,]" = PV4, "[5,]" = PV5, "[6,]" = PV6
              , "[7,]" = PV7, "[8,]" = PV8, "[9,]" = PV9, "[10,]" = PV10, 
              "[11,]" = PV11, "[12,]" = PV12, "[13,]" = PV13, "[14,]" = PV14, "[15,]" = PV15, 
              "[16,]" = PV16, "[17,]" = PV17, "[18,]" = PV18, "[19,]" = PV19, "[20,]" = PV20, 
              "[21,]" = PV21, "[22,]" = PV22, "[23,]" = PV23, "[24,]" = PV24, "[25,]" = PV25, 
              "[26,]" = PV26)
  
  statenames <- c("T1","T2","T3","T4","T5", "T6","T7","T8","T9","T10","T11","T12","T13","T14",
                  "T15","T16","T17","T18","T19","T20","T21","T22","T23","T24","T25","T26")
  
  TMM <- matrix(TM, ncol = 26, byrow = F, dimnames = list(statenames,statenames))
  
  mcTM <- new("markovchain", states = statenames, transitionMatrix = TMM)
  
  SS <- steadyStates(mcTM) #stationary probability
  
  # T_chain <- markovchainSequence(n = 2000, mcTM,include.t0 = FALSE)
  
  ibv <- c(1:nbatch)
  
  #assign nbatch random integer with probability of 0.5
  # gpv <- rbinom(nbatch, 1, prob = 0.5)
  tree_ini <- up_trainf$Tree
  # treev <<- rep(0,nbatch)
  treev <<- "T1"
  
  LLv <<- NULL #the LL for m and n
  Treevv <<- NULL #tree for m and n
  av_mnlv <<- NULL #the available variables of mnl model for m and n
  Prvv <<- NULL #store accumulated probability for each observation
  
  # treev[1] <- tree_ini
  # treev <<- c(tree_ini) #tree sequence generated by MH
  i_utv <- c(substr(tree_ini,2,2))
  
  PrvM <- as.data.frame(Reduce(rbind, up_trainf[[1]]$Prv))
  
  # PrvM_test <- as.data.frame(Reduce(rbind, up_test$Prv)) #get the probabiltiy vector for up_test
  
  LLM <<- matrix(rep(0,nrow(up_trainf[[1]])*nbatch),nrow = nrow(up_trainf[[1]])) #matrix, store the loglihood ratio
  
  LLvv <<- NULL #store the log-likelihood vector for each iteration
  
  mle_parv <<- NULL #store the parameter vector for the class based MNL model for each tree
  
  av_mnll <<- NULL #store the variable name vector for the class based MNL model for each tree
  
  dataul <<- vector(mode = "list", length = 500)
  
  dataull <<-  NULL #store the historical up_trainf
  par_mnlv <<- NULL #store the historical par_mnl
  av_MNLv <<- NULL #store the historical av_MNL
  PrvMv <<- NULL
  
  # dataul <- NULL

  llmv <<- NULL
  llnv <<- NULL
  
  tmv <<- NULL
  tnv <<- NULL
  
  av_ini <- av #compensatory attributes
  coe_ini <- rep(1, (length(av_ini)+1)*4)
  nt_ini <- 1
  
  av_mnl <- av_ini
  
  ##get the parameter for the initial tree model with only one root 
  mle_m2 <- optimParallel(par = coe_ini, gr = logLikGrad, nt = nt_ini, av_mnl = av_mnl, data = up_trainf[[1]],PrvM = PrvM, wv = wv, fn = LL, method = "L-BFGS-B", control = list(fnscale = -1), maxit=maxit)
  
  mle_m2_out <<- mle_m2 #output the parameters for mle_m2
  
  av_del <- Get_av_del(mle_m2$par, av_mnl, PrvM, up_trainf[[1]], wv, K)
  
  #get the Loglikelihood value
  LL_ini <- LL(mle_m2$par, av_mnl, up_trainf$nt, up_trainf[[1]],PrvM,wv)
  LL_m <- LL_ini
  
  up_trainf[[1]]$SP <- SP
  up_trainf$LL <- LL_m
  
  av_MNL_m <- av_mnl
  par_mnl_m <- mle_m2$par
  
  Treevv <<- cbind(Treevv, up_trainf$Tree)
  dataull <<- cbind(dataull, list(up_trainf))
  par_mnlv <<- cbind(par_mnlv, list(par_mnl_m))
  av_MNLv <<- cbind(av_MNLv, list(av_MNL_m))
  PrvMv <<- cbind(PrvMv, list(PrvM))
  
  for (i in ibv) {
    
    # up_trainf <- up_trainft
    # nt_i
    # tdv_i 
    # entr_i
    # iv_i
    print(paste0("the ",i, "-th loop"))
    
    # print("av is:")
    # print(av)
    
    nt_i <- up_trainf$nt
    tdv_i <- up_trainf$tdv
    Treei <- up_trainf$Tree #"T1" is the root
    entr_i <-up_trainf$entr 
    iv_i <- up_trainf$iv
    par_i <- up_trainf$par
    Prvl_i <- up_trainf$Prvt #the accumulated probability at all terminal nodes
    
    # colnames(Prvl_i) <- "Pr1"
    
    up_trainft <- up_trainf
    
    # up_trainft_temp9 <<- up_trainft
    
# up_trainft_temp9 <- up_trainft   
    tvi <- c(1:nt_i)
    
    #index vector for MTl
    
    #get the terminal node vector
    tv <- MTl[as.numeric(unlist(strsplit(Treei,"T"))[2])][[1]]
    
    # print("hello1")

    #randomly choose among Grow or Prune process (0.5,0.5)
    if(gpv[i]==1 | nt_i == 1) { #grow for 1 or the root
      print("hello2")
      #grow only for the node whose depth is less than 3
      idxi <- which(tdv_i<3)
      # print("bo1")
      # ev_i <- entr_i[idxi]
      # tv_i <- tv[idxi]
      tvi_i <<- tvi[idxi]
      
      
      if(isempty(tvi_i) | (length(av_mnl) <= K))
      next
      # print("bo2")
      #uniquely choose one terminal node and grow, n is the original index
      # n <- sample(tvi_i,1,replace = F) 
      n <- sample(length(tvi_i),1,replace = T) 
      n <- tvi_i[n]
      # print("bo3")
      # print(paste0("wang",1))
      # print("hello3")
# v1n <<- rbind(v1n,c(i,n))
      
      #grow n-th terminal node and get the new tree code
      # new_tc <- paste0(tv[n],"0 ", tv[n],"1")
      new_tc <- paste0(tv[n],"0 ", tv[n],"1")
      
      # print("hello4")
      # print("bo4")
      tv[n] <- new_tc
      # print("hello5")
      # print("bo5")
      # U_tc <- unlist(strsplit(gsub(tv[n], new_tc, tv),split = " "))
      U_tc <- unlist(strsplit(tv,split = " "))
      # print("hello6")
      # print("bo6")
      # print(paste0("wang",2))   
      #compare U_tc to MTl and determine the index of tree
      bv <- unlist(lapply(MTl, function(x){isTRUE(all.equal(U_tc,x))}))
      # print("hello7")
      # print("bo7")
      i_ut<- which(bv) #index of possible updated tree for grow
      # print("hello8")
      
      #check if there is any model tree which has been experienced
      up_trainft$Tree <- paste0("T",i_ut)
      
      # print("hello9")
      
      if(up_trainft$Tree %in% Treevv) #the updated tree in the Treevv
      # if(0) #the updated tree in the Treevv
      {
        print("the updated tree is in the Treevv")
        idxt <- which(Treevv == up_trainft$Tree)
        up_trainft <- dataull[[idxt[length(idxt)]]]
        
        LL_n <- up_trainft$LL
        par_mnl_n <- par_mnlv[[idxt[length(idxt)]]]
        av_MNL_n <- av_MNLv[[idxt[length(idxt)]]]
        
        #update PrvM
        tryCatch(
          {
            PrvMu <- PrvMv[[idxt[length(idxt)]]]
          },
          
          error = function(c){
            save(PrvMv, file = "PrvMv_LCCA.RData")
            save(dataull, file = "dataull_LCCA.RData")
            save(Treevv, file = "Treevv_LCCA.RData")
            save(par_mnlv, file = "par_mnlv_LCCA.RData")
            save(av_MNLv, file = "av_MNLv_LCCA.RData")
            save(up_trainft, file = "up_trainft_LCCA.RData")
            
            save(idxt, file = "idxt_LCCA.RData")
            save(i_ut, file = "i_ut_LCCA.RData")
            
            sfsdfsfdsfsdfsfd
          }
        )
      
      }
      else{ #the updated tree is not in the Treevv
        ##get the optimum attributes for node splitting
        print("the updated tree is NOT in the Treevv")
        
        up_trainft$nt <- up_trainft$nt + 1 #update the number of terminal nodes
        
        av_MNL <- setdiff(av_mnl, av_del) #av_mnl and av_del should be updated in each loop
        # print("shang1")
        param_ini <- c(rep(1,length(av_del)+1), rep(1, (length(av_MNL)+1)*4*up_trainft$nt))
        
        tryCatch(
          {
            mle_n <- optimParallel(par = param_ini, gr = LL_split_grad, av_del = av_del, av_MNL = av_MNL, n = n,
                                   data = up_trainft[[1]], PrvM = PrvM, wv = wv, fn = LL_split, method = "L-BFGS-B", control = list(fnscale = -1,maxit = maxit))
          },
          
          error = function(c){
            save(param_ini, file = "param_ini_debug.RData")
            save(av_del, file = "av_del_debug.RData")
            save(av_MNL, file = "av_MNL_debug.RData")
            save(n, file = "n_debug.RData")
            save(up_trainft, file = "up_trainft_debug.RData")
            save(PrvM, file = "PrvM_debug.RData")
            save(wv, file = "wv_debug.RData")
            
            mle_n <- optim(par = param_ini, gr = LL_split_grad, av_del = av_del, av_MNL = av_MNL, n = n,
                                   data = up_trainft[[1]], PrvM = PrvM, wv = wv, fn = LL_split, method = "L-BFGS-B", control = list(fnscale = -1,maxit = maxit))
            
          }
          
        )
        
        print("optimParallel is finished and succesuffly!!!")
        
        
       
        #get the loglikihood ratio 
        LL_n <-  LL_split(mle_n$par, av_del,av_MNL, n, up_trainft[[1]], PrvM,wv)
        up_trainft[[1]]$SP <- SP #selection probability
        up_trainft$LL <- LL_n
       
        
        #update the parameters for up_trainft
        up_trainft$id <- c(1:up_trainft$nt)
        
        td1_temp <- ifelse(n-1>=1,list(up_trainft$tdv[1:(n-1)]),NA)
        td2_temp <- ifelse(n+1<=length(up_trainft$tdv), list(up_trainft$tdv[(n+1):length(up_trainft$tdv)]),NA)
        up_trainft$tdv <- c(unlist(td1_temp),rep(up_trainft$tdv[n]+1,2),unlist(td2_temp))
        up_trainft$tdv <- up_trainft$tdv[!is.na(up_trainft$tdv)]
        
        svv <- av_del
        paruv <- mle_n$par[1:(1+length(av_del))]
        par_mnl <- mle_n$par[(length(av_del)+2):length(mle_n$par)]
        par_mnl_n <- par_mnl
        av_MNL_n <- av_MNL
        
        #update the Prv in up_train for the left and right children (m and n)
        data <- as.data.frame(up_trainft[[1]][,which(colnames(up_trainft[[1]]) %in% svv)])
        # data_test <- as.data.frame(up_test[,which(colnames(up_test) %in% svv)])
        
        dataM <- as.matrix(data %>% mutate(I = rep(1,nrow(data))))
        # dataM_test <- as.matrix(data_test %>% mutate(I = rep(1,nrow(data_test))))
        paruM <- as.matrix(paruv,nrow = length(paruv))
        PR <- 1/(1 + exp(dataM %*% paruM)) #right splitting probabiltiy
        PL <- 1- PR #left splitting probabiltiy
       
        Prvm <- PrvM[,n] * rep(PL,each = 5)
        Prvn <- PrvM[,n] * rep(PR,each = 5)
       
        
        Prv_temp1 <- ifelse(n-1>=1,list(PrvM[,1:(n-1)]),NA)
        Prv_temp2 <- ifelse(n+1<=ncol(PrvM),list(PrvM[,(n+1):ncol(PrvM)]),NA)
        
        temp <- cbind(Prv_temp1[[1]],Prvm,Prvn,Prv_temp2[[1]])
        temp <- as.data.frame(temp[,which(colSums(is.na(temp)) == 0)])
        temp <- temp %>% mutate("id" = as.factor(rep(1:length(PL),each = 5)))
    
        #update entropy
        tempM <- aggregate(temp,by = list(temp$id), FUN = mean)
        tempM <- tempM[,which(colnames(tempM)%in% c("Prvm","Prvn"))]
        
        up_trainC <- cbind("Class" = up_trainft[[1]]$Class,tempM)
        up_trainC$Class <- as.numeric(as.character(up_trainC$Class))
        up_trainCA <- aggregate(up_trainC, by = list(up_trainC$Class), FUN = sum)
        Freqm <- up_trainCA$Prvm/sum(up_trainCA$Prvm)
        Freqn <- up_trainCA$Prvn/sum(up_trainCA$Prvn)
        Entrm <- list(c(entr_i[[n]],sum(-log2(Freqm)*Freqm)))
        Entrn <- list(c(entr_i[[n]],sum(-log2(Freqn)*Freqn)))
        Entr_temp1 <- ifelse(n-1>=1,list(entr_i[1:(n-1)]),NA)
        Entr_temp2 <- ifelse(n+1<=length(entr_i), list(entr_i[(n+1):length(entr_i)]),NA)
        colnames(temp)[which(colnames(temp)!="id")] <- paste0("V",c(1:(ncol(temp)-1)))
        
        templ <- split(temp,temp$id,drop = T)
        
        up_trainft[[1]]$Prv <- as.matrix(lapply(templ,function(x){x[,-which(colnames(x)%in%"id")]}))
        PrvMu <- temp[,-which(colnames(temp)%in%"id")]
       
        up_trainft$entr <- c(Entr_temp1[[1]],Entrm,Entrn,Entr_temp2[[1]])
        up_trainft$entr <- up_trainft$entr[!is.na(up_trainft$entr)]
        
        ivm <- list(cbind(iv_i[[n]], list(svv))) #for the left child node
        ivn <- list(cbind(iv_i[[n]], list(svv))) #for the right child node
        iv_temp1 <- ifelse(n-1>=1,list(iv_i[1:(n-1)]),NA)
        iv_temp2 <- ifelse(n+1<= length(iv_i), list(iv_i[(n+1):length(iv_i)]),NA)
       
        up_trainft$iv <- c(iv_temp1[[1]],ivm,ivn,iv_temp2[[1]])
        up_trainft$iv <- up_trainft$iv[!is.na(up_trainft$iv)]
       
        parm <- list(cbind(par_i[[n]],list(paruv)))
        parn <- list(cbind(par_i[[n]],list(paruv)))
       
        par_temp1 <- ifelse(n-1>=1,list(par_i[1:(n-1)]),NA)
        par_temp2 <- ifelse(n+1<= length(par_i), list(par_i[(n+1):length(par_i)]), NA)  
       
        up_trainft$par <- c(par_temp1[[1]],parm,parn,par_temp2[[1]])
        up_trainft$par <- up_trainft$par[!is.na(up_trainft$par)]
        
        
       
        
        Prvlm <- as.data.frame(Prvl_i[n]) %>% mutate(V = tempM[,1])
        Prvln <- as.data.frame(Prvl_i[n]) %>% mutate(V = tempM[,2])
      
        
        colnames(Prvlm) <- paste0("Pr",c(1:ncol(Prvlm)))
        colnames(Prvln) <- paste0("Pr",c(1:ncol(Prvln)))
        
        # print("bo37")
        
        Prvl_temp1 <- ifelse(n-1>=1, list(Prvl_i[1:(n-1)]), NA)
        Prvl_temp2 <- ifelse(n+1<=length(Prvl_i), list(Prvl_i[(n+1):length(Prvl_i)]), NA)
        up_trainft$Prvt <- c(Prvl_temp1[[1]],list(Prvlm),list(Prvln),Prvl_temp2[[1]])
        up_trainft$Prvt <- up_trainft$Prvt[!is.na(up_trainft$Prvt)]
        # print("shang30")
        
      }
      
    } 
    else #prune
    {
     
      pv <- substr(tv, start = rep(1,length(tv)), stop =  nchar(tv)-1) #code for the parent node
      
    
      if(length(pv)==1){ #root
        next
      }
      
      pv <- as.data.frame(pv) %>% mutate("id" = c(1:length(pv)))
      
      
      pvg <- split(pv,pv$pv) #group the same element
      
    
      #for each group with terminal node pair, prune it
      ln <- unlist(lapply(pvg, function(x){nrow(x)}))
     
      if(isempty(which(ln > 1)))
        next

      ln_idx <- which(ln > 1)
      
      
      lnv <<- ifelse(length(ln_idx)==1,list(c(ln_idx,ln_idx)),list(ln_idx))
      
      
      # sn <- sample(which(ln > 1), 1, replace = T) #selected node pair
      sn <<- sample(unlist(lnv), 1, replace = T) #selected node pair
      
     
      tv[pvg[[sn]]$id] <- as.character(pvg[[sn]]$pv)
      
      
      tv <- tv[-pvg[[sn]]$id[1]] #possible updated tree
      
           
      nv <- pvg[[sn]]$id #index of the selected pair node
      
     
      bv <- unlist(lapply(MTl, function(x){isTRUE(all.equal(tv,x))}))
      i_ut<- which(bv) #the index of new tree, all number is 26
      
      print("word11")
      
      #check if there is model tree which has been experienced
      up_trainft$Tree <- paste0("T",i_ut)
      
      if((up_trainft$Tree %in% Treevv) && length(av_mnl) > K)
      # if(0)
      {
        print("the updated tree is in the Treevv")
        idxt <- which(Treevv == up_trainft$Tree)
        up_trainft <- dataull[[idxt[length(idxt)]]]
        
        LL_n <- up_trainft$LL
        par_mnl_n <- par_mnlv[[idxt[length(idxt)]]]
        av_MNL_n <- av_MNLv[[idxt[length(idxt)]]]
        
        # PrvMu <- PrvMv[[idxt]]
        
        #update PrvM
        tryCatch(
          {
            PrvMu <- PrvMv[[idxt[length(idxt)]]]
          },
          
          error = function(c){
            save(PrvMv, file = "PrvMv_LCCA.RData")
            save(dataull, file = "dataull_LCCA.RData")
            save(Treevv, file = "Treevv_LCCA.RData")
            save(par_mnlv, file = "par_mnlv_LCCA.RData")
            save(av_MNLv, file = "av_MNLv_LCCA.RData")
            save(up_trainft, file = "up_trainft_LCCA.RData")
            save(idxt, file = "idxt_LCCA.RData")
            save(i_ut, file = "i_ut_LCCA.RData")
            
          }
        )
        
      }
      else{
        print("the updated tree is NOT in the Treevv")
        #update the possible prune up_trainf
        up_trainft$nt <- up_trainft$nt - 1
        up_trainft$id <- c(1:up_trainft$nt)
        td1_temp <- ifelse(nv[1]-1>=1,list(up_trainft$tdv[1:(nv[1]-1)]),NA)
        td2_temp <- ifelse(nv[length(nv)]+1<=length(up_trainft$tdv), list(up_trainft$tdv[(nv[length(nv)]+1):length(up_trainft$tdv)]),NA)
        up_trainft$tdv <- c(unlist(td1_temp),up_trainft$tdv[nv[1]]-1,unlist(td2_temp))
        up_trainft$tdv <- up_trainft$tdv[!is.na(up_trainft$tdv)]
        
        Entr_temp1 <- ifelse(nv[1]-1>=1,list(entr_i[1:(nv[1]-1)]),NA)
        Entr_temp2 <- ifelse(nv[length(nv)]+1 <= length(entr_i), list(entr_i[(nv[length(nv)]+1):length(entr_i)]),NA)
      
        entr_n <- entr_i[[nv[1]]] 
        entr_n <- list(entr_n[-length(entr_n)])
        # up_trainft$entr <- rbind(Entr_temp1,entr_n,Entr_temp2)
        up_trainft$entr <- c(Entr_temp1[[1]],entr_n,Entr_temp2[[1]])
        up_trainft$entr <- up_trainft$entr[!is.na(up_trainft$entr)]
        
        
        Prvt <- up_trainft$Prvt #list
        Prvmn <- as.data.frame(Prvt[nv[1]])[,ncol(as.data.frame(Prvt[nv[1]]))-1]
        Prvmnr <- rep(Prvmn,each = 5)
        
        Prv_temp1 <- ifelse(nv[1]-1>=1,list(PrvM[,1:(nv[1]-1)]),NA)
        Prv_temp2 <- ifelse(nv[2]+1<=ncol(PrvM),list(PrvM[,(nv[2]+1):ncol(PrvM)]),NA)
        
        temp <- cbind(Prv_temp1[[1]],Prvmnr,Prv_temp2[[1]])
        temp <- as.data.frame(temp[,which(colSums(is.na(temp)) == 0)])
        temp <- temp %>% mutate("id" = as.factor(rep(1:length(PL),each = 5)))
        
        colnames(temp)[which(colnames(temp)!="id")] <- paste0("V",c(1:(ncol(temp)-1)))

        templ <- split(temp,temp$id,drop = T)
        up_trainft[[1]]$Prv <- as.matrix(lapply(templ,function(x){x[,-which(colnames(x)%in%"id")]}))
       
        #update PrvM
        PrvMu <- as.data.frame(temp[,-which(colnames(temp)%in%"id")])
        
        iv_temp1 <- ifelse(nv[1]-1>=1, list(iv_i[1:(nv[1]-1)]),NA)
        iv_temp2 <- ifelse(nv[2]+1<=length(iv_i), list(iv_i[(nv[length(nv)]+1):length(iv_i)]),NA)
        
        ivmn <- rbind(iv_i[[nv[1]]][-length(iv_i[[nv[1]]])])
        
        up_trainft$iv <- c(iv_temp1[[1]],list(ivmn),iv_temp2[[1]])
        up_trainft$iv <- up_trainft$iv[!is.na(up_trainft$iv)]
        
        par_temp1 <- ifelse(nv[1]-1>=1,list(par_i[1:(nv[1]-1)]),NA) 
        par_temp2 <- ifelse(nv[2]+1<=length(par_i), list(par_i[(nv[length(nv)]+1):length(par_i)]),NA)
        
        parmn <- rbind(par_i[[nv[1]]][-length(par_i[[nv[1]]])])
        
        up_trainft$par <- c(par_temp1[[1]],list(parmn),par_temp2[[1]])
        up_trainft$par <- up_trainft$par[!is.na(up_trainft$par)]
        
        Prvlmn <- Prvt[[nv[1]]][,-ncol(Prvt[[nv[1]]])]
        
        Prvl_temp1 <- ifelse(nv[1]-1>=1, list(Prvt[1:(nv[1]-1)]), NA)
        Prvl_temp2 <- ifelse(nv[2]+1<=length(Prvt), list(Prvt[(nv[2]+1):length(Prvt)]), NA)
        up_trainft$Prvt <- c(Prvl_temp1[[1]],list(Prvlmn),Prvl_temp2[[1]])
        up_trainft$Prvt <- up_trainft$Prvt[!is.na(up_trainft$Prvt)]
        
        ##update the LL
        av_MNL <- c(av_mnl, iv_i[[nv[1]]][[length(iv_i[[nv[1]]])]])
        av_MNL_n <- av_MNL
        
        coe_initV_n <- rep(1,4*(1 + length(av_MNL))*up_trainft$nt)
        
        tryCatch(
          {
            mle_n <- optimParallel(par = coe_initV_n, gr = logLikGrad, nt = up_trainft$nt, 
                                   av_mnl = av_MNL, data = up_trainft[[1]], PrvM = PrvMu, wv = wv,fn = LL, method = "L-BFGS-B", control = list(fnscale = -1,maxit = maxit))
          },
          
          error = function(c){

            save(coe_initV_n, file = "coe_initV_n_LCCA.RData")
            save(up_trainft, file = "up_trainft_LCCA.RData")
            save(av_MNL, file = "av_MNL_LCCA.RData")
            save(PrvMu, file = "PrvMu_LCCA.RData")
            save(wv, file = "wv_LCCA.RData")
            
            save(av_mnl, file = "av_mnl_LCCA.RData")
            save(up_trainf, file = "up_trainf_LCCA.RData")
            save(nv, file = "nv_LCCA.RData")
            
            save(mle_parv, file = "mle_parv_LCCA.RData")
            save(dataul, file = "dataul_LCCA.RData")
            save(av_mnll, file = "av_mnll_LCCA.RData")
            
            save(Treevv, file = "Treevv_LCCA.RData")
            save(PrvMv, file = "PrvMv_LCCA.RData")
            save(av_MNLv, file = "av_MNLv_LCCA.RData")
            save(par_mnlv, file = "par_mnlv_LCCA.RData")
          }
        )
        
        LL_n <- LL(mle_n$par, av_MNL,up_trainft$nt, up_trainft[[1]], PrvMu,wv)
        
        up_trainft[[1]]$SP <- SP
        up_trainft$LL <- LL_n
        
        par_mnl <- mle_n$par
        par_mnl_n <- par_mnl
        
      }
      
    }
    
  
    #transition probability and stationary probability
    m <- as.numeric(unlist(strsplit(Treei,"T"))[2]) #old one

    pvmn <- eval(parse(text = paste0("PV", m, "[", i_ut,"]")))
    pvnm <- eval(parse(text = paste0("PV", i_ut, "[", m,"]")))
    pm <- SS[m]
    pn <- SS[i_ut]

    ratio <- (LL_n - LL_m)/nrow(up_trainf[[1]]) #mean probability ratio
    
    # proba <- (pn*pvnm)/(pm*pvmn)*prod(ratio[(!is.nan(ratio)) & (!is.na(ratio)) & (!is.infinite(ratio))])
    proba <<- (pn*pvnm)/(pm*pvmn)*exp(ratio)
    
    if(proba>=1){
      #accept up_trainft with 100%
      #update up_trainf, PrvM,
      print("bo1")
      up_trainf <- up_trainft
      PrvM <- PrvMu
      # treev <<- c(treev,up_trainft$Tree)
      # treev[i] <<- up_trainft$Tree
      treev <<- cbind(treev,up_trainft$Tree)
      
      # LLv <<- c(LLv,LL_n$)
      # LLM[,i] <<- LL_n
      LLvv <<- cbind(LLvv, LL_n)
      # dataul[[mod(i-1,100)+1]] <<- up_trainft
      dataul[[i]] <<- up_trainft
      # dataul <<- cbind(dataul, list(up_trainft))
      
      # mle_parv <<- cbind(mle_parv,list(mle_n$par))
      # av_mnll <<- cbind(av_mnll, list(av_mnl_n))
      
      mle_parv <<- cbind(mle_parv,list(par_mnl_n))
      av_mnll <<- cbind(av_mnll, list(av_MNL_n))
      
      print("length of mle_parv is: ")
      print(length(mle_parv))
      
      print("length of av_mnll is: ")
      print(length(av_mnll))
      
      if(length(mle_parv) != length(av_mnll))
      {
        print(par_mnl_n)
      }
      
      av_mnl <- av_MNL_n
      LL_m <- LL_n
      
      #update av_del
      # LL(par_mnl_n, av_mnl, nt = ncol(PrvM), data = up_trainft[[1]], PrvM, wv)
      # av_del <- Get_av_del(par_mnl_n, av_mnl, PrvM, up_trainft[[1]], wv, K)
      # 
      # print("av_mnl updated is: ")
      # print(av_mnl)
      # 
      # print("av_del updated is: ")
      # print(av_del)
      
      tryCatch(
        {
          
          # LL(par_mnl_n, av_mnl, nt = ncol(PrvM), data = up_trainft[[1]], PrvM, wv)
          av_del <- Get_av_del(par_mnl_n, av_mnl, PrvM, up_trainft[[1]], wv, K)
          
          print("av_mnl updated is: ")
          print(av_mnl)
          
          print("av_del updated is: ")
          print(av_del)
          
          print("Tree is: ")
          print(up_trainft$Tree)
          
          print("variabls used for spliting are:")
          print(unique(unlist(up_trainft$iv)))

        },

        error = function(c){
          save(par_mnl_n, file = "par_mnl_n_LCCA.RData")
          save(av_mnl, file = "av_mnl_LCCA.RData")
          save(PrvM, file = "PrvM_LCCA.RData")
          save(up_trainft, file = "up_trainft_LCCA.RData")
          
          save(mle_parv, file = "mle_parv_LCCA.RData")
          save(dataul, file = "dataul_LCCA.RData")
          save(av_mnll, file = "av_mnll_LCCA.RData")
          save(dataull, file = "dataull_LCCA.RData")
          
          save(Treevv, file = "Treevv_LCCA.RData")
          

        }
      )
      
      par_mnl_m <- par_mnl_n
      av_MNL_m <- av_MNL_n
      
      
      
      # dataul <<- c(dataul,list(up_trainft))
    }
    else{
      #accept up_trainft with proba
     
      treec <- c("update tree","old tree")
      # set.seed(123)
      idxa <- sample(length(treec),1,prob = c(proba,abs(1-proba)),replace = T)
      if(idxa == 1){
        up_trainf <- up_trainft
        PrvM <- PrvMu
        # treev <<- c(treev,up_trainft$Tree)
        # treev[i] <<- up_trainft$Tree
        treev <<- cbind(treev,up_trainft$Tree)
        # LLM[,i] <<- LL_n
        LLvv <<- cbind(LLvv, LL_n)
        # dataul[[mod(i-1,100)+1]] <<- up_trainft
        dataul[[i]] <<- up_trainft
        # dataul <<- cbind(dataul, list(up_trainft))
        
        mle_parv <<- cbind(mle_parv,list(par_mnl_n))
        av_mnll <<- cbind(av_mnll,list(av_MNL_n))
        
        print("length of mle_parv is: ")
        print(length(mle_parv))
        
        print("length of av_mnll is: ")
        print(length(av_mnll))
        
        if(length(mle_parv) != length(av_mnll))
        {
          print(par_mnl_n)
        }
        
        av_mnl <- av_MNL_n
        LL_m <- LL_n
        # dataul <<- c(dataul,list(up_trainft))
        
        #update av_del
        
        tryCatch(
          {
            # LL(par_mnl_n, av_mnl, nt = ncol(PrvM), data = up_trainft[[1]], PrvM, wv)
            av_del <- Get_av_del(par_mnl_n, av_mnl, PrvM, up_trainft[[1]], wv, K)
            
            print("av_mnl updated is: ")
            print(av_mnl)
            
            print("av_del updated is: ")
            print(av_del)
            
            print("Tree is: ")
            print(up_trainft$Tree)
            
            print("variabls used for spliting are:")
            print(unique(unlist(up_trainft$iv)))
          },
          
          error = function(c){
            save(par_mnl_n, file = "par_mnl_n_LCCA.RData")
            save(av_mnl, file = "av_mnl_LCCA.RData")
            save(PrvM, file = "PrvM_LCCA.RData")
            save(up_trainft, file = "up_trainft_LCCA.RData")
            
            save(mle_parv, file = "mle_parv_LCCA.RData")
            save(dataul, file = "dataul_LCCA.RData")
            save(av_mnll, file = "av_mnll_LCCA.RData")
            save(dataull, file = "dataull_LCCA.RData")
            
            save(Treevv, file = "Treevv_LCCA.RData")
            
          }
        )
        
        par_mnl_m <- par_mnl_n
        av_MNL_m <- av_MNL_n
        
        
      }
      else{
        # treev <<- c(treev,up_trainf$Tree)
        # treev[i] <<- up_trainf$Tree
        treev <<- cbind(treev,up_trainf$Tree)
        # LLM[,i] <<- LL_m
        LLvv <<- cbind(LLvv, LL_m)
        # dataul[[mod(i-1,100)+1]] <<- up_trainf
        dataul[[i]] <<- up_trainf
        # dataul <<- cbind(dataul, list(up_trainf))
        
  
        mle_parv <<- cbind(mle_parv,list(par_mnl_m))
        av_mnll <<- cbind(av_mnll,list(av_mnl))
        
        print("length of mle_parv is: ")
        print(length(mle_parv))
        
        print("length of av_mnll is: ")
        print(length(av_mnll))
        
        if(length(mle_parv) != length(av_mnll))
        {
          print(par_mnl_m)
        }
        
        print("av_mnl updated is: ")
        print(av_mnl)
        
        print("av_del updated is: ")
        print(av_del)
        
        print("Tree is: ")
        print(up_trainf$Tree)
        
        print("variabls used for spliting are:")
        print(unique(unlist(up_trainf$iv)))
        
        # av_mnl <- av_MNL_m
        # dataul <<- c(dataul,list(up_trainf))
      }
      # sample(ncol(svsu),10,prob = Pru, replace = T)``
    }
  
    
    #update av_mnl, 
    
    #store av_mnl_m, av_mnl_n, up_trainf$Tree, up_trainft$Tree, PrvM, PrvMu, LL_m and LL_n for comparing
    # LLv <<- cbind(LLv,LL_m,LL_n)
    Treevv <<- cbind(Treevv, up_trainft$Tree)
    # Prvv <<- cbind(Prvv,list(PrvM),list(PrvMu))
    # av_mnlv <<- cbind(av_mnlv, list(av_mnl_m),list(av_mnl_n))
    dataull <<- cbind(dataull, list(up_trainft))
    
    par_mnlv <<- cbind(par_mnlv, list(par_mnl_n))
    
    av_MNLv <<- cbind(av_MNLv, list(av_MNL_n))
    
    PrvMv <<- cbind(PrvMv, list(PrvMu))
    
  }
  
  return(treev)
}


print(current_filename())

alpha <- 0.95
beta <- 1
nbatch <- 100
M <- 2 #the number of attributes per LCCA
maxit <- 300
K <- 3 #

print("wangshangbo1")
##for ada boost algorithm
##parameters for ada boost
Nt <- 5#the number of iterations
wv <- rep(1/nrow(up_trainf[[1]]),nrow(up_trainf[[1]])) #the initial weighting 
Ntv <- c(1:Nt)
rou_m <- 1 #initial rou
P_k_n_m <- rep(0,nrow(up_trainf[[1]])) #initial Pknm:   Pknm = rou_1*q_kn_1 + rou_2*q_kn_2 + ... + rou_m*q_kn_m
q_k_n_m <- rep(0,nrow(up_trainf[[1]])) #intital q_kn_m
print("wangshangbo2")
#add weighting vector to up_trainf
# up_trainf <- up_trainf %>% 
#   mutate(wv = rep(wv[1],nrow(up_trainf[[1]])))

rou_mv <- NULL
wvv <- wv
P_k_n_mv <- P_k_n_m
q_k_n_mv <- q_k_n_m
print("wangshangbo3")

datallSL <- NULL #list for storage of dataul
mle_parvL <- NULL #list for storage of mle_parv
av_mnllL <- NULL #list for storeage of av_mnll

mle_m2L <- NULL #list for storeage of mle_m2

up_trainf_temp <- up_trainf

#the initial parameters for model tree at one root

for (idxi in Ntv) { #for each iteration, trees are determined by MCMC, rou is calculated and 
  #weighing vector for the next iteration is updated
  
  #add weighting vector to up_trainf
  
  # print("nrow of up_train is:")
  # 
  # print(nrow(up_trainf[[1]]))
  
  print(paste0("this is ", idxi, "-th adaboost"))
  
  # set.seed(5)
  gpv <- rbinom(nbatch, 1, prob = 0.5)
  
  up_trainf <- up_trainf_temp
  
  tic()
  print("time for GMT_MH is: ")
  treev <- GMT_MH(up_trainf,nbatch,par_root, alpha, beta, MT,gpv,av,wv,M,maxit) #return value is list of tree, weighting vector and rou
  toc()
  
  mle_m2L <- cbind(mle_m2L, list(mle_m2_out))
  
  # print("nrow of up_train is:")
  # 
  # print(nrow(up_trainf[[1]]))
  #get dataul, treev, mle_parv vector
  treev <-  treev[-1]
  
  tt <- table(treev)
  
  # prop.table(tt)
  TreeP <- rownames(tt)[which.max(tt)]
  
  # K <- 1 #the number of trees used to predict the results. e.g. only T2 or multiple Trees
  #when multiple trees, K is larger than 1
  
  tt_o <- order(tt)
  
  ttn <- rownames(tt)[tt_o] #the ascending order of tree name
  
  #descend order, the first element is the most important
  TS <- rev(tail(ttn,K)) #selected tree vector for data prediction, c("T1", "T2")
  
  #get the tree vector
  idxt <- which(sapply(dataul,is.null))
  
  if(isempty(idxt))
  {
    datallt <- dataul  #delete the element with NULL
  }
  else {
    datallt <- dataul[-idxt] #delete the element with NULL
  }
  
  lv <- c(1:length(datallt))
 
  tryCatch(
    {
      TV <- unlist(lapply(lv, function(x){datallt[[x]]$Tree}))
    },

    error = function(c){
      # save(vstr_n, file = "vstr_n_debug.RData")
      # save(av_mnl_n, file = "av_mnl_n_debug.RData")
      # save(up_trainDl, file = "up_trainDl_debug.RData")
      # save(av, file = "av_debug.RData")
      # save(up_trainft, file = "up_trainft_debug.RData")
      # 
      # save(up_trainf, file = "up_trainf_debug.RData")
      save(datallt, file = "error_datallt.RData")
      save(dataul, file = "error_dataul.RData")
    }
  )
  
  #calculate the selection probability
  tl <- c(1:length(TS))
  
  p_sum <- rep(0,nrow(up_train)) #sum of selection probability
  
  Nd <- 0 #the number of used data
  
  idxvv <- NULL
  print("wangshangbo5")
  for (j in tl) {
    idxv <- which(TV%in%TS[j]) #index of selected tree
    
    datallx <- datallt[idxv] #extract the selected data list: datall[1][[1]][[1]]
    
    Nd <- Nd + length(datallx)
    
    dl <- c(1:length(datallx))
    
    for (i in dl) {
      p_sum <- p_sum + datallx[[i]][[1]]$SP
    }
    
    idxvv <- c(idxvv,idxv)
  }
  
  #get the selected tree and its parameters
  # datallSL <- c(datallSL,list(datallt[idxvv]))
  # mle_parvL <- c(mle_parvL, list(mle_parv[idxvv]))
  # av_mnllL <- c(av_mnllL, list(av_mnll[idxvv]))
  
  datallSL <- c(datallSL,list(datallt))
  mle_parvL <- c(mle_parvL, list(mle_parv))
  av_mnllL <- c(av_mnllL, list(av_mnll))
  
  # datallS <- datallt[idxvv]
  
  print("wangshangbo6")
  
  q_k_n_m <- p_sum/Nd #the estimated selection probability at the m-th iteration
  
  #determine rou_mp1
  rou_m <- (2*nrow(up_trainf[[1]]) - sum(P_k_n_m))/sum(q_k_n_m) #initial value may be very large, cause unfeasible solution
  
  #normalize rou_m with rou_mv
  rou_mv_norm <- c(rou_mv,rou_m)/sum(c(rou_mv,rou_m))
  
  #update P_k_n_m by rou_m and q_k_n_m
  # P_k_n_m <- P_k_n_m + rou_m*q_k_n_m
  #update P_k_n_m by normalized rou_m
  P_k_n_m_norm <- P_k_n_m/sum(c(rou_mv,rou_m))
  P_k_n_m <- P_k_n_m_norm + rou_mv_norm[length(rou_mv_norm)]*q_k_n_m

  #update the sample distribution wv_mp1
  wv <- wv*exp(-rou_mv_norm[length(rou_mv_norm)]*q_k_n_m)
  
  #normalize wv
  wv <- wv/sum(wv)
  
  #save variables
  # rou_mv <- cbind(rou_mv,rou_m)
  rou_mv <- rou_mv_norm
  # colnames(rou_mv) <- NULL

  wvv <- cbind(wvv,wv)
  colnames(wvv) <- NULL
  
  P_k_n_mv <- cbind(P_k_n_mv,P_k_n_m)
  colnames(P_k_n_mv) <- NULL 
  
  q_k_n_mv <- cbind(q_k_n_mv,q_k_n_m)
  colnames(q_k_n_mv) <- NULL 
  print("wangshangbo7")
  # q_kn_ml <- lapply(TS, function(x){
  #   
  #     idxv <- which(TV%in%x) #index of selected tree
  #   
  #     datallx <- datallt[idxv] #extract the selected data list: datall[1][[1]][[1]]
  # 
  #     dl <- c(1:length(datallx))
  # 
  #     p_sum <- rep(0,nrow(up_train))
  # 
  #     for (i in dl) {#for each tree, get the alternative selection probability
  #       #datallx[i][[1]]$SP: selection probability for the i-th datallx
  #       p_sum <- p_sum + datallx[i][[1]]$SP
  #     }
  #     
  #     # p_sum<- p_sum/length(datallx)
  #     return(list(p_sum,length(idxv)))
  #   }
  # )
  #the filename for store
  idx_file_name <- paste0("idxv_Ntv_",idxi,"_",Sys.time(),".RData")
  datault_file_name <- paste0("datault_Ntv_",idxi,"_",Sys.time(),".RData")
  TS_file_name <- paste0("TS_",idxi,"_",Sys.time(),".RData")
  mle_parv_file_name <- paste0("mle_parv_", Sys.time(),".RData")
  av_mnll_file_name <- paste0("av_mnll_", Sys.time(),".RData")
  treev_file_name <- paste0("treev_", Sys.time(),".RData")
  
  # save(idxvv, file = idx_file_name)
  # save(datallt,file = datault_file_name)
  # save(TS, file = TS_file_name)
  # save(mle_parv, file = mle_parv_file_name)
  # save(av_mnll, file = av_mnll_file_name)
  # save(treev, file = treev_file_name)

  
  print("wangshangbo8")
  # wv <- wv
}

#save partial results
rou_mv_file_name <- paste0("rou_mv_",nbatch, "_alpha_", alpha, "_beta_", beta, "_Nt_", Nt, "_M_", M, "_K_" ,K, "_maxit_", maxit,"_length_uptrain_", nrow(up_train), "_sample_size_", ds, "_nationalData_", Sys.time(),".RData")
wvv_file_name <- paste0("wvv_",nbatch, "_alpha_", alpha, "_beta_", beta, "_Nt_", Nt, "_M_", M, "_K_" ,K, "_maxit_", maxit, "_length_uptrain_", nrow(up_train), "_sample_size_", ds,"_nationalData_", Sys.time(),".RData")
P_k_n_mv_file_name <- paste0("P_k_n_mv_",nbatch, "_alpha_", alpha, "_beta_", beta, "_Nt_", Nt, "_M_", M, "_K_" ,K, "_maxit_", maxit, "_length_uptrain_", nrow(up_train), "_sample_size_", ds, "_nationalData_" , Sys.time(),".RData")
q_k_n_mv_file_name <-paste0("q_k_n_mv_",nbatch, "_alpha_", alpha, "_beta_", beta, "_Nt_", Nt, "_M_", M, "_K_" ,K, "_maxit_", maxit, "_length_uptrain_", nrow(up_train),"_sample_size_", ds, "_nationalData_",Sys.time(),".RData")
datallSL_file_name <- paste0("datallSL_",nbatch, "_alpha_", alpha, "_beta_", beta, "_Nt_", Nt,"_M_", M, "_K_" ,K, "_maxit_", maxit, "_length_uptrain_", nrow(up_train), "_sample_size_", ds,"_nationalData_",Sys.time(),".RData")
mle_parvL_file_name <- paste0("mle_parvL_",nbatch, "_alpha_", alpha, "_beta_", beta, "_Nt_", Nt,"_M_", M, "_K_" ,K,"_maxit_", maxit, "_length_uptrain_", nrow(up_train),"_sample_size_", ds, "_nationalData_",Sys.time(),".RData")
av_mnllL_file_name <- paste0("av_mnllL_",nbatch, "_alpha_", alpha, "_beta_", beta, "_Nt_", Nt, "_M_", M, "_K_" ,K, "_maxit_", maxit, "_length_uptrain_", nrow(up_train),"_sample_size_", ds, "_nationalData_",Sys.time(),".RData")
mle_m2L_file_name <- paste0("mle_m2L_", nbatch, "_alpha_", alpha, "_beta_", beta, "_Nt_", Nt, "_M_", M, "_K_" ,K,"_maxit_", maxit, "_length_uptrain_", nrow(up_train),"_sample_size_", ds, "_nationalData_",Sys.time(),".RData")

Treevv_file_name <- paste0("Treevv_", nbatch, "_alpha_", alpha, "_beta_", beta, "_Nt_", Nt, "_M_", M, "_K_" ,K, "_maxit_", maxit, "_length_uptrain_", nrow(up_train),"_sample_size_", ds, "_nationalData_",Sys.time(),".RData")
dataull_file_name <- paste0("dataull_", nbatch, "_alpha_", alpha, "_beta_", beta, "_Nt_", Nt, "_M_", M, "_K_" ,K, "_maxit_", maxit, "_length_uptrain_", nrow(up_train),"_sample_size_", ds, "_nationalData_",Sys.time(),".RData")
par_mnlv_file_name <- paste0("par_mnlv_", nbatch, "_alpha_", alpha, "_beta_", beta, "_Nt_", Nt, "_M_", M, "_K_" ,K, "_maxit_", maxit, "_length_uptrain_", nrow(up_train),"_sample_size_", ds, "_nationalData_",Sys.time(),".RData")
av_MNLv_file_name <- paste0("av_MNLv_", nbatch, "_alpha_", alpha, "_beta_", beta, "_Nt_", Nt, "_M_", M, "_K_" ,K, "_maxit_", maxit, "_length_uptrain_", nrow(up_train),"_sample_size_", ds, "_nationalData_",Sys.time(),".RData")
PrvMv_file_name <- paste0("PrvMv_", nbatch, "_alpha_", alpha, "_beta_", beta, "_Nt_", Nt, "_M_", M, "_K_" ,K, "_maxit_", maxit, "_length_uptrain_", nrow(up_train),"_sample_size_", ds, "_nationalData_",Sys.time(),".RData")


save(rou_mv, file = rou_mv_file_name)
save(wvv, file = wvv_file_name)
save(P_k_n_mv, file = P_k_n_mv_file_name)
save(q_k_n_mv, file = q_k_n_mv_file_name)
save(datallSL, file = datallSL_file_name)
save(mle_parvL, file = mle_parvL_file_name)
save(av_mnllL, file = av_mnllL_file_name)
save(mle_m2L, file = mle_m2L_file_name)

save(Treevv, file = Treevv_file_name)
save(dataull,file = dataull_file_name)
save(par_mnlv, file = par_mnlv_file_name)
save(av_MNLv, file = av_MNLv_file_name)
save(PrvMv, file = PrvMv_file_name)


# PE_GMT_train <- length(which(up_train$Class != GMT_pre_train))/length(GMT_pre_train)
#give the final model:  rou_1*(mean(T_1,T_2,...,T_Nd)) + rou_2*(mean(T_1,T_2,...,T_Nd)) + ... + rou_Ntv*(mean(T_1,T_2,...,T_Nd))
#rou_mv[1]*(1/(length(datallSL[1][[1]]))*(sum(LL(mle_parv1, av_mnl, datallSL[1][[1]], PrvM,wv)))) 
# + rou_mv[2]*(1/(length(datallSL[2][[1]]))*(sum(LL(mle_parv2,av_mnl, datallSL[2][[1]], PrvM,wv))))
# + rou_mv[3]*(1/(length(datallSL[3][[1]]))*(sum(LL(mle_parv3,av_mnl, datallSL[3][[1]], PrvM,wv))))
# mle_parv1: mle_parv[idxv_Ntv_1], av_mnl:  setdiff(av,unique(unlist(up_trainf$iv)))
#PrvM: as.data.frame(Reduce(rbind, datallSL[[1]][[2]][[1]]$Prv))
#wv: all one vector
#datallSL[[1]][[1]][[1]]$distance_miles, datallSL[[2]][[1]][[1]]$workrelated: datallSL[[5]][[1]] (up_trainf); datallSL[[5]][[1]][[1]] (up_train)
#length(rou_mv): Nt; mle_parvL: length(mle_parvL): Nt; length(mle_parvL[[1]]): length(idxvv)
#av_mnl_m <- setdiff(av,unique(unlist(up_trainf$iv)))
#up_train: the training data
#datalli: data used for model training
#testdata: data for testing
#parv: paramter for MNL model at the terminal nodes
#av_mnl: available variables for MNL model


get_PrvM <- function(trainingdata, testdata){

  iv <- trainingdata$iv #historical variables of ancestors of each terminal node
  par <- trainingdata$par #parameters of each variable
  Tree <-trainingdata$Tree #

  #get the number of terminal nodes
  nt <- trainingdata$nt
  
  #define the PrvM matrix: 5*nrow(trainingdata[[1]])*nt
  PrvM <- matrix(rep(1,5*nrow(testdata)*nt), ncol = nt)
  
  vec <- c(1:nt)

  if(nt == 1) #root, use MNL model directly, Prvlt is a all-one vector

  {
    # Prvlt <- as.data.frame(rep(1, nrow(trainingdata[[1]])))
    return(PrvM)
  }

  else{ #PrvM is a matrix 
    
    TC <- MTl[as.numeric(unlist(strsplit(Tree,"T"))[2])][[1]]
    
    TC_split <- strsplit(TC,"")
    
    for (i in vec) { #for each terminal node, determine accumulated probability
      ivi <- iv[[i]]
      pari <- par[[i]] #parameter vector for the i-th terminal node
      
      # MTl[as.numeric(unlist(strsplit(Treei,"T"))[2])][[1]] 
      # TCi <- unlist(strsplit(TC[i],""))
      TC_spliti <- TC_split[[i]]
      
      tv <- c(1: (length(TC_spliti)-1))
      
      #for each split, calculate the splitting probability by ivi and testdata
      for (j in tv) {
        
        dataij <- as.data.frame(testdata[,which(colnames(testdata) %in% ivi[j][[1]])])
        dataijM <- as.matrix(dataij %>% mutate(I = rep(1,nrow(dataij))))
        parij <- pari[j][[1]] #get the parameter vector 
        parijM <- as.matrix(parij, nrow = length(parij))
      
        PR <- 1/(1 + exp(dataijM %*% parijM)) #right
        PL <- 1 - PR #left
        
        if(TC_spliti[j+1] == "0") #left
        {
          PLr <- rep(PL, each = 5)
          PrvM[,i] <- PrvM[,i]*PLr
        }
        else #right
        {
          PRr <- rep(PR, each = 5)
          PrvM[,i] <- PrvM[,i]*PRr
        }
      }
    }
    
    return(PrvM)
  }
}

# PrvM <- get_PrvM(datallSL[[1]][[1]], up_train)

# PrvMR <- Reduce(rbind,datallSL[[1]][[1]][[1]]$Prv)
#for Nt = 5. nbatch = 100
# load("rou_mv_0_95_beta_0_5_nbatch_100_2021-08-03 04%3A20%3A35.RData")
# load("mle_parvL_0_95_beta_0_5_nbatch_100_2021-08-03 04%3A20%3A35.RData")
# load("datallSL_0_95_beta_0_5_nbatch_100_2021-08-03 04%3A20%3A35.RData")
# load("av_mnllL_0_95_beta_0_5_nbatch_100_2021-08-03 04%3A20%3A35.RData")

#define the validation function
#data: up_train or up_test
GT_Val <- function(rou_mv,datallSL,data,mle_parvL,av_mnllL){

  rv <- c(1:length(rou_mv)) #length of ensemble trees
  
  print("shangbo1")
  
  # GT_Val <- matrix(rep(0,nrow(datallSL[[1]][[1]][[1]])*5),ncol = 5)  
  GT_Val <- matrix(rep(0,nrow(data)*5),ncol = 5)  
  
  print("shangbo2")

  for (i in rv) {
    rou_i <- rou_mv[i]

    datalli <<- datallSL[[i]] #length(datalli): number of DT

    mle_parv <<- mle_parvL[[i]] #required parameter vector for each decision tree

    av_mnll <<- av_mnllL[[i]] #required variable name vector for each decision tree

    dv <- c(1:length(datalli))

    sum_ll <- 0

    # wv <- rep(1,nrow(datalli[[1]][[1]])) #same weight for all observations
    wv <<- rep(1, nrow(data))#same weight for all observations
    
    prob_SMm <- matrix(rep(0,length(wv)*5),ncol = 5)

    for (j in dv) { #get the mean of selection probability

      # av_mnl <- setdiff(av,unique(unlist(datalli[[j]]$iv)))
      # datalli[[j]]$iv

      # PrvM <<- as.data.frame(Reduce(rbind, datalli[[j]][[1]]$Prv))
      #get the accumulated probability for each observation at each terminal node
      #by using  datalli[[j]]$iv and datalli[[j]]$par
      PrvM <<- get_PrvM(datalli[[j]], data)
      nt <- datalli[[j]]$nt
      
      # sum_ll <- sum_ll +
      
      tryCatch(
        {
          print(paste0("wv is: ", length(wv)))
          LL(mle_parv[[j]], av_mnll[[j]], nt, data, PrvM,wv)
        },
        
        error = function(c)
        {
          print(paste0("i is: ", i))
          print(paste0("j is: ", j))
          
          save(mle_parv[[j]], file = "debug_mle_parv.RData")
          save(av_mnll[[j]], file = "debug_av_mnll.RData")
          save(datalli[[j]], file = "debug_datalli.RData")
          save(PrvM, file = "debug_PrvM.RData")
          save(wv, file = "debug_wv.RData")
        }
      )
      
      # LL(mle_parv[[j]], av_mnll[[j]], datalli[[j]], PrvM,wv)

      #get the selection probability for five alternatives for each observation
      prob_SMm <- prob_SMm + prob_SM

    }
    
    prob_SMm <- prob_SMm/length(dv) #get the average prediction for each alterantive each observation
    
    GT_Val <- GT_Val + prob_SMm*rou_i #get the final result
  }
  
  return(GT_Val)

}


my_GT_val_train<- GT_Val(rou_mv, datallSL, up_train, mle_parvL, av_mnllL)
my_GT_val_test <- GT_Val(rou_mv, datallSL, up_test, mle_parvL, av_mnllL)

# my_GT_val_train<- GT_Val(rou_mv, datallSL_temp, up_train, mle_parvL_temp, av_mnllL_temp)
# my_GT_val_test <- GT_Val(rou_mv, datallSL_temp, up_test, mle_parvL_temp, av_mnllL_temp)

#save the prediction result
my_GT_val_train_filename <- paste0("my_GT_val_train_", nbatch, "_alpha_", alpha, "_beta_", beta, "_Nt_", Nt, "_M_", M, "_K_" ,K,"_maxit_", maxit, "_length_uptrain_", nrow(up_train),"_sample_size_", ds, "_nationalData_", Sys.time(), ".RData")
my_GT_val_test_filenmae <- paste0("my_GT_val_test_", nbatch, "_alpha_", alpha, "_beta_", beta, "_Nt_", Nt, "_M_", M, "_K_" ,K,"_maxit_", maxit, "_length_uptrain_", nrow(up_train),"_sample_size_", ds, "_nationalData_", Sys.time(), ".RData")

save(my_GT_val_train, file = my_GT_val_train_filename)
save(my_GT_val_test, file = my_GT_val_test_filenmae)

print("finished!!!")


###############################################################################
#
# Capital One Data Mining Cup 2015
# Code submission
#
###############################################################################
# Team Name:   
#     Hello Data
#
###############################################################################
# Team Member:
#
#     Ziang (Allen) Lu
#     Jiajia (Deborah) Yin
#     Yi Zhang
#     Yilun (Tom) Zhang
#
###############################################################################

###############################################################################
# Set work directory
setwd("F:/Documents/Mine/Capital One Data Mining Cup 2015/data")

###############################################################################
# Install libraries
# install.packages("rpart")
# install.packages("caret")
# install.packages("splitstackshape")
# install.packages("plyr")
library(rpart)
library(caret)
library(splitstackshape)
library(plyr)

# Set random seed
set.seed(123);

###############################################################################
# Helper functions

# 1. Count number of occurences of a character in a string
# input: character and string
# output: number of occurrences
countChar <- function(char, s) {
  s2 <- gsub(char,"",s)
  return (nchar(s) - nchar(s2))
}

# 2. Transform raw data to training data
buildToTrain <- function(buildSet){
  
  # Select relevant features
  buildSet <- buildSet[,c(2:5, 7, 9, 21:24, 26:28)]
  
  # NA to 0
  buildSet$VISITS[is.na(buildSet$VISITS)] <- 0
  buildSet$TOTAL_QUALITY_SCORE[is.na(buildSet$TOTAL_QUALITY_SCORE)] <- 0
  buildSet$APPLICATIONS[is.na(buildSet$APPLICATIONS)] <- 0
  
  # Deal with Campaign name:
  temp <- buildSet$CMPGN_NM
  temp <- sapply(temp,function(s){toString(s)})
  temp <- sapply(temp,function(s){substring(s,4,nchar(s))})
  temp <- sapply(temp,function(s){substring(s,1,6)})
  temp <- as.vector(temp)
  buildSet$CMPGN_NM <- temp
  buildSet$CMPGN_NM <- as.factor(buildSet$CMPGN_NM)
  
  # Split keywords
  out <- cSplit(indt=buildSet,splitCols="KEYWD_TXT",sep="+",direction="long");
  out <- out[out$KEYWD_TXT !=""];
  
  # Group rows by the 5 keys
  splitList <- dlply(out,.(ENGN_ID, LANG_ID, DVIC_ID, KEYWD_TXT, MTCH_TYPE_ID))
  
  # Group them to one row, add to resulting training set
  final <- matrix(NA, 0, ncol(out));
  for(i in 1:length(splitList)){
    thisM <- splitList[[i]];
    thisRow <- thisM[1,];
    thisRow[7:13] <- colMeans(thisM[7:13]);
    final <- rbind(final, thisRow);
  }
  
  # Add conversion rate
  final$CONV_RATE <- final$APPLICATIONS/final$CLICKS;
  final$CLICKS <- NULL;
  final$APPLICATIONS <- NULL;
  
  return(final) 
}

# 3. Transform raw data to test data
buildToTest <- function(buildSet){
  
  buildSet <- buildSet[,c(2:5,7, 9, 21:24, 26:28)];
  
  # NA to 0
  buildSet$VISITS[is.na(buildSet$VISITS)] <- 0
  buildSet$TOTAL_QUALITY_SCORE[is.na(buildSet$TOTAL_QUALITY_SCORE)] <- 0
  buildSet$APPLICATIONS[is.na(buildSet$APPLICATIONS)] <- 0
  
  # Deal with Campaign name:
  temp <- buildSet$CMPGN_NM
  temp <- sapply(temp,function(s){toString(s)})
  temp <- sapply(temp,function(s){substring(s,4,nchar(s))})
  temp <- sapply(temp,function(s){substring(s,1,6)})
  temp <- as.vector(temp)
  buildSet$CMPGN_NM <- temp
  buildSet$CMPGN_NM <- as.factor(buildSet$CMPGN_NM)
  
  buildSet$CONV_RATE <- buildSet$APPLICATIONS/buildSet$CLICKS;
  buildSet$CLICKS <- NULL;
  buildSet$APPLICATIONS <- NULL;
  
  buildSet$CONV_RATE[!is.finite(buildSet$CONV_RATE)] <- 0;
  
  return(buildSet)
  
}

# 4. Prediction with CART
# input: model,trainset, testset
# output: a predicted vector of conversion rates
predictWithCart <- function(cartModel, trainSet, testSet, campaignInfo){
  
  # Count number of words in each keyword text
  numWords <- countChar("K", as.character(testSet$KEYWD_TXT));
  addGroups <- as.numeric(testSet$CMPGN_NM);
  keyWords <- as.character(testSet$KEYWD_TXT);
  
  # Split keywords and deal with NA
  testSet <- cSplit(testSet,splitCols="KEYWD_TXT",sep="+",direction="long");
  testSet <- testSet[testSet$KEYWD_TXT !=""];
  testSet$KEYWD_TXT[!testSet$KEYWD_TXT %in% trainSet$KEYWD_TXT] <- NA
  
  # Make single predictions
  pred <- predict(cartModel, testSet, type="vector", na.action=na.pass)
  
  # Group results together to list first.
  results <- split(pred, rep(1:length(numWords), numWords))
  
  # Combine the results.
  for(i in 1:length(results)){
    
    vec <- results[[i]];
    thisWords <- keyWords[i]
    thisWords <- strsplit(thisWords, "+", fixed=T)[[1]][-1]
    
    wordTest <- thisWords %in%  names(campaignInfo[[addGroups[i]]]);
    propImp <- length(thisWords[wordTest])/length(thisWords);
    
    if(propImp > 0 && propImp < 1){
      res <- mean(vec[wordTest])*sqrt(propImp) + mean(vec[!wordTest])*(1-sqrt(propImp));
    }else{
      res <- mean(vec);
    }    
    
    #myMax <- max(vec);
    #myMin <- min(vec);
    #res <- ifelse(any(vec > 0.4), myMax*length(vec), myMin);
    #res <- min(vec)
    
    results[[i]] <- res;
  }
  
  # Get prediction values.
  results <- as.numeric(results)
  
  return(results);
}

# 5. Run CART method
# input: trainset, testset, train values, testvalues, alpha (parameter)
# output: mean square error of prediction
runCart <- function(trainSet, testSet, trainVals, testVals, alpha, campaignInfo){
  
  control <- rpart.control(cp=alpha);
  cartModel <- rpart(trainVals ~., trainSet, control=control);
  
  pred <- predictWithCart(cartModel, trainSet, testSet, campaignInfo);
  errors <- testVals - pred
  
  return(mean(errors^2));
}

# 6. CART cross validations
# input: parameter and number of folds
# output: a vector of k errors (error for each fold)
cartWithCV <- function(alpha, k){
  
  CVErr <- matrix(NA, 1, k);
  
  for(i in 1:k){
    
    trainData <- trainAndTestList[[i]]$trainData;
    nTrain <- ncol(trainData);
    
    trainSet <- trainData[,1:(nTrain-1)];
    trainVals<- trainData[,nTrain];
    
    testData <- trainAndTestList[[i]]$testData;
    nTest <- ncol(testData);
    
    testSet <- testData[,1:(nTest-1)];
    testVals <- testData[, nTest];
    rownames(testSet) <- NULL;
    
    thisCampaignInfo <- trainAndTestList[[i]]$campaignInfo;
    
    CVErr[i] <- runCart(trainSet, testSet, trainVals, testVals, alpha, thisCampaignInfo);
    
  }
  
  return(CVErr);
  
}

# 7. Calculate AR for every entry in validation set
getARForEntry <- function(row){
  
  keyText <- as.character(getElement(row, "KEYWD_TXT"));
  
  keyWords <- strsplit(keyText, "+", fixed=T)[[1]][-1]
  
  arVecs <- matrix(NA, length(keyWords), 1);
  
  for(i in 1:length(keyWords)){
    arVecs[i] <- getARForKey(keyWords[i]);
  }
  
  return(mean(arVecs));
  
}

# 8. Calculate REV for every entry in validation set
getRevForEntry <- function(row){
  
  keyText <- as.character(getElement(row,"KEYWD_TXT"));
  
  keyWords <- strsplit(keyText, "+", fixed=T)[[1]][-1]
  
  arVecs <- matrix(NA, length(keyWords), 1);
  
  for(i in 1:length(keyWords)){
    arVecs[i] <- getRevForKey(keyWords[i]);
  }
  
  return(mean(arVecs));
  
}

# 9. Calculate AR for each key
getARForKey <- function(key){
  
  keySet <- as.character(key_prod_table[,1])
  
  ar <- ifelse(key %in% keySet, AR_Table[keySet==key, 2], 0);
  return(ar)
  
}


# 10. Calculate REV for each key

getRevForKey <- function(key){
  
  keySet <- as.character(key_prod_table[,1])
  rev <- ifelse(key %in% keySet, REVS_Table[keySet==key, 2], 0);
  
  return(rev)
  
}

# 11. Get AR for whole test set
predictAR <- function(testSet){
  result <- apply(testSet, 1, getARForEntry)
  return(result)
}

# 12. Get Rev for whole test set
predictREV <- function(testSet){
  result <- apply(testSet, 1, getRevForEntry)
  return(result)
}

# 13. Transform validation set to right format.
validationToTest <- function(buildSet){
  
  buildSet <- buildSet[,c(2:5,7, 9, 20:24)];
  
  # NA to 0
  buildSet$VISITS[is.na(buildSet$VISITS)] <- 0
  buildSet$TOTAL_QUALITY_SCORE[is.na(buildSet$TOTAL_QUALITY_SCORE)] <- 0
  
  # Deal with Campaign name:
  temp <- buildSet$CMPGN_NM
  temp <- sapply(temp,function(s){toString(s)})
  temp <- sapply(temp,function(s){substring(s,4,nchar(s))})
  temp <- sapply(temp,function(s){substring(s,1,6)})
  temp <- as.vector(temp)
  buildSet$CMPGN_NM <- temp
  buildSet$CMPGN_NM <- as.factor(buildSet$CMPGN_NM)
  
  return(buildSet)
  
}

# 14. Summarize Campaign info
getCampaignInfo <- function(buildSet){  
  
  buildSet <- buildSet[,c(2:5,7, 9, 21:24, 26:28)]
  
  # Remove "GS/YB_" and "LANG2/3" from CMPGN_NM
  temp <- buildSet$CMPGN_NM
  temp <- sapply(temp,function(s){toString(s)})
  temp <- sapply(temp,function(s){substring(s,4,nchar(s))})
  temp <- sapply(temp,function(s){substring(s,1,6)})
  temp <- as.vector(temp)
  buildSet$CMPGN_NM <- temp
  buildSet$CMPGN_NM <- as.factor(buildSet$CMPGN_NM)
  
  # Transformation
  trans <- cSplit(indt=buildSet,splitCols="KEYWD_TXT",sep="+",direction="long");
  trans <- trans[trans$KEYWD_TXT !=""];
  
  CMPGN1 <- as.vector(trans$KEYWD_TXT[trans$CMPGN_NM == "CMPGN1"])
  CMPGN2 <- as.vector(trans$KEYWD_TXT[trans$CMPGN_NM == "CMPGN2"])
  CMPGN3 <- as.vector(trans$KEYWD_TXT[trans$CMPGN_NM == "CMPGN3"])
  CMPGN4 <- as.vector(trans$KEYWD_TXT[trans$CMPGN_NM == "CMPGN4"])
  CMPGN5 <- as.vector(trans$KEYWD_TXT[trans$CMPGN_NM == "CMPGN5"])
  CMPGN6 <- as.vector(trans$KEYWD_TXT[trans$CMPGN_NM == "CMPGN6"])
  CMPGN8 <- as.vector(trans$KEYWD_TXT[trans$CMPGN_NM == "CMPGN8"])
  CMPGN9 <- as.vector(trans$KEYWD_TXT[trans$CMPGN_NM == "CMPGN9"])
  
  C1 <- table(CMPGN1)
  C2 <- table(CMPGN2)
  C3 <- table(CMPGN3)
  C4 <- table(CMPGN4)
  C5 <- table(CMPGN5)
  C6 <- table(CMPGN6)
  C8 <- table(CMPGN8)
  C9 <- table(CMPGN9)
  C1 <- C1[order(C1,decreasing=T)]
  C2 <- C2[order(C2,decreasing=T)]
  C3 <- C3[order(C3,decreasing=T)]
  C4 <- C4[order(C4,decreasing=T)]
  C5 <- C5[order(C5,decreasing=T)]
  C6 <- C6[order(C6,decreasing=T)]
  C8 <- C8[order(C8,decreasing=T)]
  C9 <- C9[order(C9,decreasing=T)]

  CMPGNKeyWords <- list(C1[1:7], C2[1:7],C3[1:7],C4[1:7],C5[1:7],C6[1:7],C8[1:7],C9[1:7])
  return(CMPGNKeyWords)
}


###############################################################################
# Data Processing

# Read build data
rawData <- read.csv("SEM_DAILY_BUILD.csv");

# Get dimensions
n <- ncol(rawData);
noObs <- nrow(rawData);

# shuffle raw data
rawData <- rawData[sample(noObs),];

# remove row names
rownames(rawData) <- NULL;

#################################################
# CART algorithm with 10-fold-cross-validation

# Number of folds and sample size for each fold
k <- 10;
fold <- floor(noObs/k);

# Generate fold index
foldList <- list();
for(i in 1:(k-1)){
  foldList[[i]] <- (fold*(i-1)+1):(fold*i);
}
foldList[[k]] <- (fold*(k-1) + 1):noObs;

# Generate train and test data for all 10 folds
# Will get a list of 10 lists
# Each list contains train and test datasets
trainAndTestList <- list();
for(i in 1:k){
  testIndices <- foldList[[i]];
  trainIndices <- (1:noObs)[-testIndices];
  
  trainData <- buildToTrain(rawData[trainIndices,]);
  trainData$CONV_RATE <- ifelse(is.nan(trainData$CONV_RATE),
                                0,trainData$CONV_RATE);
  
  testData <- buildToTest(rawData[testIndices,]);
  campaignInfo <- getCampaignInfo(rawData[trainIndices,]);
  
  thisFold <- list();
  thisFold$trainData <- trainData;
  thisFold$testData <- testData;
  thisFold$campaignInfo <- campaignInfo;
  
  trainAndTestList[[i]] <- thisFold
}

# parameter tuning for CART method
alphas <- seq(0, 0.5, 0.05);
alphas <- matrix(alphas, length(alphas), 1);
allErr <- apply(alphas, 1, cartWithCV, k)
meanErr <- colMeans(allErr);
bestAlpha <- alphas[meanErr == min(meanErr)];
print(sprintf("Best Root Mean Squared Error is: %f", sqrt(min(meanErr))));

###############################################################################
# Build train model based on all BUILD data

build <- read.csv("SEM_DAILY_BUILD.csv",sep=",",header=T)

finalTrainData <- buildToTrain(build);
finalTrainData$CONV_RATE <- ifelse(is.nan(finalTrainData$CONV_RATE),
                                   0,finalTrainData$CONV_RATE);

numCol <- ncol(finalTrainData);

finalTrainSet <- finalTrainData[, 1:(numCol-1)];
finalTrainVals <- finalTrainData[, numCol];

finalCartModel <- rpart(finalTrainVals ~., finalTrainSet, cp=bestAlpha);
finalCampaignInfo <- getCampaignInfo(build);

###############################################################################
# Calculate Product Approval Rate
AR_1 <- sum(build$PROD_1_APPROVED, na.rm=T) / sum(build$APPS_PROD_1,na.rm=T)
AR_2 <- sum(build$PROD_2_APPROVED, na.rm=T) / sum(build$APPS_PROD_2,na.rm=T)
AR_3 <- sum(build$PROD_3_APPROVED, na.rm=T) / sum(build$APPS_PROD_3,na.rm=T)
AR_4 <- sum(build$PROD_4_APPROVED, na.rm=T) / sum(build$APPS_PROD_4,na.rm=T)
AR_5 <- sum(build$PROD_5_APPROVED, na.rm=T) / sum(build$APPS_PROD_5,na.rm=T)
AR_6 <- sum(build$PROD_6_APPROVED, na.rm=T) / sum(build$APPS_PROD_6,na.rm=T)

PROD_AR <- c(AR_1,AR_2,AR_3,AR_4,AR_5,0)

# Filter for all applications > 0 and not NA
filter <- build[(!is.na(build$APPLICATIONS) & build$APPLICATIONS >0),]

# Calculate product revenue (average overall)
PROD_REV_1 <- sum(filter$PROD_1_REVENUE)/sum(filter$PROD_1_APPROVED)
PROD_REV_2 <- sum(filter$PROD_2_REVENUE)/sum(filter$PROD_2_APPROVED)
PROD_REV_3 <- sum(filter$PROD_3_REVENUE)/sum(filter$PROD_3_APPROVED)
PROD_REV_4 <- sum(filter$PROD_4_REVENUE)/sum(filter$PROD_4_APPROVED)
PROD_REV_5 <- sum(filter$PROD_5_REVENUE)/sum(filter$PROD_5_APPROVED)
PROD_REV_6 <- sum(filter$PROD_6_REVENUE)/sum(filter$PROD_6_APPROVED)

PROD_REVS <- c(PROD_REV_1,PROD_REV_2,PROD_REV_3,PROD_REV_4,PROD_REV_5,0)

###############################################################################
# Determing relationship between keywords and Ad groups

# Remove unuseful columns
buildSet <- build[,c(2:5,7, 9, 21:24, 26:28)]

# Remove "GS/YB_" and "LANG2/3" from CMPGN_NM
temp <- buildSet$CMPGN_NM
temp <- sapply(temp,function(s){toString(s)})
temp <- sapply(temp,function(s){substring(s,4,nchar(s))})
temp <- sapply(temp,function(s){substring(s,1,6)})
temp <- as.vector(temp)
buildSet$CMPGN_NM <- temp
buildSet$CMPGN_NM <- as.factor(buildSet$CMPGN_NM)

# Transformation
trans <- cSplit(indt=buildSet,splitCols="KEYWD_TXT",sep="+",direction="long");
trans <- trans[trans$KEYWD_TXT !=""];

# create list of shown-up keywords for each Ad group
CMPGN1 <- as.vector(trans[trans$CMPGN == "CMPGN1"]$KEYWD_TXT)
CMPGN2 <- as.vector(trans[trans$CMPGN == "CMPGN2"]$KEYWD_TXT)
CMPGN3 <- as.vector(trans[trans$CMPGN == "CMPGN3"]$KEYWD_TXT)
CMPGN4 <- as.vector(trans[trans$CMPGN == "CMPGN4"]$KEYWD_TXT)
CMPGN5 <- as.vector(trans[trans$CMPGN == "CMPGN5"]$KEYWD_TXT)
CMPGN6 <- as.vector(trans[trans$CMPGN == "CMPGN6"]$KEYWD_TXT)
CMPGN8 <- as.vector(trans[trans$CMPGN == "CMPGN8"]$KEYWD_TXT)
CMPGN9 <- as.vector(trans[trans$CMPGN == "CMPGN9"]$KEYWD_TXT)
CMPGNKeyWords <- list(CMPGN1, CMPGN2,CMPGN3,CMPGN4,CMPGN5,CMPGN6,CMPGN8,CMPGN9)

# Sorted frequency table for each Ad group
C1 <- table(CMPGN1)
C2 <- table(CMPGN2)
C3 <- table(CMPGN3)
C4 <- table(CMPGN4)
C5 <- table(CMPGN5)
C6 <- table(CMPGN6)
C8 <- table(CMPGN8)
C9 <- table(CMPGN9)
C1 <- C1[order(C1,decreasing=F)]
C2 <- C2[order(C2,decreasing=F)]
C3 <- C3[order(C3,decreasing=F)]
C4 <- C4[order(C4,decreasing=F)]
C5 <- C5[order(C5,decreasing=F)]
C6 <- C6[order(C6,decreasing=F)]
C8 <- C8[order(C8,decreasing=F)]
C9 <- C9[order(C9,decreasing=F)]

# Plots
barplot(C1,horiz=T, main="Distribution of Keyword frequency in CMPGN1", 
        xlab="frequency", ylab="keyword", yaxt='n')
barplot(C2,horiz=T, main="Distribution of Keyword frequency in CMPGN2",
        xlab="frequency", ylab="keyword", yaxt='n')
barplot(C3,horiz=T, main="Distribution of Keyword frequency in CMPGN3",
        xlab="frequency", ylab="keyword", yaxt='n')
barplot(C4,horiz=T, main="Distribution of Keyword frequency in CMPGN4",
        xlab="frequency", ylab="keyword", yaxt='n')
barplot(C5,horiz=T, main="Distribution of Keyword frequency in CMPGN5", 
        xlab="frequency", ylab="keyword", yaxt='n')
barplot(C6,horiz=T, main="Distribution of Keyword frequency in CMPGN6", 
        xlab="frequency", ylab="keyword", yaxt='n')
barplot(C8,horiz=T, main="Distribution of Keyword frequency in CMPGN8", 
        xlab="frequency", ylab="keyword", yaxt='n')
barplot(C9,horiz=T, main="Distribution of Keyword frequency in CMPGN9",
        xlab="frequency", ylab="keyword", yaxt='n')

###############################################################################
# Approval Rate for each keyword

# Get all unique keyword
keywd <- unique(sapply(finalTrainData$KEYWD_TXT,toString))

# Create an empty table to store the probability of reaching each product from
# key word
key_prod_table <- data.frame(KEYWD_TXT = character(),
                             PROD_1_PROB = numeric(),
                             PROD_2_PROB = numeric(),
                             PROD_3_PROB = numeric(),
                             PROD_4_PROB = numeric(),
                             PROD_5_PROB = numeric(),
                             PROD_6_PROB = numeric())

# Change all NA to 0 in build data
build[is.na(build)] <- 0

# Generate probabilities
for (i in 1:length(keywd)){

  key <- keywd[i]
  k <- paste(key,"[$/+]",sep="")
  sub <- build[grep(k,build$KEYWD_TXT),]
  
  PROD_1 <- sum(sub$APPS_PROD_1)/sum(sub$APPLICATIONS)
  if (is.nan(PROD_1)){
    PROD_1 <- 0
  }
  PROD_2 <- sum(sub$APPS_PROD_2)/sum(sub$APPLICATIONS)
  if (is.nan(PROD_2)){
    PROD_2 <- 0
  }
  PROD_3 <- sum(sub$APPS_PROD_3)/sum(sub$APPLICATIONS)
  if (is.nan(PROD_3)){
    PROD_3 <- 0
  }
  PROD_4 <- sum(sub$APPS_PROD_4)/sum(sub$APPLICATIONS)
  if (is.nan(PROD_4)){
    PROD_4 <- 0
  }
  PROD_5 <- sum(sub$APPS_PROD_5)/sum(sub$APPLICATIONS)
  if (is.nan(PROD_5)){
    PROD_5 <- 0
  }
  PROD_6 <- sum(sub$APPS_PROD_6)/sum(sub$APPLICATIONS)
  if (is.nan(PROD_6)){
    PROD_6 <- 0
  }
  key_prod_table <- rbind(key_prod_table,data.frame(key,PROD_1,PROD_2,PROD_3,
                                                    PROD_4,PROD_5,PROD_6))
}

###############################################################################
# Calculate average approval rate for each key
AR <- as.matrix(key_prod_table[,2:7]) %*% as.matrix(PROD_AR,6,1)
AR_Table <- data.frame(KEYWD_TXT <- key_prod_table[,1],
                       AR <- AR)

###############################################################################
# Calculate average revenue for product

REVS <- as.matrix(key_prod_table[,2:7]) %*% as.matrix(PROD_REVS,6,1)
REVS_Table <- data.frame(KEYWD_TXT <- key_prod_table[,1],
                         REVS <- REVS)

###############################################################################
# Use model to predict values in validation set

# read data and transform
validationTest <- read.csv("SEM_DAILY_VALIDATION.csv");
validationSet <- validationToTest(validationTest);

# CART prediction
predConversionRate <- predictWithCart(finalCartModel, 
                                      finalTrainSet, 
                                      validationSet, finalCampaignInfo)

# predict approval rate, revenue and max bid
predAR <- predictAR(validationSet)
predRev <- predictREV(validationSet)
predMaxBid <- predRev*predAR*predConversionRate

# create submission file
submit <- cbind(validationSet, predAR, predRev, predConversionRate, predMaxBid)
colnames(submit)[12] <- "PREDICTED_APPROVAL_RATE"
colnames(submit)[13] <- "PREDICTED_REVENUE"
colnames(submit)[14] <- "CR_PRED"
colnames(submit)[15] <- "BE_BID";

submitNew <- cbind(validationTest[,1:7], submit[,14:15])

# write submission file into .csv
# write.csv(submitNew, "Hello_Data_Submission.csv", quote=F, row.names=F)

###############################################################################
# Post processing info & plot

# get conversion rate and max bid columns
allConv <- submit$CR_PRED;
allMaxBid <- submit$BE_BID;

# Max Bid plot
plot(allMaxBid[order(allMaxBid, decreasing=T)], type='p', pch=16,
     ylab="Breakeven Bid")
title("Distribution of Breakeven Bid")

# Conversion Rate plot
plot(allConv[order(allConv, decreasing=T)], type='p', pch=16, 
     ylab="Breakeven Conversion Rate")
title("Distribution of Breakeven Conversion Rate")

# Check number of zeros in conversion rate and max bid
numZerosConv <- length(allConv[allConv == 0]);
numZerosBid <- length(allMaxBid[allMaxBid == 0]);
percentageZerosConv <- numZerosConv/length(allConv);
percentageZerosBid <- numZerosBid/length(allMaxBid);


###############################################################################
# Data exploration

###########################################################
# CR for each product
build$CR <- build$APPLICATIONS / build$CLICKS
build$CR[!is.finite(build$CR)] <- 0

# product 1 - 6
CRForProd <- matrix(NA,6,1)
CRForProd[1] <- sum(build$APPS_PROD_1) / sum(build$CLICKS[build$APPS_PROD_1 >0])
CRForProd[2] <- sum(build$APPS_PROD_2) / sum(build$CLICKS[build$APPS_PROD_2 >0])
CRForProd[3] <- sum(build$APPS_PROD_3) / sum(build$CLICKS[build$APPS_PROD_3 >0])
CRForProd[4] <- sum(build$APPS_PROD_4) / sum(build$CLICKS[build$APPS_PROD_4 >0])
CRForProd[5] <- sum(build$APPS_PROD_5) / sum(build$CLICKS[build$APPS_PROD_5 >0])
CRForProd[6] <- sum(build$APPS_PROD_6) / sum(build$CLICKS[build$APPS_PROD_6 >0])
CRForProd[6] <- 0

# plot
barplot(CRForProd[,1],xlab="Product",ylab="Average Conversion Rate",
        names.arg=c(1:6),main="Average Conversion Rate per Product",col="lightblue",
        ylim=c(0,0.04))

###########################################################
# Revenue vs. ad_group

# Deal with Campaign name
temp <- build$CMPGN_NM
temp <- sapply(temp,function(s){toString(s)})
temp <- sapply(temp,function(s){substring(s,4,nchar(s))})
temp <- sapply(temp,function(s){substring(s,1,6)})
temp <- as.vector(temp)
build$CMPGN_NM <- temp
build$CMPGN_NM <- as.factor(buildSet$CMPGN_NM)

# revenue
REVForCMPGN <- matrix(NA,8,1)
build$TotalREV <- build$PROD_1_REVENUE + build$PROD_2_REVENUE + build$PROD_3_REVENUE +
  build$PROD_4_REVENUE + build$PROD_5_REVENUE + build$PROD_6_REVENUE
REVForCMPGN[1] <- mean(build$TotalREV[build$CMPGN_NM == "CMPGN1"])
REVForCMPGN[2] <- mean(build$TotalREV[build$CMPGN_NM == "CMPGN2"])
REVForCMPGN[3] <- mean(build$TotalREV[build$CMPGN_NM == "CMPGN3"])
REVForCMPGN[4] <- mean(build$TotalREV[build$CMPGN_NM == "CMPGN4"])
REVForCMPGN[5] <- mean(build$TotalREV[build$CMPGN_NM == "CMPGN5"])
REVForCMPGN[6] <- mean(build$TotalREV[build$CMPGN_NM == "CMPGN6"])
REVForCMPGN[7] <- mean(build$TotalREV[build$CMPGN_NM == "CMPGN8"])
REVForCMPGN[8] <- mean(build$TotalREV[build$CMPGN_NM == "CMPGN9"])

# plot
barplot(REVForCMPGN[,1],xlab="Campaign",ylab="Average Revenue per Bid",
        names.arg=c(1:6,8,9),main="Average Revenue per Bid for Each Campaign",col="lightblue",
        ylim=c(0,35))

# CMPGN2
table(CMPGNKeyWords[2])[order(table(CMPGNKeyWords[2]),decreasing=T)]
# 104, 382

# CMPGN3
table(CMPGNKeyWords[3])[order(table(CMPGNKeyWords[3]),decreasing=T)]
# 104, 382

# CMPGN4
table(CMPGNKeyWords[4])[order(table(CMPGNKeyWords[4]),decreasing=T)]
# 178, 426, 121, 587

# CMPGN7
table(CMPGNKeyWords[7])[order(table(CMPGNKeyWords[7]),decreasing=T)]
# 195, 25

###########################################################
# Google vs. Yahoo
CRForG <- mean(build$CR[build$ENGN_ID == "G"]) # 0.0075
CRForY <- mean(build$CR[build$ENGN_ID == "Y"]) # 0.0027

# plot
barplot(c(CRForG,CRForY),horiz=T,
        names.arg=c("Google","Yahoo"),
        col="lightblue",main="Average Conversion Rate by Platform",
        xlab="Average Conversion Rate",ylab="Platform",
        xlim=c(0,0.008))

###########################################################
# Device difference
CRForD <- mean(build$CR[build$DVIC_ID == "D"])
CRForM <- mean(build$CR[build$DVIC_ID == "M"])
CRForT <- mean(build$CR[build$DVIC_ID == "T"])

# Device Usage
summary(build$DVIC_ID)[1] / nrow(build)  # Desktop
summary(build$DVIC_ID)[2] / nrow(build)  # Mobile
summary(build$DVIC_ID)[3] / nrow(build)  # Tablet


# ttest
t.test(build$CR[build$DVIC_ID == "T"],build$CR[build$DVIC_ID == "D"])
t.test(build$CR[build$DVIC_ID == "T"],build$CR[build$DVIC_ID == "M"])
t.test(build$CR[build$DVIC_ID == "D"],build$CR[build$DVIC_ID == "M"])

# ftest
var.test(build$CR[build$DVIC_ID == "T"],build$CR[build$DVIC_ID == "D"])
var.test(build$CR[build$DVIC_ID == "T"],build$CR[build$DVIC_ID == "M"])
var.test(build$CR[build$DVIC_ID == "D"],build$CR[build$DVIC_ID == "M"])

# Plot
barplot(c(CRForD,CRForM,CRForT),names.arg=c("Desktop","Mobile","Tablet"),
        col="lightblue",xlab="Devices",ylab="Average Conversion Rate",
        ylim=c(0,0.007),main="Average Conversion Rate by Device")


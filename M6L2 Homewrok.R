date: "2016.03.12"
output: pdf_document
---

# 1. Additional packages needed
`install.packages("ggplot2");`            
`install.packages("C50");`     
`install.packages("gmodels");`     
`install.packages("rpart");`     
`install.packages("rattle");`     
`install.packages("RColorBrewer");`     
`install.packages("tree");`     
`install.packages("party");`     

```{r}
require(ggplot2)
require(C50)
require(gmodels)
require(rpart)
require(RColorBrewer)
require(tree)
require(party)
```

# 2. Data analysis

**I will choose the dataset "Abalone" which I used before as my dataset.**

##2.1 Setting dataset
```{r}
Abalone<-read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",header=FALSE)
colnames(Abalone)<-c("Sex","Length","Diameter","Height","Wholeweight","Shuckedweight","Visceraweight","Shellweight","Rings")
str(Abalone)
head(Abalone)
summary(Abalone)
length(Abalone)
names(Abalone)
table(Abalone$Sex)
Abalone$Sex
length(Abalone$Sex)
```

##2.2 Plot the data

```{r}
qplot(Abalone$Length,Abalone$Diameter,data=Abalone)+geom_point(aes(colour = factor(Abalone$Sex),shape = factor(Abalone$Sex)))
qplot(Abalone$Length,Abalone$Height,data=Abalone)+geom_point(aes(colour = factor(Abalone$Sex),shape = factor(Abalone$Sex)))
qplot(Abalone$Diameter,Abalone$Height,data=Abalone)+geom_point(aes(colour = factor(Abalone$Sex),shape = factor(Abalone$Sex)))
qplot(Abalone$Rings,Abalone$Height,data=Abalone)+geom_point(aes(colour = factor(Abalone$Sex),shape = factor(Abalone$Sex)))
summary(Abalone)
```

**From the graphs we can see that female and male mixes together. And infants don't have a significant seperation with another two types. Due to the graphs are not very well, I plan to use all eight predictors to do classification.**

##2.3 Decision Trees in R

###2.3.1 Reorder the data

```{r}
set.seed(12345)
abalone_rand <- Abalone[order(runif(4177)), ]
```

##2.4 C5.0 Model 

**I will use C5.0 as the main model to do the analysis.**

###2.4.1 Original data
```{r}
# compare the Abalone(In original order) and abalone_rand( random order) data frames
summary(Abalone$Height)
summary(abalone_rand$Height)
head(Abalone$Height)
head(abalone_rand$Height)

#split the data frames
Abalone_train <- abalone_rand[1:4000, ]
Abalone_test <- abalone_rand[4001:4177, ]

#check the proportion of class variable
prop.table(table(Abalone_train$Sex))
prop.table(table(Abalone_test$Sex))
```

```{r}
#C5.0 model
model <- C5.0(Abalone_train[-1], Abalone_train$Sex)
model 
summary(model)
```

```{r}
#Evaluating model performance
Aba_Sex_prep <- predict(model, Abalone_test)
CrossTable(Abalone_test$Sex,Aba_Sex_prep, prop.chisq=FALSE, prop.c=FALSE,prop.r=FALSE, dnn=c('actual sex', 'predicted sex'))
```

**The model has 130 size with 34.6% errors.It is acceptable considering we have 4177 data. We can see the rules from the decision tree. I think this way makes sense. It can split the dataframe, and help us make prediction.We check rules: Visceraweight > 0.149;Wholeweight > 1.0015:...Diameter > 0.525: F (154/71), we can conclude the conlusion. In general, female is always have a larger diameter, and a heavier weight. So this method makes sense.**


###2.4.2 Try different size


```{r}
Abalone_train <- abalone_rand[1:2000, ]
Abalone_test <- abalone_rand[4001:4177, ]

#check the proportion of class variable
prop.table(table(Abalone_train$Sex))
prop.table(table(Abalone_test$Sex))
```

```{r}
#C5.0 model
model <- C5.0(Abalone_train[-1], Abalone_train$Sex)
model 
summary(model)
```

```{r}
#Evaluating model performance
Aba_Sex_prep <- predict(model, Abalone_test)
CrossTable(Abalone_test$Sex,Aba_Sex_prep, prop.chisq=FALSE, prop.c=FALSE,prop.r=FALSE, dnn=c('actual sex', 'predicted sex'))
```

**Although I change the size form 4000 to 2000, it doesn't make any significant change. Maybe if I change the data size in a small scale, it will change a lot. So for this question, size has no significant influence on the result. Maybe this is only a wrong answer, and this need us to do more experiments.** 


###2.4.3 Scaled data
```{r}
Abalone.scaled <- as.data.frame(lapply(Abalone[, c(2:9)],scale))
head(Abalone.scaled)
summary(Abalone.scaled)
```


```{r}
set.seed(12345)
Abalone_train_2 <- data.frame(Sex=Abalone[1:4000,1],Abalone.scaled[1:4000,]) 

Abalone_test_2 <- data.frame(Sex=Abalone[4001:4177,1],Abalone.scaled[4001:4177,])
model_2 <- C5.0(Abalone_train_2[-1],Abalone_train_2$Sex) 
model_2
summary(model_2)
```

**By using scaled data, we get the result:size of 78 with the error rate:37.2%.**


###2.4.4 Normalized data
```{r}
normalize <- function(x)
{
  return ((x-min(x))/(max(x)-min(x)))
}

Abalone.normalized <- as.data.frame(lapply(Abalone[,c(2:9)], normalize))
head(Abalone.normalized)
summary(Abalone.normalized)
head(Abalone.normalized)
summary(Abalone.normalized)
```


```{r}
set.seed(12345)
Abalone_train_3<-data.frame(Sex=Abalone[1:4000,1],Abalone.normalized[1:4000,])
Abalone_test_3<-data.frame(Sex=Abalone[4001:4177,1],Abalone.normalized[4001:4177,])
model_3 <- C5.0(Abalone_train_3[-1],Abalone_train_3$Sex) 
model_3
summary(model_3)
```

**The result of normalized is the same with the scaled data. Both the scaled and normalized results are very similar to the original result. So those methods don't help a lot.** 


###2.4.5 Repart
```{r}
Abalone.train <- Abalone[1:4177,]
formula <- Sex~ Length + Height+Diameter + Wholeweight + Shuckedweight+ Visceraweight+Rings+Shellweight
fit <- rpart(formula, method="anova" ,data=Abalone.train)
printcp(fit)
plotcp(fit)
summary(fit)
```


###2.4.6 Regression
```{r}
# grow tree 
fit <- rpart(formula, method="anova", data=Abalone.train)

printcp(fit) # display the results 
plotcp(fit) # visualize cross-validation results 
summary(fit) # detailed summary 

# create additional plots 
par(mfrow=c(1,2)) # two plots on one page 
rsq.rpart(fit) # visualize cross-validation results    
```

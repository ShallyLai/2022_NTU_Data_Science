setwd("~/Desktop/DataScience/HW5")

p1 <- read.csv("p1.csv", sep = ',')

library(rpart)
library(rpart.plot)
library(dplyr)

gini_process <- function(classes, splitvar = NULL){
  if(is.null(splitvar)){
    base_prob <- table(classes) / length(classes)
    return(1 - sum(base_prob ** 2))
  }
  base_prob <- table(splitvar) / length(splitvar) 
  crosstab <- table(classes, splitvar)
  crossprob <- prop.table(crosstab, 2)
  No_Node_Gini <- 1 - sum(crossprob[, 1] ** 2)
  Yes_Node_Gini <- 1 - sum(crossprob[, 2] ** 2)
  return (sum(base_prob * c(No_Node_Gini, Yes_Node_Gini)))
}

dtree <- rpart(formula = Class ~ Gender + Car.Type + Shirt.Size, 
                 data = p1, 
                 parms = list(split = "gini"),
                 method = "class",
                 control = rpart.control(minsplit = 2, minbucket = 1, cp = 0.001))
rpart.plot(dtree)
summary(dtree)
gini_process(p1$Class, p1$Car.Type == "Family")

sub_1 <- p1 %>% filter(Car.Type != "Family")
gini_process(sub_1$Class, (sub_1$Shirt.Size == "Extra Large" | sub_1$Shirt.Size == "Large"))

sub_2 <- sub_1 %>% filter(Shirt.Size == "Extra Large" | Shirt.Size == "Large")
gini_process(sub_2$Class, sub_2$Car.Type == "Luxury")

sub_3 <- sub_2 %>% filter(Car.Type == "Sports")
gini_process(sub_3$Class, sub_3$Shirt.Size == "Extra Large")


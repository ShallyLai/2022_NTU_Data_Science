setwd("~/Desktop/DataScience")

# Set two classes
class1 <- rbind(c(5, 3), c(3, 5), c(3, 4), c(4, 5), c(4, 7), c(5, 6))
class2 <- rbind(c(9, 10), c(7, 7), c(8, 5), c(8, 8), c(7, 2), c(10, 8))

# Get mean of two classes
mean1 <- matrix(colMeans(class1))
mean2 <- matrix(colMeans(class2))
total_mean <- (mean1 + mean2) / 2

# Between-class scatter matrix
Sb1 <- (mean1 - total_mean) %*% t(mean1 - total_mean)
Sb2 <- (mean2 - total_mean) %*% t(mean2 - total_mean)
SB <- Sb1 + Sb2

# Within-class scatter matrix
SWithin <- function(class, mean){
  tmp_Sw <- t(class)
  for(i in 1:6){
    tmp_Sw[, i] <- tmp_Sw[, i] - mean
  }
  tmp_Sw <- tmp_Sw %*% (t(tmp_Sw))
}

Sw1 <- SWithin(class1, mean1)
Sw2 <- SWithin(class2, mean2)
SW <- Sw1 + Sw2

# Calculate eigenvector and eigenvalue
A <- solve(SW) %*% SB
eig <- eigen(A)
eig

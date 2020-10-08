library(tidyverse)
library(httr)
library(gridExtra)
library(Rborist)
library(caret)

fs <- httr::GET("https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip")
temp <- tempfile()
download.file(fs$url,temp)
bancos <- read.csv(unz(temp, "bank-full.csv"), sep = ";")
unlink(temp)

names(bancos)
head(bancos)

# Global percentage of subscription

cat(mean(bancos$y == "yes"))

# Age

summary(bancos$age)

bancos %>% ggplot(aes(age)) +
  geom_histogram(aes(color = y),binwidth = 10) +
  facet_grid(y ~ ., scales = "free_y") +
  theme(legend.position = "none") +
  ggtitle("Histograms: Age classified by output")

bancos1 <- bancos %>% filter(y == "yes")
bancos0 <- bancos %>% filter(y == "no")
cat(round(mean(bancos1$age),2))
cat(round(mean(bancos0$age),2))

# Jobs

bancos_jobs <- bancos %>% group_by(job) %>%
  summarize(perc.subs = mean(y == "yes"), n = n()) %>%
  arrange(desc(perc.subs))
bancos_jobs %>% ggplot(aes(perc.subs,job)) + geom_col() + xlab("Subscription Rate") + ggtitle("Subscription rates classified by candidate's job")

# Marital Status

bancos_mar <- bancos %>% group_by(marital) %>%
  summarize(perc.subs = mean(y == "yes"), n = n()) %>%
  arrange(desc(perc.subs))
bancos_mar %>% ggplot(aes(perc.subs,marital)) +
  geom_col() + xlab("Subscription Rate") +
  ggtitle("Subscription rates classified by candidate's marital status")

# Education

bancos_ed <- bancos %>% group_by(education) %>%
  summarize(perc.subs = mean(y == "yes"), n = n()) %>%
  arrange(desc(perc.subs))
bancos_ed %>% ggplot(aes(perc.subs,education)) + geom_col() +
  xlab("Subscription Rate") +
  ggtitle("Subscription rates classified by candidate's educational level")

# Credit History

bancos_def <- bancos %>% group_by(default) %>%
  summarize(perc.subs = mean(y == "yes",na.rm = TRUE), n = n()) %>%
  arrange(desc(perc.subs))
default <- bancos_def %>% ggplot(aes(perc.subs,default)) + geom_col() +
  scale_x_continuous(limits = c(0,0.2)) + xlab("Subscription Rate") +
  ylab("Default")
bancos_house <- bancos %>% group_by(housing) %>%
  summarize(perc.subs = mean(y == "yes",na.rm = TRUE), n = n()) %>%
  arrange(desc(perc.subs))
housing <- bancos_house %>% ggplot(aes(perc.subs,housing)) + geom_col() + scale_x_continuous(limits = c(0,0.2)) + xlab("Subscription Rate") +
  ylab("Housing Loan")
bancos_pers <- bancos %>% group_by(loan) %>%
  summarize(perc.subs = mean(y == "yes",na.rm = TRUE), n = n()) %>%
  arrange(desc(perc.subs))
personal <- bancos_pers %>% ggplot(aes(perc.subs,loan)) + geom_col() +
  scale_x_continuous(limits = c(0,0.2)) + xlab("Subscription Rate") +
  ylab("Personal Loan")
library(grid)
grid.newpage()
grid.draw(rbind(ggplotGrob(default), ggplotGrob(personal),
                ggplotGrob(housing)))


# Type of contact

bancos_cont <- bancos %>% group_by(contact) %>%
  summarize(perc.subs = mean(y == "yes"), n = n()) %>%
  arrange(desc(perc.subs))
bancos_cont %>%
  ggplot(aes(perc.subs,contact)) + geom_col() +
  xlab("Subscription Rate") +
  ggtitle("Subscription rates classified by type of contact")

# Month seasonality

bancos_month <- bancos %>% group_by(month) %>%
  summarize(perc.subs = mean(y == "yes"), n = n()) %>%
  arrange(desc(perc.subs))
bancos_month %>%
  mutate(month = fct_relevel(month, "jan", "feb", "mar", "apr", "may",
                             "jun", "jul", "aug", "sep", "oct", "nov",
                             "dec")) %>%
  ggplot(aes(month,perc.subs)) + geom_col() +
  ggtitle("Subscription rates per month")


# Day of the Month

bancos_day <- bancos %>% group_by(day) %>%
  summarize(perc.subs = mean(y == "yes"), n = n()) %>%
  arrange(desc(perc.subs))
bancos_day %>% ggplot(aes(day,perc.subs)) + geom_col() + ggtitle("Subscription rates per day of the month")

# Number of contacts

bancos %>% ggplot(aes(campaign)) +
  geom_histogram(aes(color = y),binwidth = 3) + 
  facet_grid(y ~ ., scales = "free_y") +
  theme(legend.position = "none") +
  ggtitle("Histograms: Number of contacts classified by output")


# Test set will be 20% of data
set.seed(2015, sample.kind="Rounding")
test_index <- createDataPartition(y = bancos$y, times = 1, p = 0.2, list = FALSE)
train_set <- bancos[-test_index,]
test_set <- bancos[test_index,]

# Setting every output as "no"
test_set <- test_set %>% mutate(all_no = "no")
cat(paste("Sensitivity:",
          round(confusionMatrix(as.factor(test_set$all_no),as.factor(test_set$y))$byClass["Sensitivity"],5)))
cat(paste("Specificity:",
          round(confusionMatrix(as.factor(test_set$all_no),as.factor(test_set$y))$byClass["Specificity"],5)))
cat(paste("Accuracy:",
          round(confusionMatrix(as.factor(test_set$all_no),as.factor(test_set$y))$overall["Accuracy"],5)))

# Logistic Regression
train_logit <- train(y~age+job+marital+education+default+housing+loan+
                       contact+day+month+campaign,
                     method = "glm", data = train_set)
summary(train_logit)

# Logistic Regression Without default and day
train_logit <- train(y~age+job+marital+education+housing+loan+contact+
                       month+campaign, method = "glm", data = train_set)
y_hat_logit <- predict(train_logit, test_set, type = "raw")
cat(paste("Sensitivity:",
          round(confusionMatrix(y_hat_logit,as.factor(test_set$y))$byClass["Sensitivity"],5)))
cat(paste("Specificity:",
          round(confusionMatrix(y_hat_logit,as.factor(test_set$y))$byClass["Specificity"],5)))
cat(paste("Accuracy:",
          round(confusionMatrix(y_hat_logit,as.factor(test_set$y))$overall["Accuracy"],5)))


# Linear Discriminant Analysis
train_lda <- train(as.factor(y)~age+job+marital+education+housing+loan+
                     contact+month+campaign+day, method = "lda",
                   data = train_set)
y_hat_lda <- predict(train_lda, test_set, type = "raw")
cat(paste("Sensitivity:",
          round(confusionMatrix(y_hat_lda,as.factor(test_set$y))$byClass["Sensitivity"],5)))
cat(paste("Specificity:",
          round(confusionMatrix(y_hat_lda,as.factor(test_set$y))$byClass["Specificity"],5)))
cat(paste("Accuracy:",
          round(confusionMatrix(y_hat_lda,as.factor(test_set$y))$overall["Accuracy"],5)))

# Quadratic Discriminant Analysis
train_qda <- train(as.factor(y)~age+job+marital+education+housing+loan+
                     contact+month+campaign+day, method = "qda",
                   data = train_set)
y_hat_qda <- predict(train_qda, test_set, type = "raw")
cat(paste("Sensitivity:",
          round(confusionMatrix(y_hat_qda,as.factor(test_set$y))$byClass["Sensitivity"],5)))
cat(paste("Specificity:",
          round(confusionMatrix(y_hat_qda,as.factor(test_set$y))$byClass["Specificity"],5)))
cat(paste("Accuracy:",
          round(confusionMatrix(y_hat_qda,as.factor(test_set$y))$overall["Accuracy"],5)))

# K nearest neighbours
train_knn <- train(as.factor(y)~age+job+marital+education+housing+loan+
                     contact+month+campaign+day, method = "knn",
                   tuneGrid = data.frame(k = c(1,3,5,7,9)),
                   data = train_set)
y_hat_knn <- predict(train_knn, test_set, type = "raw")
cat(paste("Sensitivity:",
          round(confusionMatrix(y_hat_knn,as.factor(test_set$y))$byClass["Sensitivity"],5)))
cat(paste("Specificity:",
          round(confusionMatrix(y_hat_knn,as.factor(test_set$y))$byClass["Specificity"],5)))
cat(paste("Accuracy:",
          round(confusionMatrix(y_hat_knn,as.factor(test_set$y))$overall["Accuracy"],5)))

# Random Forest
train_rf <- train(as.factor(y)~age+job+marital+education+housing+loan+
                    contact+month+campaign+day, method = "Rborist",
                  tuneGrid = expand.grid(minNode = 1:5,
                                         predFixed = c(10,15,20,25,30)),
                  nTree = 100, nSamp = 5000,data = train_set)
y_hat_rf <- predict(train_rf, test_set, type = "raw")
cat(paste("Sensitivity:",
          round(confusionMatrix(y_hat_rf,as.factor(test_set$y))$byClass["Sensitivity"],5)))
cat(paste("Specificity:",
          round(confusionMatrix(y_hat_rf,as.factor(test_set$y))$byClass["Specificity"],5)))
cat(paste("Accuracy:",
          round(confusionMatrix(y_hat_rf,as.factor(test_set$y))$overall["Accuracy"],5)))

# Ensemble
y_hat_ensemble <- as.factor(ifelse(((y_hat_lda == "yes") +
                                      (y_hat_qda == "yes") +
                                      (y_hat_logit == "yes") +
                                      (y_hat_knn == "yes") +
                                      (y_hat_rf == "yes")) < 3,"no","yes" ))
cat(paste("Sensitivity:",
          round(confusionMatrix(y_hat_ensemble,as.factor(test_set$y))$byClass["Sensitivity"],5)))
cat(paste("Specificity:",
          round(confusionMatrix(y_hat_ensemble,as.factor(test_set$y))$byClass["Specificity"],5)))
cat(paste("Accuracy:",
          round(confusionMatrix(y_hat_ensemble,as.factor(test_set$y))$overall["Accuracy"],5)))

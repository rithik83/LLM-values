#!/bin/env Rscript

levels <- c("1", "2", "3")

evaluation <- read.csv("../evaluation.tsv", sep="\t")
evaluation <- evaluation[evaluation$Label != "Mean",]

getF1ScoresPerLevel <- function(method.name, sets) {
  evaluation.method <- evaluation[evaluation$Method == method.name,]
  
  evaluation.method.levels <- list()
  for (i in 1:length(levels)) {
    evaluation.method.levels[[i]] <- (evaluation.method[evaluation.method$Level == levels[i] & evaluation.method$Test.dataset %in% c(sets),])$F1
  }
  
  evaluation.method.levels
}


##### USA dataset #####

print("USA")

evaluation.svm.usa <- getF1ScoresPerLevel("SVM", "usa")
evaluation.baseline.usa <- getF1ScoresPerLevel("1-Baseline", "usa")
evaluation.bert.usa <- getF1ScoresPerLevel("BERT", "usa")

for (i in  1:2) {
  print(paste("Level", levels[i]))
  print(wilcox.test(evaluation.svm.usa[[i]], evaluation.bert.usa[[i]], conf.int = T, alternative = "two.sided", paired = T, conf.level = 0.95, exact = F))
}

for (i in  1:2) {
  print(paste("Level", levels[i]))
  print(wilcox.test(evaluation.bert.usa[[i]], evaluation.baseline.usa[[i]], conf.int = T, alternative = "two.sided", paired = T, conf.level = 0.95, exact = F))
}

#######################


##### Across cultures #####

print("Across cultures")

evaluation.svm <- getF1ScoresPerLevel("SVM", c("africa", "china", "india", "usa"))
evaluation.baseline <- getF1ScoresPerLevel("1-Baseline", c("africa", "china", "india", "usa"))
evaluation.bert <- getF1ScoresPerLevel("BERT", c("africa", "china", "india", "usa"))

for (i in  1:length(levels)) {
  print(paste("Level", levels[i]))
  print(wilcox.test(evaluation.svm[[i]], evaluation.bert[[i]], conf.int = T, alternative = "two.sided", paired = T, conf.level = 0.95, exact = F))
}

for (i in  1:length(levels)) {
  print(paste("Level", levels[i]))
  print(wilcox.test(evaluation.bert[[i]], evaluation.baseline[[i]], conf.int = T, alternative = "two.sided", paired = T, conf.level = 0.95, exact = F))
}

###########################

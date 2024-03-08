#!/bin/env Rscript

arguments <- read.csv("../arguments.tsv", sep="\t")
datasets <- sort(unique(arguments$Dataset))

cat("Label\tLevel")
for (dataset in datasets) {
  cat("\t")
  cat(dataset)
}
cat("\tTotal\n")

for (level in c("1", "2", "3", "4a", "4b")) {
  labels <- read.csv(paste("../labels-level", level, ".tsv", sep=""), sep="\t", check.names=FALSE)
  counts <- colSums(as.matrix(labels[,2:dim(labels)[2]]))
  counts.datasets <- list()
  for (dataset in datasets) {
    labels.dataset <- labels[labels[["Argument ID"]] %in% arguments[arguments$Dataset == dataset,]$ID,]
    counts.datasets[[dataset]] <- colSums(as.matrix(labels.dataset[,2:dim(labels)[2]]))
  }

  for (l in 1:length(counts)) {
    cat(names(counts)[l])
    cat("\t")
    cat(level)
    for (dataset in datasets) {
      cat("\t")
      cat(counts.datasets[[dataset]][l])
    }
    cat("\t")
    cat(counts[l])
    cat("\n")
  }
}


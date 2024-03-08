#!/bin/env Rscript

max.num <- 10

level.colors <- c("#006837", "#31a354", "#78c679", "#c2e699", "#ffffcc") # 5-class YlGn
level.colors <- c("#E6F3FF", "#B3D9FF", "#7FBFFF", "#7FC31C", "#800080") # Paper colors
legend.names <- c("Values (Level 1)", "Value categories (Level 2)", "Higher-order values (Level 3)", "Personal/Social Focus (Level 4a)", "Motivation (Level 4b)")

level.getCounts <- function(level) {
  data <- read.csv(paste("../labels-level", level, ".tsv", sep=""), sep="\t")
  level.counts <- colSums(t(as.matrix(data[,2:dim(data)[2]])))
  return(level.counts)
}

level.getFrequencies <- function(level, relative=TRUE, length=NULL) {
  data <- read.csv(paste("../labels-level", level, ".tsv", sep=""), sep="\t")
  level.counts <- colSums(t(as.matrix(data[,2:dim(data)[2]])))
  frequencies.table <- table(level.counts)

  frequencies <- rep(0, dim(data)[2]-1)
  frequencies[as.integer(names(frequencies.table))] <- frequencies.table
  if (relative) {
    frequencies <- frequencies / dim(data)[1]
  }
  names(frequencies) <- 1:(length(frequencies)-1)
  if (!is.null(length)) {
    if (length(frequencies) < length) {
      frequencies.fixedLength <- rep(NA, length)
      frequencies.fixedLength[1:length(frequencies)] <- frequencies
      frequencies <- frequencies.fixedLength
    } else if (length(frequencies) > length) {
      frequencies <- frequencies[1:length]
    }
  }
  return(frequencies)
}

level.overview <- function(level) {
  level.counts <- level.getCounts(level)
  level.frequencies <- level.getFrequencies(level)
  write(paste("Level", level), stdout())
  write(level.frequencies, stdout())
  write(paste("Mode:  ", which.max(table(level.counts))), stdout())
  write(paste("Median:", median(level.counts)), stdout())
  write(paste("Mean:  ", mean(level.counts)), stdout())
  write("", stdout())
}

pdf("density-number-of-labels.pdf", 7, 3)
# Prepare the plot
par(mar=c(2.2,3.5,0,0))
plot(1, xlim=c(0, (max.num)*6), ylim=c(0, 1), type="n", yaxt="n", xaxt="n", ylab="", xlab="", bty="n")
mtext(sprintf("%d", 1:max.num), 1, 0, at=c(3.5 + (0:1)*6, 2.5 + (2:3)*6, 2.15 + (4:(max.num-1))*6), cex = 1.2)
mtext("Number of labels per argument", 1, 1.2, cex = 1.2)
axis(2, labels=FALSE)
axis(2, at=(0:3)/5 + 0.1, labels=FALSE, tck=-0.025)
mtext(sprintf("%.1f", (0:5)/5), 2, 0.75, at=(0:5)/5, las=1, cex = 1.2)
mtext("Argument density", 2, 2.4, cex = 1.2)
# Draw horizontal lines
for (y in (1:10)/10) {
  lines(c(-0.65, (max.num)*6+0.65), c(y, y), col="gray", lty=3)
}
# Plot
barplot(t(sapply(c("1", "2", "3", "4a", "4b"), level.getFrequencies, length=10)), col=level.colors, beside=TRUE, axes=FALSE, axisnames=FALSE, add=TRUE)
# Legend
legend("topright", legend.names, fill=level.colors, box.col="white", cex = 1.3)
dev.off();

for (level in c("1", "2", "3", "4a", "4b")) {
  level.overview(level)
}

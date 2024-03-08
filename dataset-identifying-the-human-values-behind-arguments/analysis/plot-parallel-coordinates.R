#!/bin/env Rscript

plot.colors <- c("blue", "orange", "purple", "yellow")
plot.pch <- c(1, 2, 0, 5)
level.names <- c("Values (Level 1)", "Value categories (Level 2)", "Higher-order values (Level 3)", "Personal/social focus (Level 4a)", "Motivation (Level 4b)")
level.xaxts.vertical <- c(TRUE, TRUE, FALSE, FALSE, FALSE)
level.bar.widths <- c(0.8, 0.8, 0.4, 0.4, 0.4)
level.legend.poss <- c("topright", FALSE, FALSE, FALSE, FALSE)
level.mars.bottom <- c(10, 8, 2.5, 2.5, 2.5)
level.mars.left <- c(3.75, 3.75, 2.5, 2.5, 2.5)
level.heights <- c(4, 4, 2.9, 2.9, 2.9)
level.widths <- c(15, 9, 6, 4, 4)
level.points.space <- c(0.13, 0.13, 0.09, 0.08, 0.08)


getPlotLabel <- function(name) {
  name <- toString(name)
  name <- gsub(", ", ",\n", name)
  name <- gsub(" focus", "\nfocus", name)
  name <- gsub("-enhancement", "-\nenhancement", name)
  name <- gsub("-transcendence", "-\ntranscendence", name)
  name <- gsub("Openness", "Openness\n", name)
  return(name)
}

arguments <- read.csv("../arguments.tsv", sep="\t")

data <- read.csv("../evaluation.tsv", sep="\t")

parallelCoordinates <- function(
    level,
    level.name = level.names[level.number],
    level.number,
    data,
    subset.attribute,
    subset.names = unique(data[data$Level == level,][[subset.attribute]]),
    labels = unique(data[data$Level == level & data$Label != "Mean",]$Label),
    argument.ids = NULL,
    bar.width = level.bar.widths[level.number],
    xaxt.vertical = level.xaxts.vertical[level.number],
    legend.pos = level.legend.poss[level.number],
    points.space = level.points.space[level.number],
    ...) {
  data.level <- data[data$Level == level,]
  xs <- c(1:length(labels), length(labels) + 1)

  drawF1s <- function(f1s, ...) {
    lines(1:length(labels), f1s, type="b", ...)
    #wrap.slope <- f1s[1] - f1s[length(labels)]  
    #lines(c(0.6, 0.9), c(f1s[1] - wrap.slope / 5 * 2, f1s[1] - wrap.slope / 10), ...)
    #lines(c(length(labels) + 0.1, length(labels) + 0.4), c(f1s[length(labels)] + wrap.slope / 10, f1s[1] - wrap.slope / 5 * 2), ...)
    points(xs[length(xs)], mean(f1s), ...)
  }

  drawGrid <- function() {
    # Draw vertical lines
    for (x in xs) {
      lines(c(x, x), c(0, 1), col="gray")
    }

    # Draw horizontal lines
    for (y in (0:10)/10) {
      lines(c(1-bar.width/2, max(xs) + 0.2), c(y, y), col="gray", lty=3)
    }
  }

  # Prepare the plot
  plot(1, xlim=c(0.6, max(xs) + 0.2), ylim=c(0, 1), type="n", yaxt="n", xaxt="n", ylab="", xlab="", bty="n")

  if (!is.null(argument.ids)) {
    counts <- read.csv(paste("../labels-level", level, ".tsv", sep=""), sep="\t")
    counts <- counts[counts$Argument.ID %in% argument.ids,]
    frequencies <- sapply(labels, function(label) {return(mean(counts[[gsub("[ ,-:]", ".", label)]]))})

    # Plot background bar chart
    bar.space <- (1 - bar.width) / bar.width
    barplot(frequencies, width=bar.width, space=c(0.5 + bar.space, rep(bar.space, length(labels) - 1)), col="lightgray", axes=FALSE, axisnames=FALSE, border=FALSE, add=TRUE)

    drawGrid()

    # Plot baseline
    l = length(subset.names) + 1
    f1s <- 2 * frequencies * 1 / (frequencies + 1)
    drawF1s(f1s, col = plot.colors[l], pch = plot.pch[l], ...)
  } else {
    drawGrid()
  }


  # Draw subset
  drawSubsetByName <- function(subset.name, ...) {
    data.subset <- data.level[data.level[[subset.attribute]] == subset.name,]
    data.subset <- data.subset[match(labels, data.subset$Label),]
    drawF1s(data.subset$F1, ...)
  }
  for (l in 1:length(subset.names)) {
    drawSubsetByName(subset.names[l], col = plot.colors[l], pch = plot.pch[l], ...)
  }

  # Draw Y axis
  axis(2, at=(0:5)/5, line=-1, las=1)
  axis(2, at=(0:4)/5 + 0.1, labels=FALSE, line=-1, tck=-0.025)
  mtext("F1-score", side=2, line=1.5, srt=90)

  # Draw X axis
  axis(1, at=xs, labels=FALSE)
  xaxt.srt <- 0
  xaxt.adj <- c(0.5, 1)
  if (xaxt.vertical) {
    xaxt.adj <- c(1, 0.5)
    xaxt.srt <- 45
  }
  text(xs, y=-0.10, xpd=TRUE, labels=c(sapply(labels, getPlotLabel), "Mean"), srt=xaxt.srt, adj=xaxt.adj)
  mtext(level.name, side=3, line=0)

  # Draw legend
  if (legend.pos != FALSE) {
    legend.names <- subset.names
    if (!is.null(argument.ids)) {
      legend.names <- c(legend.names, "1-Baseline")
    }
    legend(legend.pos, legend=legend.names, col=plot.colors[1:length(legend.names)], box.col="white", bg="white", pch=plot.pch[1:length(legend.names)])
  }
}

parallelCoordinatesPdf <- function(filename, width, height=5, mar.bottom=2.5, mar.left=2.5, ...) {
  pdf(filename, width, height)
  par(mar=c(mar.bottom, mar.left, 1, 0))
  parallelCoordinates(...)
  dev.off()
}

parallelCoordinatesLevelPdfs <- function(filename.prefix, levels=c("1", "2", "3", "4a", "4b"), widths=level.widths, heights=level.heights, mars.bottom=level.mars.bottom, mars.left=level.mars.left, ...) {
  for (i in 1:length(levels)) {
    level <- levels[i]
    parallelCoordinatesPdf(paste(filename.prefix, "-level", level, ".pdf", sep=""), width=widths[i], height=heights[i], level=level, level.number=i, mar.bottom=mars.bottom[i], mar.left=mars.left[i], ...)
  }
}

argument.ids.usa.test <- arguments[arguments$Usage == "test" & arguments$Part == "usa",]$Argument.ID
data.usa <- data[data$Test.dataset == "usa", ]
methods <- c("BERT", "SVM")
parallelCoordinatesLevelPdfs("parallel-coordinates-values-in-arguments-us", data = data.usa, subset.attribute = "Method", subset.names = methods, argument.ids = argument.ids.usa.test)

data.bert <- data[data$Method == "BERT", ]
parallelCoordinatesLevelPdfs("parallel-coordinates-values-in-arguments-bert", data = data.bert, subset.attribute = "Test.dataset")


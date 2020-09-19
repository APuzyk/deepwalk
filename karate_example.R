library(ggplot2)
library(data.table)
weights <- data.table::fread("karate_weights.txt", sep = " ", header=FALSE)
names(weights) <- c("node_id", paste0("v", c(1:(ncol(weights)-1))))
karate_edges <- data.table::fread("karate_network.txt")
communities <- data.table::fread("karate_communities.txt")
weights <- merge(weights, communities)
segments <- data.table::as.data.table(t(apply(karate_edges, 1, function(x) unlist(lapply(x, function(y) weights[node_id==y, c(2, 3)])))))
names(segments) <- paste0("V", c(1:ncol(segments)))
weights[, community:=as.factor(community)]
p <- ggplot() + 
  geom_point(data=weights, aes(x=v1, y=v2, color=community, size = 1)) +
  geom_segment(data=segments, aes(x=V1, y=V2, xend=V3, yend=V4), size=0.25) +
  theme(legend.position="none",
        axis.title = element_blank(),
        axis.text = element_text(size=12))

ggsave("karate_2d.png", plot=p, device=png())

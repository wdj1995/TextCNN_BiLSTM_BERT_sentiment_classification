library(igraph)
library(ggraph)
library(ggplot2)
library(rgexf)
library(xml2)
library(readxl)
library(circlize)
library(readxl)
library(tidygraph)
library(colorspace)
# hcl_palettes(plot = TRUE)
library(RColorBrewer)
# display.brewer.all()
# install.packages("colorspace")
# install.packages("patchwork")
library(stringr)
library(gridExtra)
# install.packages("wrapr")  # 用于自动换行
# 假设 Excel 文件名为 "example.xlsx"，并且位于当前工作目录下
node_df <- read_excel("./od_df_country_count_source_target_node.xlsx")
edge_df <- read_excel("./od_df_country_count_source_target_edge.xlsx")

# 添加节点的所属地区类别
node_region <- as.factor(node_df$Region)
node_degree <- node_df$degree
node_class <- as.factor(node_df$modularity_class)
node_name <- node_df$label

# 添加边的权重属性
edge_weight <- edge_df$count

# # 检查边数据框中是否有NA/NaN值
# # 检查数据框中是否存在任何NA值
# any_na <- anyNA(edge_df)
# # 输出结果
# if (any_na) {
#   message("数据框中存在空值。")
# } else {
#   message("数据框中不存在空值。")
# }
# 创建一个空的图
g <- tbl_graph(nodes = node_df, edges = edge_df)

# 添加节点属性
# V(g)$node_category <- sample(c("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"), 
#                              vcount(g), replace = TRUE)
# V(g)$label_category <- sample(c('Northern Africa','Sub-Saharan Africa',
#                                 'Latin America and the Caribbean','Northern America',
#                                 'Eastern Asia', 'South-eastern Asia', 'Southern Asia', 'Western Asia',
#                                 'Eastern Europe', 'Northern Europe', 'Southern Europe', 'Western Europe',
#                                 'Australia and New Zealand'), 
#                               vcount(g), replace = TRUE)

# 定义不同的颜色映射
modularity_colors <- c('0'="#E64B35B2",'1'="#4DBBD5B2",'2'="#00A087B2",'3'="#3C5488B2",'4'="#F39B7FB2",
            '5'="#8491B4B2",'6'="#91D1C2B2",'7'="#DC0000B2",'8'="#7E6148B2",'9'="#8EA325",
            '10'="#4387B5",'11'="#F5AE6B",'12'="#737373")

region_colors <- c('Northern Africa'="#ffe788",'Sub-Saharan Africa'="#ffc556",
                   'Latin America and the Caribbean'="#f06152",'Northern America'="#DC0000B2",
                   'Eastern Asia'="#5066a1",'South-eastern Asia'="#76afda",'Southern Asia'="#0fbcf9",'Western Asia'="#34e7e4",
                   'Eastern Europe'="#b0d45d",'Northern Europe'="#7fb961", 'Southern Europe'="#4c9568",'Western Europe'="#356d67",
                   'Australia and New Zealand'="#737373")

# 自动换行
# node_df$label <- str_wrap(node_df$label, width = 2)  # 设定换行宽度

# # 将这些颜色映射到图的节点上
# V(g)$node_category <- sample(c("A", "B", "C"), vcount(g), replace = TRUE)
# V(g)$label_category <- sample(c("X", "Y", "Z"), vcount(g), replace = TRUE)

# p_1 <- ggraph(g, layout='circle') +
#   geom_edge_bend(mapping=aes(edge_width=edge_weight, color="#F5AE6B"), strength=0.3, alpha=0.3) +
#   scale_edge_width(range = c(0.2, 3)) + # 设置宽度的范围
#   geom_node_point(aes(size=node_degree, colour=node_class), alpha=1) +  # 设置节点大小和颜色
#   # geom_node_text(aes(x=1.0275*x, y=1.0275*y,
#   #                    label=node_name, color=node_region,
#   #                    angle=-((-node_angle(x, y) + 90) %% 180) + 90),
#   #                hjust='outward', size=2.4) + # 设置点的注释
# 
#   # coord_cartesian(ylim = c(0, 10), xlim = c(0, 10)) +
#   scale_color_manual(values = modularity_colors) +
# 
#   # coord_flip() +
#   coord_fixed(clip = "off") +
# 
#   theme_void(base_family = "", base_size = 12) + # 设置主题，背景为白色
#   theme(legend.position = "none",
#         plot.title = element_blank(),
#         axis.title.y = element_blank(),
#         axis.title.x.top = element_blank(),
#         axis.text = element_blank(),
#         axis.ticks = element_blank(),
#         axis.line = element_blank(),
#         axis.ticks.length = unit(0, "pt"),
#         legend.title = element_blank(),
#         legend.text = element_blank(),
#         panel.grid.major = element_blank(),
#         legend.key = element_blank(),
#         plot.margin = unit(rep(9, 4), units = "mm")
#         )

p_2 <- ggraph(g, layout='circle') +

  geom_edge_bend(mapping=aes(edge_width=edge_weight, color="#F5AE6B"), strength=0.4, alpha=0.2) +

  geom_node_point(aes(size=node_degree, colour=node_class), alpha=1) +  # 设置节点大小和颜色

  geom_node_text(aes(x=1.0275*x, y=1.0275*y,
                     label=node_name, color=node_region,
                     angle=-((-node_angle(x, y) + 90) %% 180) + 90),
                 hjust='outward', size=2.4) + # 设置点的注释

  scale_color_manual(values = region_colors) +

  # labs(color = 'Geographic Regions Worldwide') +

  coord_fixed(clip = "on") +

  theme_void(base_family = "", base_size = 12) + # 设置主题，背景为白色
  theme(
        plot.title = element_blank(),
        axis.title.y = element_blank(),
        axis.title.x.top = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        axis.line = element_blank(),
        axis.ticks.length = unit(0, "pt"),
        # legend.title = element_blank(),
        # legend.text = element_blank(),
        panel.grid.major = element_blank(),
        legend.key = element_blank(),
        plot.margin = unit(rep(9, 4), units = "mm")
  )


# 保存图形
ggsave(filename = "./20241004-images/legend.jpeg", # 文件路径
       device = "jpeg",
       width = 6, height = 6, units = "in",
       dpi = 600) # 每英寸点数 (DPI)


library(ggplot2)
library(cowplot)
library(sitools)
library(viridis)
library(dplyr)

W=4.804
H=2
S=1
point_size=0.8
line_size=0.5
linecolors=scale_color_brewer(palette="Set1")
theme = theme_cowplot(font_size=7)

sisec=Vectorize(function(t)if(is.na(t))NA else sitools::f2si(t / 10^9, 's'))

{
    data = read.csv('../results/n_to_m_saturated/multimat-both.csv', header=T, sep=',')
    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))

    data = subset(data, warps_per_thread_block == 4)

    data = subset(data, right_matrices_per_thread %in% c(1,2,4,8))    
    data = subset(data, left_matrices_per_thread %in% c(1,4,8))    
    data = subset(data, Input.matrix.rows <= 512)

    left_matrices_per_thread.labs = c(`1` = "Left matrices per thread: 1", `2` = "Left matrices per thread: 2",`4` = "Left matrices per thread: 4",`8` = "Left matrices per thread: 8")

    ggsave("n-to-m/multimat-both.pdf", device='pdf', units="in", scale=S, width=W, height=H,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel/Input.left.matrices/Input.right.matrices/(2 * Input.matrix.rows -1 )^2, color=factor(right_matrices_per_thread), shape=factor(right_matrices_per_thread))) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Time per overlap (log-scale)")+
        labs(color="Right matrices per thread", shape="Right matrices per thread") +
        scale_color_manual(values=RColorBrewer::brewer.pal(9,'YlGnBu')[2:9]) +
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512)) +
        facet_wrap(~left_matrices_per_thread, labeller=as_labeller(left_matrices_per_thread.labs)) +
        theme + background_grid() + theme(legend.position="bottom", axis.text.x = element_text(angle = -20, vjust=0.05))
    )
}

{
    data = read.csv('../results/n_to_m_saturated/multimat-both-multirow-both.csv', header=T, sep=',')
    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))

    data = subset(data, warps_per_thread_block == 4)

    data["comb"] = paste0(data$left_matrices_per_thread,"x",data$right_matrices_per_thread)

    data = subset(data, comb %in% c("1x1", "2x2", "2x1", "2x4", "4x4"))

    data = subset(data, left_rows_per_iteration %in% c(1, 2, 4))
    data = subset(data, shifts_per_thread_right_matrix %in% c(1, 2, 4))

    ggsave("n-to-m/multimat-both-grouped-overlap-LxR.pdf", device='pdf', units="in", scale=S, width=W*2, height=H*2,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel/Input.left.matrices/Input.right.matrices/(2 * Input.matrix.rows -1 )^2, color=comb, shape=comb)) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Time per overlap (log-scale)")+
        labs(color="Left rows x right rows per thread", shape="Left rows x right rows per thread") +
        #scale_color_manual(values=RColorBrewer::brewer.pal(9,'YlGnBu')[2:9]) +
        scale_color_brewer(palette="Paired") +
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512)) +
        facet_grid(shifts_per_thread_right_matrix~left_rows_per_iteration, labeller=label_both) +
        theme + background_grid() + theme(legend.position="bottom", axis.text.x = element_text(angle = -20, vjust=0.05))
    )
}

{
    data = read.csv('../results/n_to_m_saturated/multimat-both-multirow-both.csv', header=T, sep=',')
    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))

    data = subset(data, warps_per_thread_block == 4)

    data["Multimat"] = paste0(data$left_matrices_per_thread,"x",data$right_matrices_per_thread)
    data["comb"] = paste0(data$left_rows_per_iteration,"x",data$shifts_per_thread_right_matrix)

    data = subset(data, Multimat %in% c("1x1", "2x2", "4x4"))
    data = subset(data, comb %in% c("1x1","2x2","4x4"))

    ggsave("n-to-m/multimat-both-grouped-overlap.pdf", device='pdf', units="in", scale=S, width=W, height=H,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel/Input.left.matrices/Input.right.matrices/(2 * Input.matrix.rows -1 )^2, color=comb, shape=comb)) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Time per overlap (log-scale)")+
        labs(color="Left rows x shifts per thread", shape="Left rows x shifts per thread") +
        #scale_color_manual(values=RColorBrewer::brewer.pal(9,'YlGnBu')[2:9]) +
        scale_color_brewer(palette="Paired") +
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512)) +
        facet_wrap(~Multimat, labeller=label_both) +
        theme + background_grid() + theme(legend.position="bottom", axis.text.x = element_text(angle = -20, vjust=0.05))
    )
}

{
    data = read.csv('../results/n_to_m/multimat-both-work-distribution.csv', header=T, sep=',')

    data = subset(data, distribution_type == "triangle")

    data_basic = read.csv('../results/n_to_m/multimat-both.csv', header=T, sep=',')

    data_basic["rows_per_thread"] = "ALL"
    
    data_basic = subset(data_basic, select=c(Kernel, Input.matrix.rows, Input.left.matrices, Input.right.matrices, rows_per_thread, right_matrices_per_thread, left_matrices_per_thread, warps_per_thread_block))
    data = subset(data, select=c(Kernel, Input.matrix.rows, Input.left.matrices, Input.right.matrices, rows_per_thread, right_matrices_per_thread, left_matrices_per_thread, warps_per_thread_block))
    data = rbind(data, data_basic)

    data["Multimat"] = paste0(data$left_matrices_per_thread,"x",data$right_matrices_per_thread)
    data["Matrices"] = paste0(data$Input.left.matrices,"x",data$Input.right.matrices)

    data = subset(data, Matrices %in% c("2x2", "8x8", "32x32"))
    data = subset(data, Multimat %in% c("1x1", "2x2", "2x4", "4x4"))
    data = subset(data, rows_per_thread %in% c(1, 4, 16, "ALL"))
    
    data = subset(data, warps_per_thread_block == 4)

    data_s <- split(data, paste0(data$Matrices, data$Multimat, data$rows_per_thread))
    data <- NULL
    for (i in 1:length(data_s)){
        tmp = subset(data_s[[i]], !(Kernel %in% boxplot(data_s[[i]]$Kernel, plot = FALSE)$out))
        data <- rbind(data, tmp)
    }

    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))

    ggsave("n-to-m/multimat-both-split-row-M.pdf", device='pdf', units="in", scale=S, width=W*2, height=H*2,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel/Input.right.matrices/Input.left.matrices/(2 * Input.matrix.rows -1 )^2, color=rows_per_thread, shape=rows_per_thread)) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Time per overlap (log-scale)")+
        labs(color="Rows per thread", shape="Rows per thread") +
        scale_color_brewer(palette="Paired")+
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512)) +
        facet_grid(Matrices~Multimat, labeller=label_both) +
        theme + background_grid() + theme(legend.position="bottom", axis.text.x = element_text(angle = -20, vjust=0.05))
    )
}

{
    data = read.csv('../results/n_to_m/multimat-both-work-distribution.csv', header=T, sep=',')

    data = subset(data, distribution_type == "triangle")

    data_basic = read.csv('../results/n_to_m/multimat-both.csv', header=T, sep=',')

    data_basic["rows_per_thread"] = "n"
    
    data_basic = subset(data_basic, select=c(Kernel, Input.matrix.rows, Input.left.matrices, Input.right.matrices, rows_per_thread, right_matrices_per_thread, left_matrices_per_thread, warps_per_thread_block))
    data = subset(data, select=c(Kernel, Input.matrix.rows, Input.left.matrices, Input.right.matrices, rows_per_thread, right_matrices_per_thread, left_matrices_per_thread, warps_per_thread_block))
    data = rbind(data, data_basic)

    data["Multimat"] = paste0(data$left_matrices_per_thread,"x",data$right_matrices_per_thread)
    data["Matrices"] = paste0(data$Input.left.matrices,"x",data$Input.right.matrices)

    data = subset(data, Matrices %in% c("2x2", "8x8", "32x32"))
    data = subset(data, Multimat %in% c("4x4"))
    data = subset(data, rows_per_thread %in% c(1, 4, 16, "n"))
    
    data$rows_per_thread = factor(data$rows_per_thread, levels=c("1", "2", "4", "8", "16", "32", "n"))
    data$Matrices = factor(data$Matrices, levels=c("2x2", "8x8", "32x32"))
    
    data = subset(data, warps_per_thread_block == 4)

    data_s <- split(data, paste0(data$Matrices, data$Multimat, data$rows_per_thread))
    data <- NULL
    for (i in 1:length(data_s)){
        tmp = subset(data_s[[i]], !(Kernel %in% boxplot(data_s[[i]]$Kernel, plot = FALSE)$out))
        data <- rbind(data, tmp)
    }

    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))

    ggsave("n-to-m/multimat-both-split-row.pdf", device='pdf', units="in", scale=S, width=W, height=H,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel/Input.right.matrices/Input.left.matrices/(2 * Input.matrix.rows -1 )^2, color=rows_per_thread, shape=rows_per_thread)) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Time per overlap (log-scale)")+
        labs(color="Rows per thread", shape="Rows per thread") +
        scale_color_brewer(palette="Paired")+
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512)) +
        facet_wrap(~Matrices, labeller=label_both) +
        theme + background_grid() + theme(legend.position="bottom", axis.text.x = element_text(angle = -20, vjust=0.05))
    )
}

{
    data = c()

    data_original = read.csv('../results/n_to_m/original.csv', header=T, sep=',')
    data_original["alg"] = "overlap-wise"
    data_original = subset(data_original, select=c(alg, Kernel, Input.matrix.rows, Input.type))

    data_wd = read.csv('../results/n_to_m/multimat-both-work-distribution.csv', header=T, sep=',')
    data_wd["alg"] = "split-row"
    data_wd = subset(data_wd, warps_per_thread_block == 4)
    data_wd = subset(data_wd, distribution_type == "triangle")
    data_wd = subset(data_wd, rows_per_thread == 1)
    data_wd = subset(data_wd, right_matrices_per_thread == 4)
    data_wd = subset(data_wd, left_matrices_per_thread == 4)
    data_wd = subset(data_wd, select=c(alg, Kernel, Input.matrix.rows, Input.type))

    data_mb = read.csv('../results/n_to_m/multimat-both-multirow-both.csv', header=T, sep=',')
    data_mb["alg"] = "grouped-overlap"
    data_mb = subset(data_mb, warps_per_thread_block == 4)
    data_mb = subset(data_mb, left_rows_per_iteration == 4)
    data_mb = subset(data_mb, shifts_per_thread_right_matrix == 4)
    data_mb = subset(data_mb, right_matrices_per_thread == 4)
    data_mb = subset(data_mb, left_matrices_per_thread == 4)
    data_mb = subset(data_mb, select=c(alg, Kernel, Input.matrix.rows, Input.type))
    
    data = rbind(data, data_original, data_wd, data_mb)
    data = subset(data, Input.matrix.rows <= 256)

    data["Matrices"] = factor(data$Input.type, levels=c("2x2", "8x8", "32x32"))
    
    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))

    ggsave("n-to-m/n-to-m.pdf", device='pdf', units="in", scale=S, width=W, height=H,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel, color=alg, shape=alg)) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Wall time (log-scale)")+
        scale_color_brewer(palette="Set1")+
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512)) +
        facet_wrap(~Matrices, labeller=label_both, scales="free_y") +
        theme + background_grid() + theme(legend.position="bottom", axis.text.x = element_text(angle = -20, vjust=0.05))
    )
}

{
    data = c()

    data_wd = read.csv('../results/n_to_m/multimat-both-work-distribution.csv', header=T, sep=',')
    data_wd["alg"] = "split-row"
    data_wd = subset(data_wd, warps_per_thread_block == 4)
    data_wd = subset(data_wd, distribution_type == "triangle")
    data_wd = subset(data_wd, rows_per_thread == 1)
    data_wd = subset(data_wd, right_matrices_per_thread == 4)
    data_wd = subset(data_wd, left_matrices_per_thread == 4)
    data_wd = subset(data_wd, select=c(alg, Kernel, Input.matrix.rows, Input.type))

    data_mb = read.csv('../results/n_to_m/multimat-both-multirow-both.csv', header=T, sep=',')
    data_mb["alg"] = "grouped-overlap"
    data_mb = subset(data_mb, warps_per_thread_block == 4)
    data_mb = subset(data_mb, left_rows_per_iteration == 4)
    data_mb = subset(data_mb, shifts_per_thread_right_matrix == 4)
    data_mb = subset(data_mb, right_matrices_per_thread == 4)
    data_mb = subset(data_mb, left_matrices_per_thread == 4)
    data_mb = subset(data_mb, select=c(alg, Kernel, Input.matrix.rows, Input.type))

    data_fft = read.csv('../results/n_to_m/fft.csv', header=T, sep=',')
    data_fft["alg"] = "fft"
    data_fft["Kernel"] = data_fft["Forward.FFT"] + data_fft["Inverse.FFT"] + data_fft["Hadamard"]
    data_fft = subset(data_fft, select=c(alg, Kernel, Input.matrix.rows, Input.type))

    data_fft2 = read.csv('../results/n_to_m/fft.csv', header=T, sep=',')
    data_fft2["alg"] = "fft+prepare"
    data_fft2["Kernel"] = data_fft["Kernel"] + data_fft2["Plan"]
    data_fft2_s <- split(data_fft2, data_fft2$Input.size)
    data_fft2 <- NULL
    for (i in 1:length(data_fft2_s)){
        tmp = subset(data_fft2_s[[i]], !(Kernel %in% boxplot(data_fft2_s[[i]]$Kernel, plot = FALSE)$out))
        data_fft2 <- rbind(data_fft2, tmp)
    }
    data_fft2 = subset(data_fft2, select=c(alg, Kernel, Input.matrix.rows, Input.type))
    
    data = rbind(data, data_wd, data_mb, data_fft, data_fft2)
    data = subset(data, Input.matrix.rows <= 256)
    data["Matrices"] = factor(data$Input.type, levels=c("2x2", "8x8", "32x32"))

    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))

    ggsave("n-to-m/n-to-m-fft.pdf", device='pdf', units="in", scale=S, width=W, height=H,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel, color=alg, shape=alg)) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Wall time (log-scale)")+
        scale_color_brewer(palette="Set1")+
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512)) +
        facet_wrap(~Matrices, labeller=label_both, scales="free_y") +
        theme + background_grid() + theme(legend.position="bottom", axis.text.x = element_text(angle = -20, vjust=0.05))
    )
}

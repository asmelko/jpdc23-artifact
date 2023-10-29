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
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel/Input.left.matrices/Input.right.matrices/Input.matrix.rows^4, color=factor(right_matrices_per_thread), shape=factor(right_matrices_per_thread))) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Time per FMA (log-scale)")+
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

    data = subset(data, Multimat %in% c("1x1", "1x4", "4x4"))
    data = subset(data, comb %in% c("1x1","2x2","4x4"))

    data["Multimat"][data["Multimat"] == "1x1"] <- "no multi-matrix"
    data["Multimat"][data["Multimat"] == "1x4"] <- "multi-matrix-right (4)"
    data["Multimat"][data["Multimat"] == "4x4"] <- "multi-matrix-both (4x4)"

    data["Multimat"] = factor(data$Multimat, levels=c("no multi-matrix", "multi-matrix-right (4)", "multi-matrix-both (4x4)"))

    ggsave("n-to-m/multimat-both-grouped-overlap.pdf", device='pdf', units="in", scale=S, width=W, height=H,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel/Input.left.matrices/Input.right.matrices/Input.matrix.rows^4, color=factor(shifts_per_thread_right_matrix), shape=factor(shifts_per_thread_right_matrix))) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Time per FMA (log-scale)")+
        labs(color="Grouped overlaps per thread", shape="Grouped overlaps per thread") +
        #scale_color_manual(values=RColorBrewer::brewer.pal(9,'YlGnBu')[2:9]) +
        scale_color_brewer(palette="Paired") +
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512)) +
        facet_wrap(~Multimat) +
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
    data["Matrices"] = paste0(data$Input.left.matrices,"-to-",data$Input.right.matrices)

    data = subset(data, Matrices %in% c("2-to-2", "8-to-8", "32-to-32"))
    data = subset(data, Multimat %in% c("1x1", "2x2", "2x4", "4x4"))
    data = subset(data, rows_per_thread %in% c(1, 4, 16, "ALL"))
    
    data = subset(data, warps_per_thread_block == 4)

    data_s <- split(data, paste0(data$Matrices, data$Multimat, data$rows_per_thread, data$Input.matrix.rows))
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
    data["Matrices"] = paste0(data$Input.left.matrices,"-to-",data$Input.right.matrices)

    data = subset(data, Matrices %in% c("2-to-2", "8-to-8"))
    data = subset(data, Multimat %in% c("1x1", "1x4", "4x4"))
    data = subset(data, rows_per_thread %in% c(1, "n"))
    
    data["Multimat"][data["Multimat"] == "1x1"] <- "no multi-matrix"
    data["Multimat"][data["Multimat"] == "1x4"] <- "multi-matrix-right (4)"
    data["Multimat"][data["Multimat"] == "4x4"] <- "multi-matrix-both (4x4)"

    data["Multimat"] = factor(data$Multimat, levels=c("no multi-matrix", "multi-matrix-right (4)", "multi-matrix-both (4x4)"))

    data$rows_per_thread[data$rows_per_thread == 1] = "split-row (1 row per thread)"
    data$rows_per_thread[data$rows_per_thread == "n"] = "no split-row"
    
    data = subset(data, warps_per_thread_block == 4)

    data_s <- split(data, paste0(data$Matrices, data$Multimat, data$rows_per_thread, data$Input.matrix.rows))
    data <- NULL
    for (i in 1:length(data_s)){
        tmp = subset(data_s[[i]], !(Kernel %in% boxplot(data_s[[i]]$Kernel, plot = FALSE)$out))
        data <- rbind(data, tmp)
    }

    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))

    # print(subset(data, Input.matrix.rows == 16 & Multimat == "4x4" & Matrices == "Matrices: 8x8" & rows_per_thread == "the finest split-row (1)")$Kernel)   
    # print(subset(data, Input.matrix.rows == 16 & Multimat == "4x4" & Matrices == "Matrices: 8x8" & rows_per_thread == "no split-row")$Kernel)

    ggsave("n-to-m/multimat-both-split-row.pdf", device='pdf', units="in", scale=S, width=W, height=H,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel/Input.right.matrices/Input.left.matrices/(2 * Input.matrix.rows -1 )^2, color=Multimat, shape=Multimat)) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Time per overlap (log-scale)")+
        labs(color="multi-matrix", shape="multi-matrix") +
        scale_color_brewer(palette="Paired")+
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512)) +
        facet_grid(Matrices~rows_per_thread, scales="free_y") +
        theme + background_grid() + theme(legend.position="bottom", axis.text.x = element_text(angle = -20, vjust=0.05))
    )
}

{
    data = c()

    data_original = read.csv('../results/n_to_m/original.csv', header=T, sep=',')
    data_original["Algorithm"] = "overlap-wise"
    data_original = subset(data_original, select=c(Algorithm, Kernel, Input.matrix.rows, Input.left.matrices, Input.right.matrices))

    data_wd = read.csv('../results/n_to_m/multimat-both-work-distribution.csv', header=T, sep=',')
    data_wd["Algorithm"] = "multimat-both split-row"
    data_wd = subset(data_wd, warps_per_thread_block == 4)
    data_wd = subset(data_wd, distribution_type == "triangle")
    data_wd = subset(data_wd, rows_per_thread == 1)
    data_wd = subset(data_wd, right_matrices_per_thread == 4)
    data_wd = subset(data_wd, left_matrices_per_thread == 4)
    data_wd = subset(data_wd, select=c(Algorithm, Kernel, Input.matrix.rows, Input.left.matrices, Input.right.matrices))

    data_mb = read.csv('../results/n_to_m/multimat-both-multirow-both.csv', header=T, sep=',')
    data_mb["Algorithm"] = "multimat-both grouped-overlap"
    data_mb = subset(data_mb, warps_per_thread_block == 4)
    data_mb = subset(data_mb, left_rows_per_iteration == 4)
    data_mb = subset(data_mb, shifts_per_thread_right_matrix == 4)
    data_mb = subset(data_mb, right_matrices_per_thread == 4)
    data_mb = subset(data_mb, left_matrices_per_thread == 4)
    data_mb = subset(data_mb, select=c(Algorithm, Kernel, Input.matrix.rows, Input.left.matrices, Input.right.matrices))

    data_fft = read.csv('../results/n_to_m/fft.csv', header=T, sep=',')
    data_fft["Algorithm"] = "fft"
    data_fft["Kernel"] = data_fft["Forward.FFT"] + data_fft["Inverse.FFT"] + data_fft["Hadamard"]
    data_fft = subset(data_fft, select=c(Algorithm, Kernel, Input.matrix.rows, Input.left.matrices, Input.right.matrices))

    data_fft2 = read.csv('../results/n_to_m/fft.csv', header=T, sep=',')
    data_fft2["Algorithm"] = "fft+plan"
    data_fft2["Kernel"] = data_fft["Kernel"] + data_fft2["Plan"]
    data_fft2_s <- split(data_fft2, data_fft2$Input.size)
    data_fft2 <- NULL
    for (i in 1:length(data_fft2_s)){
        tmp = subset(data_fft2_s[[i]], !(Kernel %in% boxplot(data_fft2_s[[i]]$Kernel, plot = FALSE)$out))
        data_fft2 <- rbind(data_fft2, tmp)
    }
    data_fft2 = subset(data_fft2, select=c(Algorithm, Kernel, Input.matrix.rows, Input.left.matrices, Input.right.matrices))
    
    data = rbind(data, data_original, data_wd, data_mb, data_fft, data_fft2)
    data = subset(data, Input.matrix.rows <= 256)
    data["Matrices"] = paste0(data$Input.left.matrices,"-to-",data$Input.right.matrices)
    data["Matrices"] = factor(data$Matrices, levels=c("2-to-2", "8-to-8", "32-to-32"))

    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))

    ggsave("n-to-m/n-to-m.pdf", device='pdf', units="in", scale=S, width=W, height=H,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel, color=Algorithm, shape=Algorithm)) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Wall time (log-scale)")+
        scale_color_brewer(palette="Set1")+
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512)) +
        facet_wrap(~Matrices, scales="free_y") +
        theme + background_grid() + theme(legend.position="bottom", axis.text.x = element_text(angle = -20, vjust=0.05))
    )
}

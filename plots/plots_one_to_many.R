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
    data = read.csv('../results/one_to_many_saturated/multimat-right.csv', header=T, sep=',')
    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))

    data = subset(data, warps_per_thread_block == 4)

    data = subset(data, right_matrices_per_thread %in% c(1,2,4,8))    
    data = subset(data, Input.matrix.rows <= 256)

    ggsave("one-to-many/multimat-right.pdf", device='pdf', units="in", scale=S, width=W, height=H,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel/Input.right.matrices/(2 * Input.matrix.rows -1 )^2, color=factor(right_matrices_per_thread), shape=factor(right_matrices_per_thread))) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Time per overlap (log-scale)")+
        labs(color="Right matrices per thread", shape="Right matrices per thread") +
        scale_color_manual(values=RColorBrewer::brewer.pal(9,'YlGnBu')[2:9]) +
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512)) +
        theme + background_grid() + theme(legend.position="bottom", axis.text.x = element_text(angle = -20, vjust=0.05))
    )
}

{
    data = read.csv('../results/one_to_many_saturated/multimat-right-multirow-both.csv', header=T, sep=',')
    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))
    
    data = subset(data, warps_per_thread_block == 4)

    data = subset(data, Input.matrix.rows <= 256)

    data = subset(data, right_matrices_per_thread %in% c(1, 2, 4))
    # data = subset(data, left_rows_per_iteration %in% c(1, 2, 4))
    # data = subset(data, shifts_per_thread_right_matrix %in% c(1, 2, 4))

    data["comb"] = paste0(data$left_rows_per_iteration,"x",data$shifts_per_thread_right_matrix)

    data = subset(data, comb %in% c("1x1","2x2","4x4"))

    ggsave("one-to-many/multimat-right-grouped-overlap.pdf", device='pdf', units="in", scale=S, width=W, height=H,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel/Input.right.matrices/(2 * Input.matrix.rows -1 )^2, color=comb, shape=comb)) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Time per overlap (log-scale)")+
        labs(color="Left rows x shifts per thread", shape="Left rows x shifts per thread") +
        scale_color_brewer(palette="Paired")+
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512)) +
        facet_wrap(~right_matrices_per_thread, labeller=as_labeller(function(x) paste0("Right matrices per thread: ", x))) +
        theme + background_grid() + theme(legend.position="bottom", axis.text.x = element_text(angle = -20, vjust=0.05))
    )
}

{
    data = read.csv('../results/one_to_many_saturated/multimat-right-multirow-both.csv', header=T, sep=',')
    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))
    
    data = subset(data, warps_per_thread_block == 4)

    data = subset(data, Input.matrix.rows <= 256)

    data = subset(data, right_matrices_per_thread %in% c(1, 2, 4))
    data = subset(data, left_rows_per_iteration %in% c(1, 2, 4))
    data = subset(data, shifts_per_thread_right_matrix %in% c(1, 2, 4))

    ggsave("one-to-many/multimat-right-grouped-overlap-LxR.pdf", device='pdf', units="in", scale=S, width=W*2, height=H*2,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel/Input.right.matrices/(2 * Input.matrix.rows -1 )^2, color=factor(right_matrices_per_thread), shape=factor(right_matrices_per_thread))) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Time per overlap (log-scale)")+
        scale_color_brewer(palette="Paired")+
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512)) +
        facet_grid(shifts_per_thread_right_matrix~left_rows_per_iteration, labeller=label_both) +
        theme + background_grid() + theme(legend.position="bottom", axis.text.x = element_text(angle = -20, vjust=0.05))
    )
}

{
    data = read.csv('../results/one_to_many/multimat-right-multirow-both.csv', header=T, sep=',')
    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))
    
    data = subset(data, warps_per_thread_block == 4)

    data = subset(data, Input.matrix.rows <= 256)

    data = subset(data, right_matrices_per_thread %in% c(1, 2, 4))

    data["multirow"] = paste0(data$left_rows_per_iteration,"x",data$shifts_per_thread_right_matrix)
    data["input"] = paste0(data$Input.left.matrices,"x",data$Input.right.matrices)
    
    data = subset(data, multirow %in% c("1x1", "1x2", "2x2", "4x4"))

    ggsave("one-to-many/multimat-right-grouped-overlap-LxR2.pdf", device='pdf', units="in", scale=S, width=W*2, height=H*2,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel/Input.right.matrices/(2 * Input.matrix.rows -1 )^2, color=factor(right_matrices_per_thread), shape=factor(right_matrices_per_thread))) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Time per overlap (log-scale)")+
        scale_color_brewer(palette="Paired")+
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512)) +
        facet_grid(input~multirow, labeller=label_both) +
        theme + background_grid() + theme(legend.position="bottom", axis.text.x = element_text(angle = -20, vjust=0.05))
    )
}

{
    data = read.csv('../results/one_to_many/multimat-right-work-distribution.csv', header=T, sep=',')
    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))

    data = subset(data, distribution_type == "triangle")

    data_basic = read.csv('../results/one_to_many/multimat-right.csv', header=T, sep=',')
    data_basic = data.frame(data_basic %>% group_by_at(names(data_basic)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data_basic))]) %>% summarise(Kernel = mean(Kernel)))

    data_basic["rows_per_thread"] = "n"
    
    data_basic = subset(data_basic, select=c(Kernel, Input.matrix.rows, Input.right.matrices, rows_per_thread, right_matrices_per_thread, warps_per_thread_block))
    data = subset(data, select=c(Kernel, Input.matrix.rows, Input.right.matrices, rows_per_thread, right_matrices_per_thread, warps_per_thread_block))
    data = rbind(data, data_basic)

    data = subset(data, Input.right.matrices %in% c(2, 8, 32))
    data = subset(data, right_matrices_per_thread %in% c(1, 8))
    data = subset(data, rows_per_thread %in% c(1, "n"))
    data = subset(data, warps_per_thread_block == 4)
    data = subset(data, Input.matrix.rows <= 256)

    data["comb"] = paste0(data$right_matrices_per_thread,"x",data$rows_per_thread)
    data["Matrices"] = paste0("1x",data$Input.right.matrices)
    data["Matrices"] = factor(data$Matrices, levels=c("1x2","1x8","1x32"))

    ggsave("one-to-many/multimat-right-split-row.pdf", device='pdf', units="in", scale=S, width=W, height=H,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel, color=comb, shape=comb)) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Wall time (log-scale)")+
        labs(color="Right matrices x rows per thread", shape="Right matrices x rows per thread") +
        scale_color_brewer(palette="Paired")+
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512)) +
        facet_wrap(~Matrices, labeller=label_both, scales="free_y") +
        theme + background_grid() + theme(legend.position="bottom", axis.text.x = element_text(angle = -20, vjust=0.05))
    )
}

{
    data = read.csv('../results/one_to_many/multimat-right-work-distribution.csv', header=T, sep=',')
    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))

    data = subset(data, distribution_type == "triangle")

    data_basic = read.csv('../results/one_to_many/multimat-right.csv', header=T, sep=',')
    data_basic = data.frame(data_basic %>% group_by_at(names(data_basic)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data_basic))]) %>% summarise(Kernel = mean(Kernel)))

    data_basic["rows_per_thread"] = "ALL"
    
    data_basic = subset(data_basic, select=c(Kernel, Input.matrix.rows, Input.right.matrices, rows_per_thread, right_matrices_per_thread, warps_per_thread_block))
    data = subset(data, select=c(Kernel, Input.matrix.rows, Input.right.matrices, rows_per_thread, right_matrices_per_thread, warps_per_thread_block))
    data = rbind(data, data_basic)

    data = subset(data, Input.right.matrices %in% c(2, 8, 32))
    data = subset(data, right_matrices_per_thread %in% c(1, 2, 4, 8))
    data = subset(data, rows_per_thread %in% c(1, 4, 16, "ALL"))
    data = subset(data, warps_per_thread_block == 4)
    data = subset(data, Input.matrix.rows <= 256)

    ggsave("one-to-many/multimat-right-split-row-MxR.pdf", device='pdf', units="in", scale=S, width=W*2, height=H*2,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel/(2 * Input.matrix.rows -1 )^2, color=factor(rows_per_thread), shape=factor(rows_per_thread))) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Time per overlap (log-scale)")+
        scale_color_brewer(palette="Paired")+
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512)) +
        facet_grid(Input.right.matrices~right_matrices_per_thread, labeller=label_both) +
        theme + background_grid() + theme(legend.position="bottom", axis.text.x = element_text(angle = -20, vjust=0.05))
    )
}

{
    data = c()

    data_original = read.csv('../results/one_to_many/original.csv', header=T, sep=',')
    data_original["alg"] = "overlap-wise"
    data_original = subset(data_original, select=c(alg, Kernel, Input.matrix.rows, Input.right.matrices))

    data_wd = read.csv('../results/one_to_many/multimat-right-work-distribution.csv', header=T, sep=',')
    data_wd["alg"] = "split-row"
    data_wd = subset(data_wd, warps_per_thread_block == 4)
    data_wd = subset(data_wd, distribution_type == "triangle")
    data_wd = subset(data_wd, rows_per_thread == 1)
    data_wd = subset(data_wd, right_matrices_per_thread == 8)
    data_wd = subset(data_wd, select=c(alg, Kernel, Input.matrix.rows, Input.right.matrices))

    data_mb = read.csv('../results/one_to_many/multimat-right-multirow-both.csv', header=T, sep=',')
    data_mb["alg"] = "grouped-overlap"
    data_mb = subset(data_mb, warps_per_thread_block == 4)
    data_mb = subset(data_mb, left_rows_per_iteration == 4)
    # data_mb_s = subset(data_mb, shifts_per_thread_right_matrix == 1)
    # data_mb_s = subset(data_mb_s, right_matrices_per_thread == 1)
    # data_mb_s = subset(data_mb_s, Input.matrix.rows < 32)
    # data_mb_b = subset(data_mb, shifts_per_thread_right_matrix == 4)
    # data_mb_b = subset(data_mb_b, right_matrices_per_thread == 4)
    # data_mb_b = subset(data_mb_b, Input.matrix.rows >= 32)
    # data_mb = rbind(data_mb_s, data_mb_b)
    data_mb = subset(data_mb, shifts_per_thread_right_matrix == 4)
    data_mb = subset(data_mb, right_matrices_per_thread == 4)
    data_mb = subset(data_mb, select=c(alg, Kernel, Input.matrix.rows, Input.right.matrices))
    
    data = rbind(data, data_original, data_wd, data_mb)
    data = subset(data, Input.matrix.rows <= 256)
    data = subset(data, Input.right.matrices %in% c(2, 8, 32))

    data["Matrices"] = paste0("1x",data$Input.right.matrices)
    data["Matrices"] = factor(data$Matrices, levels=c("1x2","1x8","1x32"))

    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))

    ggsave("one-to-many/one-to-many.pdf", device='pdf', units="in", scale=S, width=W, height=H,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel, color=alg, shape=alg)) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Wall time (log-scale)")+
        labs(color="Algorithm", shape="Algorithm") +
        scale_color_brewer(palette="Set1")+
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512)) +
        facet_wrap(~Matrices, labeller=label_both, scales="free_y") +
        theme + background_grid() + theme(legend.position="bottom", axis.text.x = element_text(angle = -20, vjust=0.05))
    )
}

{
    data = c()

    data_wd = read.csv('../results/one_to_many/multimat-right-work-distribution.csv', header=T, sep=',')
    data_wd["alg"] = "split-row"
    data_wd = subset(data_wd, warps_per_thread_block == 4)
    data_wd = subset(data_wd, distribution_type == "triangle")
    data_wd = subset(data_wd, rows_per_thread == 1)
    data_wd = subset(data_wd, right_matrices_per_thread == 8)
    data_wd = subset(data_wd, select=c(alg, Kernel, Input.matrix.rows, Input.right.matrices))

    data_mb = read.csv('../results/one_to_many/multimat-right-multirow-both.csv', header=T, sep=',')
    data_mb["alg"] = "grouped-overlap"
    data_mb = subset(data_mb, warps_per_thread_block == 4)
    data_mb = subset(data_mb, left_rows_per_iteration == 4)
    data_mb = subset(data_mb, shifts_per_thread_right_matrix == 4)
    data_mb = subset(data_mb, right_matrices_per_thread == 4)
    data_mb = subset(data_mb, select=c(alg, Kernel, Input.matrix.rows, Input.right.matrices))

    data_fft = read.csv('../results/one_to_many/fft.csv', header=T, sep=',')
    data_fft["alg"] = "fft"
    data_fft["Kernel"] = data_fft["Forward.FFT"] + data_fft["Inverse.FFT"] + data_fft["Hadamard"]
    data_fft = subset(data_fft, select=c(alg, Kernel, Input.matrix.rows, Input.right.matrices))

    data_fft2 = read.csv('../results/one_to_many/fft.csv', header=T, sep=',')
    data_fft2["alg"] = "fft+prepare"
    data_fft2["Kernel"] = data_fft2["Forward.FFT"] + data_fft2["Inverse.FFT"] + data_fft2["Hadamard"] + data_fft2["Plan"]
    data_fft2_s <- split(data_fft2, data_fft2$Input.size)
    data_fft2 <- NULL
    for (i in 1:length(data_fft2_s)){
        tmp = subset(data_fft2_s[[i]], !(Kernel %in% boxplot(data_fft2_s[[i]]$Kernel, plot = FALSE)$out))
        data_fft2 <- rbind(data_fft2, tmp)
    }
    data_fft2 = subset(data_fft2, select=c(alg, Kernel, Input.matrix.rows, Input.right.matrices))
    
    data = rbind(data, data_original, data_wd, data_mb, data_fft, data_fft2)
    data = subset(data, Input.matrix.rows <= 256)
    data = subset(data, Input.right.matrices %in% c(2, 8, 32))

    data["Matrices"] = paste0("1x",data$Input.right.matrices)
    data["Matrices"] = factor(data$Matrices, levels=c("1x2","1x8","1x32"))
    
    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))

    ggsave("one-to-many/one-to-many-fft.pdf", device='pdf', units="in", scale=S, width=W, height=H,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel, color=alg, shape=alg)) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Wall time (log-scale)")+
        labs(color="Algorithm", shape="Algorithm") +
        scale_color_brewer(palette="Set1")+
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512)) +
        facet_wrap(~Matrices, labeller=label_both, scales="free_y") +
        theme + background_grid() + theme(legend.position="bottom", axis.text.x = element_text(angle = -20, vjust=0.05))
    )
}

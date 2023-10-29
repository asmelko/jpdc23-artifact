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
saturation_multiplier=4000

sisec=Vectorize(function(t)if(is.na(t))NA else sitools::f2si(t / 10^9, 's'))

{
    data = read.csv('../results/one_to_one_saturated/multirow_both.csv', header=T, sep=',')
    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))

    # data = subset(data, warps_per_thread_block == 4)
    data = subset(data, shifts_per_thread %in% c(1, 2, 4, 8))
    data = subset(data, left_rows_per_iteration %in% c(1, 2, 4, 8))

    ggsave("one-to-one/grouped-overlap-LxR.pdf", device='pdf', units="in", scale=S, width=W*2, height=H*2,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel/(Input.matrix.rows^4)/saturation_multiplier, color=factor(warps_per_thread_block), shape=factor(warps_per_thread_block))) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Time per FMA op (log-scale)")+
        #scale_color_brewer("Warps per block", palette="Set1")+
        scale_color_manual(values=RColorBrewer::brewer.pal(9,'YlGnBu')[2:9]) +
        labs(color="warps_per_thread_block", shape="warps_per_thread_block") +
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512,1024)) +
        facet_grid(shifts_per_thread~left_rows_per_iteration, labeller=label_both) +
        theme + background_grid() + theme(legend.position="bottom", axis.text.x = element_text(angle = -20, vjust=0.05))
    )
}

{
    data = read.csv('../results/one_to_one_saturated/multirow_both.csv', header=T, sep=',')
    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))

    data = subset(data, warps_per_thread_block == 4)
    
    data["comb"] = paste0(data$left_rows_per_iteration,"x",data$shifts_per_thread)

    data = subset(data, comb %in% c("1x1", "2x2", "4x4", "8x8"))

    ggsave("one-to-one/grouped-overlap.pdf", device='pdf', units="in", scale=S, width=W/1.7, height=H,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel/(Input.matrix.rows^4)/saturation_multiplier, color=factor(shifts_per_thread), shape=factor(shifts_per_thread))) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Time per FMA op (log-scale)")+
        scale_color_manual(values=RColorBrewer::brewer.pal(9,'YlGnBu')[2:9]) +
        labs(color="Grouped overlaps per thread", shape="Grouped overlaps per thread") +
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512,1024)) +
        theme + background_grid() + theme(legend.position="bottom") +guides(color=guide_legend(nrow=2,byrow=TRUE))
    )
}

{
    data = read.csv('../results/one_to_one_saturated/multirow_both.csv', header=T, sep=',')
    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))

    data = subset(data, warps_per_thread_block == 4)
    
    data["comb"] = paste0(data$left_rows_per_iteration,"x",data$shifts_per_thread)

    data = subset(data, comb %in% c("4x1", "4x2", "4x4", "4x8"))

    ggsave("one-to-one/grouped-overlap-L4.pdf", device='pdf', units="in", scale=S, width=W, height=H,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel/(Input.matrix.rows^4)/saturation_multiplier, color=comb, shape=comb)) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Time per FMA op (log-scale)")+
        scale_color_manual(values=RColorBrewer::brewer.pal(9,'YlGnBu')[2:9]) +
        labs(color="Left rows x Shifts per thread", shape="Left rows x Shifts per thread") +
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512,1024)) +
        theme + background_grid() + theme(legend.position="bottom", axis.text.x = element_text(angle = -20, vjust=0.05))
    )
}

{
    data = read.csv('../results/one_to_one_saturated/multirow_both.csv', header=T, sep=',')
    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))

    data = subset(data, warps_per_thread_block == 4)
    
    data["comb"] = paste0(data$left_rows_per_iteration,"x",data$shifts_per_thread)

    data = subset(data, comb %in% c("1x4", "2x4", "4x4", "8x4"))

    ggsave("one-to-one/grouped-overlap-R4.pdf", device='pdf', units="in", scale=S, width=W, height=H,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel/(Input.matrix.rows^4)/saturation_multiplier, color=comb, shape=comb)) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Time per FMA op (log-scale)")+
        scale_color_manual(values=RColorBrewer::brewer.pal(9,'YlGnBu')[2:9]) +
        labs(color="Left rows x Shifts per thread", shape="Left rows x Shifts per thread") +
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512,1024)) +
        theme + background_grid() + theme(legend.position="bottom", axis.text.x = element_text(angle = -20, vjust=0.05))
    )
}

{
    data = read.csv('../results/one_to_one/work_distribution.csv', header=T, sep=',')
    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))
    data = subset(data, distribution_type == "triangle")
    data = subset(data, warps_per_thread_block == 4)

    data_basic = read.csv('../results/one_to_one/basic.csv', header=T, sep=',')
    data_basic = subset(data_basic, warps_per_thread_block == 4)
    data_basic["rows_per_thread"] = "n"
    
    data_basic = subset(data_basic, select=c(Kernel, Input.matrix.rows, rows_per_thread))
    data = subset(data, select=c(Kernel, Input.matrix.rows, rows_per_thread))

    data = rbind(data, data_basic)

    data = subset(data, Input.matrix.rows <= 256)

    rows_factored = factor(data$rows_per_thread, levels=c("1", "2", "4", "8", "16", "32", "n"))

    ggsave("one-to-one/split-row.pdf", device='pdf', units="in", scale=S, width=W/1.7, height=H,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel, color=rows_factored, shape=rows_factored)) +
        geom_line(linewidth=line_size) +
        geom_point(size=point_size) +
        xlab("Matrices size")+
        ylab("Wall time (log-scale)")+
        #scale_color_brewer("Warps per block", palette="Set1")+
        scale_color_manual(values=RColorBrewer::brewer.pal(7,'YlGnBu')[2:7]) +
        labs(color="Overlap rows per thread", shape="Overlap rows per thread") +
        scale_y_log10(labels = sisec) +
        scale_x_continuous(labels = function(x) paste0(x,"x",x), breaks=c(16,64,128,192, 256)) +
        theme + background_grid() + theme(legend.position="bottom")
    )
}

{
    data = read.csv('../results/one_to_one/warp-per-shift-shared-memory.csv', header=T, sep=',')
    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))
    
    data["comb2"] = paste0(data$shifts_per_thread_block,"x",data$shared_mem_row_size)

    data = subset(data, comb2 %in% c("16x32", "16x128", "16x256",  "32x32", "32x128"))

    data$comb2 = factor(data$comb2, levels=c("16x32", "16x128", "16x256",  "32x32", "32x128"))

    ggsave("one-to-one/warp-per-shift-shared-memory.pdf", device='pdf', units="in", scale=S, width=W, height=H*1.5,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel/(2 * Input.matrix.rows -1 )^2, color=comb2)) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Time per overlap (log-scale)")+
        #scale_color_brewer("Warps per block", palette="Set1")+
        scale_color_manual(name="Shifts x shmem row size", values=RColorBrewer::brewer.pal(9,'YlGnBu')[2:9]) +
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512)) +
        facet_grid(column_group_per_block~strided_load, labeller=label_both) +
        theme + background_grid() + theme(legend.position="bottom")
    )
}

{
    data = read.csv('../results/one_to_one/warp-per-shift.csv', header=T, sep=',')
    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))
    
    ggsave("one-to-one/warp-per-shift.pdf", device='pdf', units="in", scale=S, width=W, height=H,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel/(2 * Input.matrix.rows -1 )^2, color=factor(shifts_per_thread_block))) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices size (log-scale)")+
        ylab("Time per overlap (log-scale)")+
        #scale_color_brewer("Warps per block", palette="Set1")+
        scale_color_manual(name="Shifts per block", values=RColorBrewer::brewer.pal(9,'YlGnBu')[2:9]) +
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512)) +
        theme + background_grid() + theme(legend.position="bottom")
    )
}

{
    data = c()

    data_wd = read.csv('../results/one_to_one/work_distribution.csv', header=T, sep=',')
    data_wd["alg"] = "split-row"
    data_wd = subset(data_wd, warps_per_thread_block == 4)
    data_wd = subset(data_wd, rows_per_thread == 1)
    data_wd = subset(data_wd, distribution_type == "triangle")
    data_wd = subset(data_wd, select=c(alg, Kernel, Input.matrix.rows))

    data_wps = read.csv('../results/one_to_one/warp-per-shift.csv', header=T, sep=',')
    data_wps["alg"] = "warp-per-shift"
    data_wps = subset(data_wps, shifts_per_thread_block == 4)
    data_wps = subset(data_wps, select=c(alg, Kernel, Input.matrix.rows))

    data_wpsi = read.csv('../results/one_to_one/warp-per-shift-simple-indexing.csv', header=T, sep=',')
    data_wpsi["alg"] = "warp-per-shift-simple-idx"
    data_wpsi = subset(data_wpsi, shifts_per_thread_block == 4)
    data_wpsi = subset(data_wpsi, select=c(alg, Kernel, Input.matrix.rows))

    data_wpss = read.csv('../results/one_to_one/warp-per-shift-shared-memory.csv', header=T, sep=',')
    data_wpss["alg"] = "warp-per-shift-shared"
    data_wpss = subset(data_wpss, shifts_per_thread_block == 32)
    data_wpss = subset(data_wpss, shared_mem_row_size == 128)
    data_wpss = subset(data_wpss, strided_load == 1)
    data_wpss = subset(data_wpss, column_group_per_block == 1)
    data_wpss = subset(data_wpss, select=c(alg, Kernel, Input.matrix.rows))
    
    data = rbind(data, data_wd, data_wps, data_wpsi, data_wpss)

    data = subset(data, Input.matrix.rows <= 256)
    
    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))

    ggsave("one-to-one/one-to-one-warp-per-shift-and-split-row.pdf", device='pdf', units="in", scale=S, width=W, height=H,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel, color=alg, shape=alg)) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices sizes (log-scale)")+
        ylab("Wall Time (log-scale)")+
        labs(color="Algorithm", shape="Algorithm") +
        scale_color_brewer(palette="Set1")+
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512)) +
        theme + background_grid() + theme(legend.position="bottom")
    )
}

{
    data = c()

    data_original = read.csv('../results/one_to_one/original.csv', header=T, sep=',')
    data_original["alg"] = "overlap-wise"
    data_original = subset(data_original, select=c(alg, Kernel, Input.matrix.rows))

    data_wd = read.csv('../results/one_to_one/work_distribution.csv', header=T, sep=',')
    data_wd["alg"] = "split-row"
    data_wd = subset(data_wd, warps_per_thread_block == 4)
    data_wd = subset(data_wd, rows_per_thread == 1)
    data_wd = subset(data_wd, distribution_type == "triangle")
    data_wd = subset(data_wd, select=c(alg, Kernel, Input.matrix.rows))

    data_mb = read.csv('../results/one_to_one/multirow_both.csv', header=T, sep=',')
    data_mb["alg"] = "grouped-overlap"
    data_mb = subset(data_mb, warps_per_thread_block == 4)
    data_mb = subset(data_mb, shifts_per_thread == 8)
    data_mb = subset(data_mb, left_rows_per_iteration == 8)
    data_mb = subset(data_mb, select=c(alg, Kernel, Input.matrix.rows))

    data_wps = read.csv('../results/one_to_one/warp-per-shift.csv', header=T, sep=',')
    data_wps["alg"] = "warp-per-shift"
    data_wps = subset(data_wps, shifts_per_thread_block == 4)
    data_wps = subset(data_wps, select=c(alg, Kernel, Input.matrix.rows))

    data_fft = read.csv('../results/one_to_one/fft.csv', header=T, sep=',')
    data_fft["alg"] = "fft"
    data_fft["Kernel"] = data_fft["Forward.FFT"] + data_fft["Inverse.FFT"] + data_fft["Hadamard"]
    data_fft = subset(data_fft, select=c(alg, Kernel, Input.matrix.rows))

    data_fft2 = read.csv('../results/one_to_one/fft.csv', header=T, sep=',')
    data_fft2["alg"] = "fft+plan"
    data_fft2["Kernel"] = data_fft2["Forward.FFT"] + data_fft2["Inverse.FFT"] + data_fft2["Hadamard"] + data_fft2["Plan"]
    data_fft2_s <- split(data_fft2, data_fft2$Input.size)
    data_fft2 <- NULL
    for (i in 1:length(data_fft2_s)){
        tmp = subset(data_fft2_s[[i]], !(Kernel %in% boxplot(data_fft2_s[[i]]$Kernel, plot = FALSE)$out))
        data_fft2 <- rbind(data_fft2, tmp)
    }
    data_fft2 = subset(data_fft2, select=c(alg, Kernel, Input.matrix.rows))

    data_original["cmp"] = "overlap-wise comparison"
    data_wd["cmp"] = "overlap-wise comparison"
    data_mb["cmp"] = "overlap-wise comparison"
    data_wps["cmp"] = "overlap-wise comparison"
    data = rbind(data, data_original, data_wd, data_mb, data_wps)

    data_wd["cmp"] = "fft comparison"
    data_mb["cmp"] = "fft comparison"
    data_wps["cmp"] = "fft comparison"
    data_fft["cmp"] = "fft comparison"
    data_fft2["cmp"] = "fft comparison"
    data = rbind(data,  data_wd, data_mb, data_wps, data_fft, data_fft2)
    
    data$cmp = factor(data$cmp, levels=c("overlap-wise comparison", "fft comparison"))
    
    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))

    ggsave("one-to-one/one-to-one.pdf", device='pdf', units="in", scale=S, width=W, height=H,
        ggplot(data, aes(x=Input.matrix.rows,y=Kernel, color=alg, shape=alg)) +
        geom_point(size=point_size) +
        geom_line(linewidth=line_size) +
        xlab("Matrices sizes (log-scale)")+
        ylab("Wall Time (log-scale)")+
        labs(color="Algorithm", shape="Algorithm") +
        scale_color_brewer(palette="Set1")+
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = function(x) paste0(x,"x",x), breaks=c(16,32,64,128,256,512)) +
        facet_wrap(~cmp) +
        theme + background_grid() + theme(legend.position="bottom", axis.text.x = element_text(angle = -20, vjust=0.05))
    )
}

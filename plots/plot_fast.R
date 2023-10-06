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
    data = c()

    data_original = read.csv('../results/one_to_one_fast/original.csv', header=T, sep=',')
    data_original["alg"] = "overlap-wise"
    data_original = subset(data_original, select=c(alg, Kernel, Input.matrix.rows))

    data_wd = read.csv('../results/one_to_one_fast/work_distribution.csv', header=T, sep=',')
    data_wd["alg"] = "split-row"
    data_wd = subset(data_wd, warps_per_thread_block == 4)
    data_wd = subset(data_wd, rows_per_thread == 1)
    data_wd = subset(data_wd, distribution_type == "triangle")
    data_wd = subset(data_wd, select=c(alg, Kernel, Input.matrix.rows))

    data_mb = read.csv('../results/one_to_one_fast/multirow_both.csv', header=T, sep=',')
    data_mb["alg"] = "multirow both"
    data_mb = subset(data_mb, warps_per_thread_block == 4)
    data_mb = subset(data_mb, shifts_per_thread == 8)
    data_mb = subset(data_mb, left_rows_per_iteration == 8)
    data_mb = subset(data_mb, select=c(alg, Kernel, Input.matrix.rows))

    data_wps = read.csv('../results/one_to_one_fast/warp-per-shift.csv', header=T, sep=',')
    data_wps["alg"] = "warp-per-shift"
    data_wps = subset(data_wps, shifts_per_thread_block == 4)
    data_wps = subset(data_wps, select=c(alg, Kernel, Input.matrix.rows))
    
    data = rbind(data, data_original, data_wd, data_mb, data_wps)

    data = subset(data, Input.matrix.rows <= 256)

    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))

    ggsave("one-to-one-fast/one-to-one.pdf", device='pdf', units="in", scale=S, width=W, height=H,
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

    data_wd = read.csv('../results/one_to_one_fast/work_distribution.csv', header=T, sep=',')
    data_wd["alg"] = "split-row"
    data_wd = subset(data_wd, warps_per_thread_block == 4)
    data_wd = subset(data_wd, rows_per_thread == 1)
    data_wd = subset(data_wd, distribution_type == "triangle")
    data_wd = subset(data_wd, select=c(alg, Kernel, Input.matrix.rows))

    data_mb = read.csv('../results/one_to_one_fast/multirow_both.csv', header=T, sep=',')
    data_mb["alg"] = "multirow both"
    data_mb = subset(data_mb, warps_per_thread_block == 4)
    data_mb = subset(data_mb, shifts_per_thread == 8)
    data_mb = subset(data_mb, left_rows_per_iteration == 8)
    data_mb = subset(data_mb, select=c(alg, Kernel, Input.matrix.rows))

    data_wps = read.csv('../results/one_to_one_fast/warp-per-shift.csv', header=T, sep=',')
    data_wps["alg"] = "warp-per-shift"
    data_wps = subset(data_wps, shifts_per_thread_block == 4)
    data_wps = subset(data_wps, select=c(alg, Kernel, Input.matrix.rows))

    data_fft = read.csv('../results/one_to_one_fast/fft.csv', header=T, sep=',')
    data_fft["alg"] = "fft"
    data_fft["Kernel"] = data_fft["Forward.FFT"] + data_fft["Inverse.FFT"] + data_fft["Hadamard"]
    data_fft = subset(data_fft, select=c(alg, Kernel, Input.matrix.rows))

    data_fft2 = read.csv('../results/one_to_one_fast/fft.csv', header=T, sep=',')
    data_fft2["alg"] = "fft+prepare"
    data_fft2["Kernel"] = data_fft2["Forward.FFT"] + data_fft2["Inverse.FFT"] + data_fft2["Hadamard"] + data_fft2["Plan"]
    data_fft2_s <- split(data_fft2, data_fft2$Input.size)
    data_fft2 <- NULL
    for (i in 1:length(data_fft2_s)){
        tmp = subset(data_fft2_s[[i]], !(Kernel %in% boxplot(data_fft2_s[[i]]$Kernel, plot = FALSE)$out))
        data_fft2 <- rbind(data_fft2, tmp)
    }
    data_fft2 = subset(data_fft2, select=c(alg, Kernel, Input.matrix.rows))
    
    data = rbind(data, data_wd, data_mb, data_wps, data_fft, data_fft2)

    data = subset(data, Input.matrix.rows <= 256)
    
    data = data.frame(data %>% group_by_at(names(data)[-grep("(Kernel)|(Kernel_iterations)|(X)|(Args)", names(data))]) %>% summarise(Kernel = mean(Kernel)))

    ggsave("one-to-one-fast/one-to-one-fft.pdf", device='pdf', units="in", scale=S, width=W, height=H,
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

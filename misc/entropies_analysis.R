library(lsr)
library(ggplot2)
theme_set(theme_minimal())
library(dplyr)
source("data_io.r")

SUBJECTS_TYPE = 'C'
CLUSTER_TYPE = 'POSTERIOR'

control_tibbles <- read_entropies_tibbles('C', CLUSTER_TYPE)
total_control_tibble <- do.call(rbind, control_tibbles)
# Extract the mean value
control_plot_tibble <- tidyr::pivot_longer(summarise_all(total_control_tibble, mean), everything())
control_plot_tibble$name <- c(1:20)
# Extract the CIs
ci_tibble <- as_tibble(ciMean(total_control_tibble))
col_names <- c(ci_inf = "2.5%", ci_sup = "97.5%")
ci_tibble <- rename(ci_tibble, all_of(col_names))
control_plot_tibble <- cbind(control_plot_tibble, ci_tibble)


mci_tibbles <- read_entropies_tibbles('F', CLUSTER_TYPE)
total_mci_tibble <- do.call(rbind, mci_tibbles)
# Extract the mean value
mci_plot_tibble <- tidyr::pivot_longer(summarise_all(total_mci_tibble, mean), everything())
mci_plot_tibble$name <- c(1:20)
# Extract the CIs
ci_tibble <- as_tibble(ciMean(total_mci_tibble))
col_names <- c(ci_inf = "2.5%", ci_sup = "97.5%")
ci_tibble <- rename(ci_tibble, all_of(col_names))
mci_plot_tibble <- cbind(mci_plot_tibble, ci_tibble)



ggplot(control_plot_tibble, aes(x=name)) +
  xlab("Scale") + ylab("RCmvMFE value") + 
  geom_line(aes(y = value), color = "blue") + 
  geom_line(aes(y = ci_inf), color = "blue", linetype="dashed") + 
  geom_line(aes(y = ci_sup), color="blue", linetype="dashed") +

  geom_line(aes(y = mci_plot_tibble$value), color = "red") + 
  geom_line(aes(y = mci_plot_tibble$ci_inf), color = "red", linetype="dashed") + 
  geom_line(aes(y = mci_plot_tibble$ci_sup), color="red", linetype="dashed") 

#  stat_summary(geom="ribbon", fun.data=mean_cl_boot, 
#               conf.int=0.95, alpha = 0.0, linetype="dashed", colour="red")


library("eegUtils")
library(dplyr)
source("data_io.R")
source("rcmvmfe.R")
library(reticulate)
# NOTE: provide a valid python interpreter path here
use_python("/Users/francescosheiban/.pyenv/versions/3.11.1/bin/python")
source_python("rcmvmfe.py")

SOURCE_CLUSTERS <- list(
  ANTERIOR = c("Fp1", "F3", "Fz", "Fp2", "F4"),
  CENTRAL = c("C3","Cz", "C4"),
  TEMPORAL_LEFT = c("T3","T5", "F7"),
  TEMPORAL_RIGHT = c("T4","T6", "F8"),
  POSTERIOR = c("P3", "O1", "Pz", "P4", "O2")
  )


# Access the folder in which the data is stored
DERIVATIVES_DIR <- "ds004504-download/derivatives/"

# These quantities control the number of patients whose data is loaded during function execution.
# Due to the amount of data, it is not feasible to load and keep in memory the EEG data of all the 88 patients at the same time
args <- commandArgs(trailingOnly = TRUE)
START <- args[1]
END <- args[2]
cluster_type <- args[3]

tibble_list <- read_preprocessed_eeg(DERIVATIVES_DIR, START, END)

# Define the entropy algorithm parameters 
M <- 2
R <- 0.15
N <- 2
TAU <- 1
MAX_SCALE <- 20
# The number of samples to consider (500Hz -> 5000 samples = 10 sec)
N_SAMPLES <- 5000

subject_num <- as.numeric(START)

for (tibble in tibble_list) {
  print("Computing entropy for a new subject")
  sources <- SOURCE_CLUSTERS[[cluster_type]]
  subject_entropy <- compute_subect_rcmvmfe(tibble, samples_per_segment=2500, cluster=sources)
  df <- data.frame(matrix(unlist(subject_entropy), ncol=20, byrow=TRUE))
  colnames(df) <- c(1:20)
  write.csv(df, sprintf("subjects_entropies/subject_%s_%s.csv", subject_num, cluster_type), row.names=FALSE)
  subject_num <- subject_num + as.numeric(1)
}
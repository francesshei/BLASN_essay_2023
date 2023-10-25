library(dplyr)
source("data_io.R")
library(reticulate)
# NOTE: provide a valid python interpreter path here
use_python("/Users/francescosheiban/.pyenv/versions/3.11.1/bin/python")
source_python("entropies.py")


SOURCE_CLUSTERS <- list(
  ANTERIOR = c("Fp1", "F3", "Fz", "Fp2", "F4"),
  CENTRAL = c("C3","Cz", "C4"),
  TEMPORAL_LEFT = c("T3","T5", "F7"),
  TEMPORAL_RIGHT = c("T4","T6", "F8"),
  POSTERIOR = c("P3", "O1", "Pz", "P4", "O2")
)


# Access the folder in which the data is stored
DERIVATIVES_DIR <- "ds004504-download/derivatives/"


tibble_list <- read_preprocessed_eeg(DERIVATIVES_DIR, 1,1)
control_tibble_list <- read_preprocessed_eeg(DERIVATIVES_DIR, 42,42)

sources <- SOURCE_CLUSTERS[["ANTERIOR"]]

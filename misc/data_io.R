library("eegUtils")
library("tibble")

read_preprocessed_eeg <- function(input_directory, starting_index, ending_index) {
  available_subjects <- list.files(input_directory)
  print(available_subjects)
  # Create the subfolders list automatically with a "do.call" mapping
  subjects_folders <- do.call(
    paste, 
    c(sep='', 
      expand.grid(input_directory, available_subjects[starting_index:ending_index], 
      "/eeg")
      )
    )
  subjects_dataframes <- lapply(subjects_folders, convert_to_tibble)
  return(subjects_dataframes)
}

convert_to_tibble <- function(subfolder){
  set_file = list.files(subfolder)
  # Read the file using the MATLAB package
  temp_dat <- R.matlab::readMat(paste(subfolder,set_file, sep="/"))
  # Extract the channel info
  channels_info = temp_dat$chanlocs
  channels_name = channels_info[seq(from=1, to=length(channels_info), by=12)]
  # Create the tibble dataframe
  data_tb = as_tibble(t(temp_dat$data[1:19,]))
  names(data_tb) <- unlist(channels_name)
  
  return(data_tb)
}

read_entropies_tibbles <- function (subjects, cluster_type, subfolder="subjects_entropies"){
  # Read the subjects' classification file 
  subjects_info <- read.table(file = 'ds004504-download/participants.tsv', sep = '\t', header = TRUE)
  if (typeof(subjects) == "character"){
    subjects <- which(subjects_info$Group == subjects)
  }
  tibbles <- vector(mode = "list", length = length(subjects))
  # Iterate the entropies files and collect the dataframe in a list of dfs
  for (i in seq_along(subjects)){
    tibble <- as_tibble(read.table(
      (sprintf("%s/subject_%s_%s.csv", subfolder,subjects[[i]], cluster_type)), sep=',')
    )
    tibbles[[i]] <- tibble[-1,]
  }
  
  return(tibbles)
}

save_to_file <- function(df, filename){
  R.matlab::writeMat(filename, eeg_channels = df)
}
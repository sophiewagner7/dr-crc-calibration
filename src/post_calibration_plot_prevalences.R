# Load necessary libraries
library(ggplot2)
library(tidyverse)
library(dplyr)
library(tidyr)
library(ggridges)

timestamp <- "20240923_1243"
model_version <-"US"

# Read the population data and transpose
file_path <- paste0("out/", model_version, "/interp/logs/", timestamp, "_pop.csv")
pop <- read.csv(file_path)
pop_t <- t(pop)[-1, ]  # Transpose and drop the first row
pop_t <- as.data.frame(pop_t)

# Set column names for health states
colnames(pop_t) <- c("Healthy", "LR Polyp", "HR Polyp", 
    "uLoc", "uReg", "uDis", "dLoc", "dReg", "dDis", "CSD",
    "healthy_ACM", "cancer_ACM", "polyp_ACM", "uCRC_ACM")

# Create a Year column and calculate means for every group of 12 rows
pop_yr <- pop_t |>
  mutate(Year = rep(1:(n()/12), each = 12)) |>  # Rename group to Year
  group_by(Year) |>  
  summarize(across(everything(), ~ mean(.x, na.rm = TRUE)))  # Compute group means

# Calculate the total ACM
pop_yr$ACM <- pop_yr$healthy_ACM + pop_yr$cancer_ACM + pop_yr$polyp_ACM + pop_yr$uCRC_ACM

# Convert wide format to long format for plotting
pop_yr_long <- pop_yr |>
  pivot_longer(
    cols = Healthy:ACM,
    names_to = "Health_State",
    values_to = "Value"
  )

# Normalize values for percentages
pop_yr_long$perc <- pop_yr_long$Value / 100000

# Create the final plot of all health states
plot <- ggplot(pop_yr_long, aes(x = Year, y = perc, color = Health_State)) +
  geom_line(linewidth = 1) +   # Plot lines for each health state
  theme_bw() +
  labs(title = "Overlaid Health States with Areas and Lines", x = "Year", y = "Prevalence")

# Save the plot
ggsave(paste0("out/", model_version, "/interp/plots/", timestamp, "_health_states.png"), plot = plot, width = 12, height = 8)

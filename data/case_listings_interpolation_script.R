# Colorectal Cancer Age and Stage Specific Hazard Function Using RSTMP2 Package
# Adapted from Matthew Prest, mp4090@cumc.columbia.edu
# Author: Sophie Wagner
# Contact: sw3767@cumc.columbia.edu
rm(list = ls()) # Clearing Environment and Plots
# dev.off() # Throws non-critical error if no plots exist
#Importing Packages
library(ggplot2) # For efficient plotting
library(dplyr)  # For dataframe manipulation
library(rstpm2) # For hazard modelling
library(survival)
library(forcats)
library(tidyr)
gc()
cat("\014") # Clearing Console

# Using Case Listing Data
df <- read.csv("C:\\repos\\dr-crc-calibration\\data\\s8_case_listings_1996_1999.csv")

df <- df |> 
  filter(Survival.months != "Unknown") |> 
  mutate(
    SEER.cause.specific.death.classification = factor(
      SEER.cause.specific.death.classification,
      levels = c("Alive or dead of other cause", "Dead (attributable to this cancer dx)")
    ),
    SEER.other.cause.of.death.classification = factor(
      SEER.other.cause.of.death.classification,
      levels = c("Alive or dead due to cancer", "Dead (attributable to causes other than this cancer dx)")
    ),
    Vital.status.recode..study.cutoff.used. = factor(
      Vital.status.recode..study.cutoff.used.,
      levels = c("Alive", "Dead")
    ),
    SEER.historic.stage.A..1973.2015. = factor(
      SEER.historic.stage.A..1973.2015.,
      levels = c("Localized", "Regional", "Distant")
    ),
    Survival.months = as.numeric(Survival.months),
    Event = case_when(
      Vital.status.recode..study.cutoff.used. == "Alive" ~ 0,
      SEER.cause.specific.death.classification == "Dead (attributable to this cancer dx)" ~ 1,
      SEER.other.cause.of.death.classification == "Dead (attributable to causes other than this cancer dx)" ~ 2,
      TRUE ~ NA_real_
    ),
    Cancer_death = ifelse(SEER.cause.specific.death.classification=="Alive or dead of other cause", 0, 1),
    Other_death = ifelse(SEER.other.cause.of.death.classification=="Alive or dead due to cancer", 0, 1),
    All_death = ifelse(Vital.status.recode..study.cutoff.used.=="Alive", 0, 1),
    Age = as.numeric(substr(Age.recode.with.single.ages.and.85.,1,2))
  )


df2 <- df |> 
  arrange(Age) |> 
  mutate(
  AGE = factor(cut(Age, c(0, 29, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84)), 
                     levels = c("(0,29]", "(29,39]", "(39,44]", "(44,49]", 
                                "(49,54]", "(54,59]", "(59,64]", "(64,69]", 
                                "(69,74]", "(74,79]", "(79,84]"), ordered = T),
  EVENT = ifelse(Survival.months > 120, 0, Event), 
  MONTHS = ifelse(Survival.months > 120, 120, Survival.months),
  YEARS = 1+MONTHS %/% 6
)
test<-df2 |> select(AGE, SEER.historic.stage.A..1973.2015., EVENT, YEARS) |> 
  rename(STAGE=SEER.historic.stage.A..1973.2015.) |> 
  drop_na()

# No data for distant age 0-29, will copy distant 29-29
duplicated_data <- test |> 
  filter(EVENT==2 & AGE == "(29,39]" & STAGE %in% c("Regional", "Distant")) |> 
  mutate(AGE = "(0,29]") 
test <- test |> 
  bind_rows(duplicated_data)

fit_OD <- stpm2(Surv(YEARS, EVENT==2)~AGE+STAGE, data = test, df=2)  # Creating model
fit_CD <- stpm2(Surv(YEARS, EVENT==1)~AGE+STAGE, data = test, df=2)

ages <- data.frame(expand.grid(unique(test$AGE), unique(test$STAGE), unique(test$YEARS)))
names(ages) <- c('AGE', 'STAGE', 'YEARS')
# Calculate hazard for each age, stage, year combination
# Function returns ages df plus new columns added
ages <- predict(fit_OD, newdata=ages, type="hazard", grid=F, full=T, se.fit=TRUE)
ages <- predict(fit_CD, newdata=ages, type="hazard", grid=F, full=T, se.fit=TRUE)

# Cleaning & Exporting
rm(test, fit_CD, fit_OD)
names(ages) <- c('AGE', 'STAGE', 'YEARS', 'OD_HAZARD', 'OD_LOWER', 'OD_UPPER', 'CD_HAZARD', 'CD_LOWER', 'CD_UPPER')
ages_refined <- ages |> 
  mutate(YEARS = (YEARS-1) %/% 2) |> 
  arrange(AGE, STAGE, YEARS) |> 
  group_by(AGE, STAGE, YEARS) |> 
  summarise(OD_HAZARD = mean(OD_HAZARD), OD_LOWER = mean(OD_LOWER), OD_UPPER = mean(OD_UPPER), 
            CD_HAZARD = mean(CD_HAZARD), CD_LOWER = mean(CD_LOWER), CD_UPPER = mean(CD_UPPER))

# Exporting
write.csv(ages_refined, "C:\\Users\\repos\\dr-crc-calibration\\data\\s8_hazards_1996_1999.csv", row.names = F)

# Transform to probabilities
# Define the time interval
delta_t <- 1  # 1 year

# Calculate transition probabilities using the formula P(t) = 1 - exp(-h(t) * delta_t)
survival_probs <- ages_refined |>
  mutate(
    OD_PROB_1Y = 1 - exp(-OD_HAZARD * delta_t),
    CD_PROB_1Y = 1 - exp(-CD_HAZARD * delta_t),
    OD_PROB_1M = 1 - (1-OD_PROB_1Y)^(1/12),
    CD_PROB_1M = 1 - (1-CD_PROB_1Y)^(1/12)
  ) |> 
  select(AGE, STAGE, YEARS, OD_PROB_1Y, CD_PROB_1Y, OD_PROB_1M, CD_PROB_1M)

# Exporting
write.csv(survival_probs, "C:\\repos\\dr-crc-calibration\\data\\s8_survival_1996_1999.csv", row.names = F)
         


# Make it into matrix-y format for TreeAge
library(dplyr)
library(tidyverse)
df <- read.csv("C:\\repos\\dr-crc-calibration\\data\\s8_survival_1996_1999.csv")

# Format AGE levels
df <- df |> 
  mutate(AGE = as.factor(AGE)) |> 
  mutate(AGE = factor(AGE,
                      levels = levels(AGE),
                      labels = gsub(",", "_", gsub("[\\(\\)\\[\\]]", "", levels(AGE))))) |> 
  mutate(AGE = substr(AGE, 2,nchar(as.character(AGE)))) |> 
  mutate(AGE= substr(AGE, 1, nchar(as.character(AGE))-1)) |> 
  mutate(AGE_START = as.numeric(str_extract(AGE, "^[0-9]+")),  # Extract the first number
         AGE_END = as.numeric(str_extract(AGE, "[0-9]+$"))) |> 
  mutate(AGE_START = ifelse(AGE_START==0, 20, AGE_START))


dfloc <- df |> 
  filter(STAGE=="Localized") |> 
  select(AGE, YEARS, CD_PROB_1M) |> 
  pivot_wider(names_from = AGE, names_prefix = "AGE_", values_from=CD_PROB_1M) 

dfreg <- df |> 
  filter(STAGE=="Regional") |> 
  select(AGE, YEARS, CD_PROB_1M) |> 
  pivot_wider(names_from = AGE, names_prefix = "AGE_", values_from=CD_PROB_1M)

dfdis <- df |> 
  filter(STAGE=="Distant") |> 
  select(AGE, YEARS, CD_PROB_1M) |> 
  pivot_wider(names_from = AGE, names_prefix = "AGE_", values_from=CD_PROB_1M)


write.csv(dfloc, "C:\\repos\\dr-crc-calibration\\data\\s8_probs_loc_1996_1999.csv", row.names = F)
write.csv(dfreg, "C:\\repos\\dr-crc-calibration\\data\\s8_probs_reg_1996_1999.csv", row.names = F)
write.csv(dfdis, "C:\\repos\\dr-crc-calibration\\data\\s8_probs_dis_1996_1999.csv", row.names = F)

# SEER 18 Case Listings to Survival
# Author: Matthew Prest
# Contact: mp4090@cumc.columbia.edu
rm(list = ls()) # Clearing Environment and Plots
dev.off() # Throws non-critical error if no plots exist
gc() # Freeing unused memory
#Importing Packages
library(ggplot2) # For efficient plotting
library(dplyr)  # For dataframe manipulation
library(data.table) # For quick read/write
library(haven) # For reading SAS files
library(rstpm2) # For hazard function creation
library(survival)
library(modelbased)
library(oce)
memory.limit(9999999999) # Increasing memory capacity
# `%!in%` <- Negate(`%in%`)
options(scipen = 100)
cat("\014") # Clearing Console


df1 <- read.csv("W:\\Matt P\\Projects\\202201 CISNET\\Gastric\\Case Listings 1.csv")
df2 <- read.csv("W:\\Matt P\\Projects\\202201 CISNET\\Gastric\\Case Listings 2.csv")
df3 <- read.csv("W:\\Matt P\\Projects\\202201 CISNET\\Gastric\\Case Listings 3.csv")
df4 <- read.csv("W:\\Matt P\\Projects\\202201 CISNET\\Gastric\\Case Listings 4.csv")
df5 <- read.csv("W:\\Matt P\\Projects\\202201 CISNET\\Gastric\\Case Listings 5.csv")

# Stripping excess columns, removing unknown races and 85+
df1 <- df1 %>% select(YEAR, AGE, SEX, RACE, HIST, AJCC, VITAL, SURV)
df2 <- df2 %>% select(YEAR, AGE, SEX, RACE, HIST, AJCC, VITAL, SURV)
df3 <- df3 %>% select(YEAR, AGE, SEX, RACE, HIST, AJCC, VITAL, SURV)
df4 <- df4 %>% select(YEAR, AGE, SEX, RACE, HIST, AJCC, VITAL, SURV)
df5 <- df5 %>% select(YEAR, AGE, SEX, RACE, HIST, AJCC, VITAL, SURV)

# Only need SURV in 12 month increments
df1$SURV <- ifelse(df1$SURV <= 12, 12, ifelse(df1$SURV <= 24, 24, ifelse(df1$SURV <= 36, 36, ifelse(df1$SURV <= 48, 48, ifelse(df1$SURV <= 60, 60,
            ifelse(df1$SURV <= 72, 72, ifelse(df1$SURV <= 84, 84, ifelse(df1$SURV <= 96, 96, ifelse(df1$SURV <= 108, 108, 120)))))))))
df2$SURV <- ifelse(df2$SURV <= 12, 12, ifelse(df2$SURV <= 24, 24, ifelse(df2$SURV <= 36, 36, ifelse(df2$SURV <= 48, 48, ifelse(df2$SURV <= 60, 60,
            ifelse(df2$SURV <= 72, 72, ifelse(df2$SURV <= 84, 84, ifelse(df2$SURV <= 96, 96, ifelse(df2$SURV <= 108, 108, 120)))))))))
df3$SURV <- ifelse(df3$SURV <= 12, 12, ifelse(df3$SURV <= 24, 24, ifelse(df3$SURV <= 36, 36, ifelse(df3$SURV <= 48, 48, ifelse(df3$SURV <= 60, 60,
            ifelse(df3$SURV <= 72, 72, ifelse(df3$SURV <= 84, 84, ifelse(df3$SURV <= 96, 96, ifelse(df3$SURV <= 108, 108, 120)))))))))
df4$SURV <- ifelse(df4$SURV <= 12, 12, ifelse(df4$SURV <= 24, 24, ifelse(df4$SURV <= 36, 36, ifelse(df4$SURV <= 48, 48, ifelse(df4$SURV <= 60, 60,
            ifelse(df4$SURV <= 72, 72, ifelse(df4$SURV <= 84, 84, ifelse(df4$SURV <= 96, 96, ifelse(df4$SURV <= 108, 108, 120)))))))))
df5$SURV <- ifelse(df5$SURV <= 12, 12, ifelse(df5$SURV <= 24, 24, ifelse(df5$SURV <= 36, 36, ifelse(df5$SURV <= 48, 48, ifelse(df5$SURV <= 60, 60,
            ifelse(df5$SURV <= 72, 72, ifelse(df5$SURV <= 84, 84, ifelse(df5$SURV <= 96, 96, ifelse(df5$SURV <= 108, 108, 120)))))))))

df1 <- df1 %>% group_by(YEAR, AGE, SEX, RACE, HIST, AJCC, VITAL, SURV) %>% summarise(Count = n())
df2 <- df2 %>% group_by(YEAR, AGE, SEX, RACE, HIST, AJCC, VITAL, SURV) %>% summarise(Count = n())
df3 <- df3 %>% group_by(YEAR, AGE, SEX, RACE, HIST, AJCC, VITAL, SURV) %>% summarise(Count = n())
df4 <- df4 %>% group_by(YEAR, AGE, SEX, RACE, HIST, AJCC, VITAL, SURV) %>% summarise(Count = n())
df5 <- df5 %>% group_by(YEAR, AGE, SEX, RACE, HIST, AJCC, VITAL, SURV) %>% summarise(Count = n())

a1 <- df1 %>% filter(VITAL == "Alive") %>% mutate(ALIVE = Count)
a2 <- df2 %>% filter(VITAL == "Alive") %>% mutate(ALIVE = Count)
a3 <- df3 %>% filter(VITAL == "Alive") %>% mutate(ALIVE = Count)
a4 <- df4 %>% filter(VITAL == "Alive") %>% mutate(ALIVE = Count)
a5 <- df5 %>% filter(VITAL == "Alive") %>% mutate(ALIVE = Count)

d1 <- df1 %>% filter(VITAL != "Alive") %>% mutate(DEAD = Count)
d2 <- df2 %>% filter(VITAL != "Alive") %>% mutate(DEAD = Count)
d3 <- df3 %>% filter(VITAL != "Alive") %>% mutate(DEAD = Count)
d4 <- df4 %>% filter(VITAL != "Alive") %>% mutate(DEAD = Count)
d5 <- df5 %>% filter(VITAL != "Alive") %>% mutate(DEAD = Count)

a1$Count <- NULL
a2$Count <- NULL
a3$Count <- NULL
a4$Count <- NULL
a5$Count <- NULL
d1$Count <- NULL
d2$Count <- NULL
d3$Count <- NULL
d4$Count <- NULL
d5$Count <- NULL

a1$VITAL <- NULL
a2$VITAL <- NULL
a3$VITAL <- NULL
a4$VITAL <- NULL
a5$VITAL <- NULL
d1$VITAL <- NULL
d2$VITAL <- NULL
d3$VITAL <- NULL
d4$VITAL <- NULL
d5$VITAL <- NULL

df1 <- merge(a1, d1, by=c('YEAR','AGE','SEX','RACE','HIST','AJCC','SURV'), all = T)
df2 <- merge(a2, d2, by=c('YEAR','AGE','SEX','RACE','HIST','AJCC','SURV'), all = T)
df3 <- merge(a3, d3, by=c('YEAR','AGE','SEX','RACE','HIST','AJCC','SURV'), all = T)
df4 <- merge(a4, d4, by=c('YEAR','AGE','SEX','RACE','HIST','AJCC','SURV'), all = T)
df5 <- merge(a5, d5, by=c('YEAR','AGE','SEX','RACE','HIST','AJCC','SURV'), all = T)
rm(a1, a2, a3, a4, a5, d1, d2, d3, d4, d5)

df1[is.na(df1)] <- 0
df2[is.na(df2)] <- 0
df3[is.na(df3)] <- 0
df4[is.na(df4)] <- 0
df5[is.na(df5)] <- 0

df1$BIRTHYEAR <- df1$YEAR - df1$AGE
df2$BIRTHYEAR <- df2$YEAR - df2$AGE
df3$BIRTHYEAR <- df3$YEAR - df3$AGE
df4$BIRTHYEAR <- df4$YEAR - df4$AGE
df5$BIRTHYEAR <- df5$YEAR - df5$AGE

df1$FOLLOWUP <- df1$SURV
df2$FOLLOWUP <- df2$SURV
df3$FOLLOWUP <- df3$SURV
df4$FOLLOWUP <- df4$SURV
df5$FOLLOWUP <- df5$SURV

df1$SURV <- NULL
df2$SURV <- NULL
df3$SURV <- NULL
df4$SURV <- NULL
df5$SURV <- NULL

df1 <- df1 %>% select(YEAR, AGE, BIRTHYEAR, SEX, RACE, HIST, AJCC, FOLLOWUP, ALIVE, DEAD)
df2 <- df2 %>% select(YEAR, AGE, BIRTHYEAR, SEX, RACE, HIST, AJCC, FOLLOWUP, ALIVE, DEAD)
df3 <- df3 %>% select(YEAR, AGE, BIRTHYEAR, SEX, RACE, HIST, AJCC, FOLLOWUP, ALIVE, DEAD)
df4 <- df4 %>% select(YEAR, AGE, BIRTHYEAR, SEX, RACE, HIST, AJCC, FOLLOWUP, ALIVE, DEAD)
df5 <- df5 %>% select(YEAR, AGE, BIRTHYEAR, SEX, RACE, HIST, AJCC, FOLLOWUP, ALIVE, DEAD)

# Importing Survival Template
template <- read.csv("W:\\Matt P\\Projects\\202201 CISNET\\Gastric\\Survival Template.csv")
template$ALIVE <- NULL
template$DEAD <- NULL


# Cumulative grouping
df1 <- merge(template, df1, by = c('YEAR','AGE','BIRTHYEAR','SEX','RACE','HIST','AJCC','FOLLOWUP'), all = T)
df1[is.na(df1)] <- 0
# Splitting into follow up groups
df1_12 <- df1 %>% filter(FOLLOWUP == 12)
df1_24 <- df1 %>% filter(FOLLOWUP == 24)
df1_36 <- df1 %>% filter(FOLLOWUP == 36)
df1_48 <- df1 %>% filter(FOLLOWUP == 48)
df1_60 <- df1 %>% filter(FOLLOWUP == 60)
df1_72 <- df1 %>% filter(FOLLOWUP == 72)
df1_84 <- df1 %>% filter(FOLLOWUP == 84)
df1_96 <- df1 %>% filter(FOLLOWUP == 96)
df1_108 <- df1 %>% filter(FOLLOWUP == 108)
df1_120 <- df1 %>% filter(FOLLOWUP == 120)

df1_12$ALIVE <- df1_12$ALIVE + df1_24$ALIVE + df1_36$ALIVE + df1_48$ALIVE + df1_60$ALIVE + df1_72$ALIVE + df1_84$ALIVE + df1_96$ALIVE + df1_108$ALIVE + df1_120$ALIVE
df1_24$ALIVE <- df1_24$ALIVE + df1_36$ALIVE + df1_48$ALIVE + df1_60$ALIVE + df1_72$ALIVE + df1_84$ALIVE + df1_96$ALIVE + df1_108$ALIVE + df1_120$ALIVE
df1_36$ALIVE <- df1_36$ALIVE + df1_48$ALIVE + df1_60$ALIVE + df1_72$ALIVE + df1_84$ALIVE + df1_96$ALIVE + df1_108$ALIVE + df1_120$ALIVE
df1_48$ALIVE <- df1_48$ALIVE + df1_60$ALIVE + df1_72$ALIVE + df1_84$ALIVE + df1_96$ALIVE + df1_108$ALIVE + df1_120$ALIVE
df1_60$ALIVE <- df1_60$ALIVE + df1_72$ALIVE + df1_84$ALIVE + df1_96$ALIVE + df1_108$ALIVE + df1_120$ALIVE
df1_72$ALIVE <- df1_72$ALIVE + df1_84$ALIVE + df1_96$ALIVE + df1_108$ALIVE + df1_120$ALIVE
df1_84$ALIVE <- df1_84$ALIVE + df1_96$ALIVE + df1_108$ALIVE + df1_120$ALIVE
df1_96$ALIVE <- df1_96$ALIVE + df1_108$ALIVE + df1_120$ALIVE
df1_108$ALIVE <- df1_108$ALIVE + df1_120$ALIVE

df1_120$DEAD <- df1_12$DEAD + df1_24$DEAD + df1_36$DEAD + df1_48$DEAD + df1_60$DEAD + df1_72$DEAD + df1_84$DEAD + df1_96$DEAD + df1_108$DEAD + df1_120$DEAD
df1_108$DEAD <- df1_12$DEAD + df1_24$DEAD + df1_36$DEAD + df1_48$DEAD + df1_60$DEAD + df1_72$DEAD + df1_84$DEAD + df1_96$DEAD + df1_108$DEAD
df1_96$DEAD <- df1_12$DEAD + df1_24$DEAD + df1_36$DEAD + df1_48$DEAD + df1_60$DEAD + df1_72$DEAD + df1_84$DEAD + df1_96$DEAD
df1_84$DEAD <- df1_12$DEAD + df1_24$DEAD + df1_36$DEAD + df1_48$DEAD + df1_60$DEAD + df1_72$DEAD + df1_84$DEAD
df1_72$DEAD <- df1_12$DEAD + df1_24$DEAD + df1_36$DEAD + df1_48$DEAD + df1_60$DEAD + df1_72$DEAD 
df1_60$DEAD <- df1_12$DEAD + df1_24$DEAD + df1_36$DEAD + df1_48$DEAD + df1_60$DEAD
df1_48$DEAD <- df1_12$DEAD + df1_24$DEAD + df1_36$DEAD + df1_48$DEAD
df1_36$DEAD <- df1_12$DEAD + df1_24$DEAD + df1_36$DEAD 
df1_24$DEAD <- df1_12$DEAD + df1_24$DEAD

df1 <- rbind(df1_12, df1_24)
df1 <- rbind(df1, df1_36)
df1 <- rbind(df1, df1_48)
df1 <- rbind(df1, df1_60)
df1 <- rbind(df1, df1_72)
df1 <- rbind(df1, df1_84)
df1 <- rbind(df1, df1_96)
df1 <- rbind(df1, df1_108)
df1 <- rbind(df1, df1_120)
rm(df1_12, df1_24, df1_36, df1_48, df1_60, df1_72, df1_84, df1_96, df1_108, df1_120)

df2 <- merge(template, df2, by = c('YEAR','AGE','BIRTHYEAR','SEX','RACE','HIST','AJCC','FOLLOWUP'), all = T)
df2[is.na(df2)] <- 0
# Splitting into follow up groups
df2_12 <- df2 %>% filter(FOLLOWUP == 12)
df2_24 <- df2 %>% filter(FOLLOWUP == 24)
df2_36 <- df2 %>% filter(FOLLOWUP == 36)
df2_48 <- df2 %>% filter(FOLLOWUP == 48)
df2_60 <- df2 %>% filter(FOLLOWUP == 60)
df2_72 <- df2 %>% filter(FOLLOWUP == 72)
df2_84 <- df2 %>% filter(FOLLOWUP == 84)
df2_96 <- df2 %>% filter(FOLLOWUP == 96)
df2_108 <- df2 %>% filter(FOLLOWUP == 108)
df2_120 <- df2 %>% filter(FOLLOWUP == 120)

df2_12$ALIVE <- df2_12$ALIVE + df2_24$ALIVE + df2_36$ALIVE + df2_48$ALIVE + df2_60$ALIVE + df2_72$ALIVE + df2_84$ALIVE + df2_96$ALIVE + df2_108$ALIVE + df2_120$ALIVE
df2_24$ALIVE <- df2_24$ALIVE + df2_36$ALIVE + df2_48$ALIVE + df2_60$ALIVE + df2_72$ALIVE + df2_84$ALIVE + df2_96$ALIVE + df2_108$ALIVE + df2_120$ALIVE
df2_36$ALIVE <- df2_36$ALIVE + df2_48$ALIVE + df2_60$ALIVE + df2_72$ALIVE + df2_84$ALIVE + df2_96$ALIVE + df2_108$ALIVE + df2_120$ALIVE
df2_48$ALIVE <- df2_48$ALIVE + df2_60$ALIVE + df2_72$ALIVE + df2_84$ALIVE + df2_96$ALIVE + df2_108$ALIVE + df2_120$ALIVE
df2_60$ALIVE <- df2_60$ALIVE + df2_72$ALIVE + df2_84$ALIVE + df2_96$ALIVE + df2_108$ALIVE + df2_120$ALIVE
df2_72$ALIVE <- df2_72$ALIVE + df2_84$ALIVE + df2_96$ALIVE + df2_108$ALIVE + df2_120$ALIVE
df2_84$ALIVE <- df2_84$ALIVE + df2_96$ALIVE + df2_108$ALIVE + df2_120$ALIVE
df2_96$ALIVE <- df2_96$ALIVE + df2_108$ALIVE + df2_120$ALIVE
df2_108$ALIVE <- df2_108$ALIVE + df2_120$ALIVE

df2_120$DEAD <- df2_12$DEAD + df2_24$DEAD + df2_36$DEAD + df2_48$DEAD + df2_60$DEAD + df2_72$DEAD + df2_84$DEAD + df2_96$DEAD + df2_108$DEAD + df2_120$DEAD
df2_108$DEAD <- df2_12$DEAD + df2_24$DEAD + df2_36$DEAD + df2_48$DEAD + df2_60$DEAD + df2_72$DEAD + df2_84$DEAD + df2_96$DEAD + df2_108$DEAD
df2_96$DEAD <- df2_12$DEAD + df2_24$DEAD + df2_36$DEAD + df2_48$DEAD + df2_60$DEAD + df2_72$DEAD + df2_84$DEAD + df2_96$DEAD
df2_84$DEAD <- df2_12$DEAD + df2_24$DEAD + df2_36$DEAD + df2_48$DEAD + df2_60$DEAD + df2_72$DEAD + df2_84$DEAD
df2_72$DEAD <- df2_12$DEAD + df2_24$DEAD + df2_36$DEAD + df2_48$DEAD + df2_60$DEAD + df2_72$DEAD 
df2_60$DEAD <- df2_12$DEAD + df2_24$DEAD + df2_36$DEAD + df2_48$DEAD + df2_60$DEAD
df2_48$DEAD <- df2_12$DEAD + df2_24$DEAD + df2_36$DEAD + df2_48$DEAD
df2_36$DEAD <- df2_12$DEAD + df2_24$DEAD + df2_36$DEAD 
df2_24$DEAD <- df2_12$DEAD + df2_24$DEAD

df2 <- rbind(df2_12, df2_24)
df2 <- rbind(df2, df2_36)
df2 <- rbind(df2, df2_48)
df2 <- rbind(df2, df2_60)
df2 <- rbind(df2, df2_72)
df2 <- rbind(df2, df2_84)
df2 <- rbind(df2, df2_96)
df2 <- rbind(df2, df2_108)
df2 <- rbind(df2, df2_120)
rm(df2_12, df2_24, df2_36, df2_48, df2_60, df2_72, df2_84, df2_96, df2_108, df2_120)

df3 <- merge(template, df3, by = c('YEAR','AGE','BIRTHYEAR','SEX','RACE','HIST','AJCC','FOLLOWUP'), all = T)
df3[is.na(df3)] <- 0
# Splitting into follow up groups
df3_12 <- df3 %>% filter(FOLLOWUP == 12)
df3_24 <- df3 %>% filter(FOLLOWUP == 24)
df3_36 <- df3 %>% filter(FOLLOWUP == 36)
df3_48 <- df3 %>% filter(FOLLOWUP == 48)
df3_60 <- df3 %>% filter(FOLLOWUP == 60)
df3_72 <- df3 %>% filter(FOLLOWUP == 72)
df3_84 <- df3 %>% filter(FOLLOWUP == 84)
df3_96 <- df3 %>% filter(FOLLOWUP == 96)
df3_108 <- df3 %>% filter(FOLLOWUP == 108)
df3_120 <- df3 %>% filter(FOLLOWUP == 120)

df3_12$ALIVE <- df3_12$ALIVE + df3_24$ALIVE + df3_36$ALIVE + df3_48$ALIVE + df3_60$ALIVE + df3_72$ALIVE + df3_84$ALIVE + df3_96$ALIVE + df3_108$ALIVE + df3_120$ALIVE
df3_24$ALIVE <- df3_24$ALIVE + df3_36$ALIVE + df3_48$ALIVE + df3_60$ALIVE + df3_72$ALIVE + df3_84$ALIVE + df3_96$ALIVE + df3_108$ALIVE + df3_120$ALIVE
df3_36$ALIVE <- df3_36$ALIVE + df3_48$ALIVE + df3_60$ALIVE + df3_72$ALIVE + df3_84$ALIVE + df3_96$ALIVE + df3_108$ALIVE + df3_120$ALIVE
df3_48$ALIVE <- df3_48$ALIVE + df3_60$ALIVE + df3_72$ALIVE + df3_84$ALIVE + df3_96$ALIVE + df3_108$ALIVE + df3_120$ALIVE
df3_60$ALIVE <- df3_60$ALIVE + df3_72$ALIVE + df3_84$ALIVE + df3_96$ALIVE + df3_108$ALIVE + df3_120$ALIVE
df3_72$ALIVE <- df3_72$ALIVE + df3_84$ALIVE + df3_96$ALIVE + df3_108$ALIVE + df3_120$ALIVE
df3_84$ALIVE <- df3_84$ALIVE + df3_96$ALIVE + df3_108$ALIVE + df3_120$ALIVE
df3_96$ALIVE <- df3_96$ALIVE + df3_108$ALIVE + df3_120$ALIVE
df3_108$ALIVE <- df3_108$ALIVE + df3_120$ALIVE

df3_120$DEAD <- df3_12$DEAD + df3_24$DEAD + df3_36$DEAD + df3_48$DEAD + df3_60$DEAD + df3_72$DEAD + df3_84$DEAD + df3_96$DEAD + df3_108$DEAD + df3_120$DEAD
df3_108$DEAD <- df3_12$DEAD + df3_24$DEAD + df3_36$DEAD + df3_48$DEAD + df3_60$DEAD + df3_72$DEAD + df3_84$DEAD + df3_96$DEAD + df3_108$DEAD
df3_96$DEAD <- df3_12$DEAD + df3_24$DEAD + df3_36$DEAD + df3_48$DEAD + df3_60$DEAD + df3_72$DEAD + df3_84$DEAD + df3_96$DEAD
df3_84$DEAD <- df3_12$DEAD + df3_24$DEAD + df3_36$DEAD + df3_48$DEAD + df3_60$DEAD + df3_72$DEAD + df3_84$DEAD
df3_72$DEAD <- df3_12$DEAD + df3_24$DEAD + df3_36$DEAD + df3_48$DEAD + df3_60$DEAD + df3_72$DEAD 
df3_60$DEAD <- df3_12$DEAD + df3_24$DEAD + df3_36$DEAD + df3_48$DEAD + df3_60$DEAD
df3_48$DEAD <- df3_12$DEAD + df3_24$DEAD + df3_36$DEAD + df3_48$DEAD
df3_36$DEAD <- df3_12$DEAD + df3_24$DEAD + df3_36$DEAD 
df3_24$DEAD <- df3_12$DEAD + df3_24$DEAD

df3 <- rbind(df3_12, df3_24)
df3 <- rbind(df3, df3_36)
df3 <- rbind(df3, df3_48)
df3 <- rbind(df3, df3_60)
df3 <- rbind(df3, df3_72)
df3 <- rbind(df3, df3_84)
df3 <- rbind(df3, df3_96)
df3 <- rbind(df3, df3_108)
df3 <- rbind(df3, df3_120)
rm(df3_12, df3_24, df3_36, df3_48, df3_60, df3_72, df3_84, df3_96, df3_108, df3_120)

df4 <- merge(template, df4, by = c('YEAR','AGE','BIRTHYEAR','SEX','RACE','HIST','AJCC','FOLLOWUP'), all = T)
df4[is.na(df4)] <- 0
# Splitting into follow up groups
df4_12 <- df4 %>% filter(FOLLOWUP == 12)
df4_24 <- df4 %>% filter(FOLLOWUP == 24)
df4_36 <- df4 %>% filter(FOLLOWUP == 36)
df4_48 <- df4 %>% filter(FOLLOWUP == 48)
df4_60 <- df4 %>% filter(FOLLOWUP == 60)
df4_72 <- df4 %>% filter(FOLLOWUP == 72)
df4_84 <- df4 %>% filter(FOLLOWUP == 84)
df4_96 <- df4 %>% filter(FOLLOWUP == 96)
df4_108 <- df4 %>% filter(FOLLOWUP == 108)
df4_120 <- df4 %>% filter(FOLLOWUP == 120)

df4_12$ALIVE <- df4_12$ALIVE + df4_24$ALIVE + df4_36$ALIVE + df4_48$ALIVE + df4_60$ALIVE + df4_72$ALIVE + df4_84$ALIVE + df4_96$ALIVE + df4_108$ALIVE + df4_120$ALIVE
df4_24$ALIVE <- df4_24$ALIVE + df4_36$ALIVE + df4_48$ALIVE + df4_60$ALIVE + df4_72$ALIVE + df4_84$ALIVE + df4_96$ALIVE + df4_108$ALIVE + df4_120$ALIVE
df4_36$ALIVE <- df4_36$ALIVE + df4_48$ALIVE + df4_60$ALIVE + df4_72$ALIVE + df4_84$ALIVE + df4_96$ALIVE + df4_108$ALIVE + df4_120$ALIVE
df4_48$ALIVE <- df4_48$ALIVE + df4_60$ALIVE + df4_72$ALIVE + df4_84$ALIVE + df4_96$ALIVE + df4_108$ALIVE + df4_120$ALIVE
df4_60$ALIVE <- df4_60$ALIVE + df4_72$ALIVE + df4_84$ALIVE + df4_96$ALIVE + df4_108$ALIVE + df4_120$ALIVE
df4_72$ALIVE <- df4_72$ALIVE + df4_84$ALIVE + df4_96$ALIVE + df4_108$ALIVE + df4_120$ALIVE
df4_84$ALIVE <- df4_84$ALIVE + df4_96$ALIVE + df4_108$ALIVE + df4_120$ALIVE
df4_96$ALIVE <- df4_96$ALIVE + df4_108$ALIVE + df4_120$ALIVE
df4_108$ALIVE <- df4_108$ALIVE + df4_120$ALIVE

df4_120$DEAD <- df4_12$DEAD + df4_24$DEAD + df4_36$DEAD + df4_48$DEAD + df4_60$DEAD + df4_72$DEAD + df4_84$DEAD + df4_96$DEAD + df4_108$DEAD + df4_120$DEAD
df4_108$DEAD <- df4_12$DEAD + df4_24$DEAD + df4_36$DEAD + df4_48$DEAD + df4_60$DEAD + df4_72$DEAD + df4_84$DEAD + df4_96$DEAD + df4_108$DEAD
df4_96$DEAD <- df4_12$DEAD + df4_24$DEAD + df4_36$DEAD + df4_48$DEAD + df4_60$DEAD + df4_72$DEAD + df4_84$DEAD + df4_96$DEAD
df4_84$DEAD <- df4_12$DEAD + df4_24$DEAD + df4_36$DEAD + df4_48$DEAD + df4_60$DEAD + df4_72$DEAD + df4_84$DEAD
df4_72$DEAD <- df4_12$DEAD + df4_24$DEAD + df4_36$DEAD + df4_48$DEAD + df4_60$DEAD + df4_72$DEAD 
df4_60$DEAD <- df4_12$DEAD + df4_24$DEAD + df4_36$DEAD + df4_48$DEAD + df4_60$DEAD
df4_48$DEAD <- df4_12$DEAD + df4_24$DEAD + df4_36$DEAD + df4_48$DEAD
df4_36$DEAD <- df4_12$DEAD + df4_24$DEAD + df4_36$DEAD 
df4_24$DEAD <- df4_12$DEAD + df4_24$DEAD

df4 <- rbind(df4_12, df4_24)
df4 <- rbind(df4, df4_36)
df4 <- rbind(df4, df4_48)
df4 <- rbind(df4, df4_60)
df4 <- rbind(df4, df4_72)
df4 <- rbind(df4, df4_84)
df4 <- rbind(df4, df4_96)
df4 <- rbind(df4, df4_108)
df4 <- rbind(df4, df4_120)
rm(df4_12, df4_24, df4_36, df4_48, df4_60, df4_72, df4_84, df4_96, df4_108, df4_120)

df5 <- merge(template, df5, by = c('YEAR','AGE','BIRTHYEAR','SEX','RACE','HIST','AJCC','FOLLOWUP'), all = T)
df5[is.na(df5)] <- 0
# Splitting into follow up groups
df5_12 <- df5 %>% filter(FOLLOWUP == 12)
df5_24 <- df5 %>% filter(FOLLOWUP == 24)
df5_36 <- df5 %>% filter(FOLLOWUP == 36)
df5_48 <- df5 %>% filter(FOLLOWUP == 48)
df5_60 <- df5 %>% filter(FOLLOWUP == 60)
df5_72 <- df5 %>% filter(FOLLOWUP == 72)
df5_84 <- df5 %>% filter(FOLLOWUP == 84)
df5_96 <- df5 %>% filter(FOLLOWUP == 96)
df5_108 <- df5 %>% filter(FOLLOWUP == 108)
df5_120 <- df5 %>% filter(FOLLOWUP == 120)

df5_12$ALIVE <- df5_12$ALIVE + df5_24$ALIVE + df5_36$ALIVE + df5_48$ALIVE + df5_60$ALIVE + df5_72$ALIVE + df5_84$ALIVE + df5_96$ALIVE + df5_108$ALIVE + df5_120$ALIVE
df5_24$ALIVE <- df5_24$ALIVE + df5_36$ALIVE + df5_48$ALIVE + df5_60$ALIVE + df5_72$ALIVE + df5_84$ALIVE + df5_96$ALIVE + df5_108$ALIVE + df5_120$ALIVE
df5_36$ALIVE <- df5_36$ALIVE + df5_48$ALIVE + df5_60$ALIVE + df5_72$ALIVE + df5_84$ALIVE + df5_96$ALIVE + df5_108$ALIVE + df5_120$ALIVE
df5_48$ALIVE <- df5_48$ALIVE + df5_60$ALIVE + df5_72$ALIVE + df5_84$ALIVE + df5_96$ALIVE + df5_108$ALIVE + df5_120$ALIVE
df5_60$ALIVE <- df5_60$ALIVE + df5_72$ALIVE + df5_84$ALIVE + df5_96$ALIVE + df5_108$ALIVE + df5_120$ALIVE
df5_72$ALIVE <- df5_72$ALIVE + df5_84$ALIVE + df5_96$ALIVE + df5_108$ALIVE + df5_120$ALIVE
df5_84$ALIVE <- df5_84$ALIVE + df5_96$ALIVE + df5_108$ALIVE + df5_120$ALIVE
df5_96$ALIVE <- df5_96$ALIVE + df5_108$ALIVE + df5_120$ALIVE
df5_108$ALIVE <- df5_108$ALIVE + df5_120$ALIVE

df5_120$DEAD <- df5_12$DEAD + df5_24$DEAD + df5_36$DEAD + df5_48$DEAD + df5_60$DEAD + df5_72$DEAD + df5_84$DEAD + df5_96$DEAD + df5_108$DEAD + df5_120$DEAD
df5_108$DEAD <- df5_12$DEAD + df5_24$DEAD + df5_36$DEAD + df5_48$DEAD + df5_60$DEAD + df5_72$DEAD + df5_84$DEAD + df5_96$DEAD + df5_108$DEAD
df5_96$DEAD <- df5_12$DEAD + df5_24$DEAD + df5_36$DEAD + df5_48$DEAD + df5_60$DEAD + df5_72$DEAD + df5_84$DEAD + df5_96$DEAD
df5_84$DEAD <- df5_12$DEAD + df5_24$DEAD + df5_36$DEAD + df5_48$DEAD + df5_60$DEAD + df5_72$DEAD + df5_84$DEAD
df5_72$DEAD <- df5_12$DEAD + df5_24$DEAD + df5_36$DEAD + df5_48$DEAD + df5_60$DEAD + df5_72$DEAD 
df5_60$DEAD <- df5_12$DEAD + df5_24$DEAD + df5_36$DEAD + df5_48$DEAD + df5_60$DEAD
df5_48$DEAD <- df5_12$DEAD + df5_24$DEAD + df5_36$DEAD + df5_48$DEAD
df5_36$DEAD <- df5_12$DEAD + df5_24$DEAD + df5_36$DEAD 
df5_24$DEAD <- df5_12$DEAD + df5_24$DEAD

df5 <- rbind(df5_12, df5_24)
df5 <- rbind(df5, df5_36)
df5 <- rbind(df5, df5_48)
df5 <- rbind(df5, df5_60)
df5 <- rbind(df5, df5_72)
df5 <- rbind(df5, df5_84)
df5 <- rbind(df5, df5_96)
df5 <- rbind(df5, df5_108)
df5 <- rbind(df5, df5_120)
rm(df5_12, df5_24, df5_36, df5_48, df5_60, df5_72, df5_84, df5_96, df5_108, df5_120)
rm(template)
gc()



































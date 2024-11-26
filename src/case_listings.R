library(rstpm2)
library(readr)
library(dplyr)
library(forcats)
library(survival)

data <- read.csv("data/s8_case_lisitngs_1975_1985.csv")
data$surv <- as.numeric(data$Survival.months)
data$acm <- as.factor(data$SEER.other.cause.of.death.classification)
data$acm <- fct_recode(
    data$acm,
    "0"="Alive or dead due to cancer",
    "1"="Dead (attributable to causes other than this cancer dx)")
data$csd <- as.factor(data$SEER.cause.specific.death.classification)
data$csd <- fct_recode(
    data$csd,
    "0"="Alive or dead of other cause",
    "1"="Dead (attributable to this cancer dx)"
)
data$age_at_dx <- as.numeric(substr(data$Age.recode.with.single.ages.and.85., 1,2))
data$age_at_dx <- ifelse(data$age_at_dx <= 20, 20, data$age_at_dx)
data$age_5y_groups <- cut(
    data$age_at_dx,
   breaks = seq(20, 85, by = 5),
  labels = paste(seq(20, 80, by = 5), seq(24, 84, by = 5), sep = "-"),
  right = FALSE)
data$age_5y <- cut(
    data$age_at_dx,
       breaks = seq(0, 85, by = 5),  
  right = FALSE)
data$stage<- as.factor(data$SEER.historic.stage.A..1973.2015.)
data <- data |> filter(stage != "Unstaged") |> filter(!is.na(surv))
d2035 <- data |> filter(age_at_dx <= 35)
model <- survfit(Surv(surv, csd)~stage, data=d2035)
summary(model, times=seq(0,120,by=12))
s<-summary(model, times=seq(0,120,by=12))
temp$n.event<- concat(0,s$n.event[1:12,2])
df$prob_dying_interval <- c(NA, diff(1 - df$survival_prob))


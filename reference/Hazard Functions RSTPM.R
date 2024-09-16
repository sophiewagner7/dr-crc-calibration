# Uterine & Gastric Cancer Age, Race, Sex, Stage Specific Hazard Function Using RSTMP2 Package
# Author: Matthew Prest
# Contact: mp4090@cumc.columbia.edu
rm(list = ls()) # Clearing Environment and Plots
# dev.off() # Throws non-critical error if no plots exist
#Importing Packages
library(ggplot2) # For efficient plotting
library(dplyr)  # For dataframe manipulation
library(rstpm2) # For hazard modelling
library(survival)
gc()
cat("\014") # Clearing Console


# Using Case Listing Data
df <- read.csv("W:\\Matt P\\Projects\\202201 CISNET\\Hazard Functions\\Uterine CL.csv")
events <- c(0,1,2)  # Remapping event variable to numeric
names(events) <- c("Alive", "Other Death", "Uterine Cancer Death")
df$EVENT <- events[df$EVENT]
rm(events)

# EIN Multi-cohort
test <- df %>% filter(HIST %in% c('EM', 'Non-EM'), RACE %in% c('NH White', 'NH Black')) %>% select(AGE, RACE, HIST, MONTHS, EVENT)
test <- test %>% mutate(AGE = cut(AGE, c(0, 29, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84))) # Binning ages
test <- test %>% mutate(EVENT = ifelse(MONTHS > 120, 0, EVENT), MONTHS = ifelse(MONTHS > 120, 120, MONTHS))  # max follow up to 10 years
test <- test %>% mutate(YEARS = 1+MONTHS %/% 6) %>% select(-MONTHS)  # Binning into years

fit_OD <- stpm2(Surv(YEARS, EVENT==1)~AGE+RACE+HIST, data = test, df=3)  # Creating model
fit_CD <- stpm2(Surv(YEARS, EVENT==2)~AGE+RACE+HIST, data = test, df=3)

ages <- data.frame(expand.grid(unique(test$AGE), unique(test$RACE), unique(test$HIST), unique(test$YEARS)))
names(ages) <- c('AGE', 'RACE', 'HIST', 'YEARS')
ages <- ages %>% arrange(AGE, RACE, HIST, YEARS)
ages <- predict(fit_OD, newdata=ages, type="hazard", grid=F, full=T, se.fit=TRUE)
ages <- predict(fit_CD, newdata=ages, type="hazard", grid=F, full=T, se.fit=TRUE)

# Cleaning & Exporting
rm(test, fit_CD, fit_OD)
names(ages) <- c('AGE', 'RACE', 'HIST', 'YEARS', 'OD_HAZARD', 'OD_LOWER', 'OD_UPPER', 'CD_HAZARD', 'CD_LOWER', 'CD_UPPER')
ages <- ages %>% mutate(YEARS = (YEARS-1) %/% 2) %>% group_by(AGE, RACE, HIST, YEARS) %>% summarise(OD_HAZARD = mean(OD_HAZARD), OD_LOWER = mean(OD_LOWER), 
             OD_UPPER = mean(OD_UPPER), CD_HAZARD = mean(CD_HAZARD), CD_LOWER = mean(CD_LOWER), CD_UPPER = mean(CD_UPPER))

# Exporting
write.csv(ages, "W:\\Matt P\\Projects\\202201 CISNET\\Hazard Functions\\EIN_AGE_RACE_HIST.csv", row.names = F)


# Uterine Multi-Cohort
test <- df %>% filter(HIST %in% c('EM', 'Non-EM')) %>% select(AGE, RACE, HIST, AJCC, MONTHS, EVENT)
test <- test %>% mutate(AGE = cut(AGE, c(0, 29, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84))) # Binning ages
test <- test %>% mutate(EVENT = ifelse(MONTHS > 120, 0, EVENT), MONTHS = ifelse(MONTHS > 120, 120, MONTHS))  # max follow up to 10 years
test <- test %>% mutate(YEARS = 1+MONTHS %/% 6) %>% select(-MONTHS)  # Binning into years
test$AJCC <- factor(test$AJCC, levels = c('I','II','III','IV'))

fit_OD <- stpm2(Surv(YEARS, EVENT==1)~AGE+RACE+HIST+AJCC, data = test, df=3)  # Creating model
fit_CD <- stpm2(Surv(YEARS, EVENT==2)~AGE+RACE+HIST+AJCC, data = test, df=3)

ages <- data.frame(expand.grid(unique(test$AGE), unique(test$RACE), unique(test$HIST), unique(test$AJCC), unique(test$YEARS)))
names(ages) <- c('AGE', 'RACE', 'HIST', 'AJCC', 'YEARS')
ages <- ages %>% arrange(AGE, RACE, HIST, AJCC, YEARS)

ages <- predict(fit_OD, newdata=ages, type="hazard", grid=F, full=T, se.fit=TRUE)
ages <- predict(fit_CD, newdata=ages, type="hazard", grid=F, full=T, se.fit=TRUE)

# Cleaning & Exporting
rm(test, fit_CD, fit_OD)
names(ages) <- c('AGE', 'RACE', 'HIST', 'AJCC', 'YEARS', 'OD_HAZARD', 'OD_LOWER', 'OD_UPPER', 'CD_HAZARD', 'CD_LOWER', 'CD_UPPER')
ages <- ages %>% mutate(YEARS = (YEARS-1) %/% 2) %>% group_by(AGE, RACE, HIST, AJCC, YEARS) %>% summarise(OD_HAZARD = mean(OD_HAZARD), OD_LOWER = mean(OD_LOWER), 
OD_UPPER = mean(OD_UPPER), CD_HAZARD = mean(CD_HAZARD), CD_LOWER = mean(CD_LOWER), CD_UPPER = mean(CD_UPPER))

# Exporting
write.csv(ages, "W:\\Matt P\\Projects\\202201 CISNET\\Hazard Functions\\UTERINE_AGE_RACE_HIST_AJCC.csv", row.names = F)



# EIN with Birth Cohort
test <- df %>% filter(HIST %in% c('EM', 'Non-EM'), RACE %in% c('NH White', 'NH Black')) %>% select(AGE, BIRTHYEAR, RACE, HIST, MONTHS, EVENT)
test <- test %>% mutate(AGE = cut(AGE, c(0, 29, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84))) # Binning ages
test <- test %>% mutate(BIRTHYEAR = cut(BIRTHYEAR, c(1900,1910,1920,1930,1940,1950,1960,1970,1980,1990), dig.lab=5)) # Binning cohorts
test <- test %>% mutate(EVENT = ifelse(MONTHS > 120, 0, EVENT), MONTHS = ifelse(MONTHS > 120, 120, MONTHS))  # max follow up to 10 years
test <- test %>% mutate(YEARS = 1+MONTHS %/% 6) %>% select(-MONTHS)  # Binning into years

fit_OD <- stpm2(Surv(YEARS, EVENT==1)~AGE+BIRTHYEAR+RACE+HIST, data = test, df=3)  # Creating model
fit_CD <- stpm2(Surv(YEARS, EVENT==2)~AGE+BIRTHYEAR+RACE+HIST, data = test, df=3)  # Creating model


ages <- data.frame(expand.grid(unique(test$AGE), unique(test$BIRTHYEAR), unique(test$RACE),unique(test$HIST), unique(test$YEARS)))
names(ages) <- c('AGE', 'BIRTHYEAR', 'RACE', 'HIST', 'YEARS')
ages <- ages %>% arrange(AGE, BIRTHYEAR, RACE, HIST, YEARS)
ages <- predict(fit_OD, newdata=ages, type="hazard", grid=F, full=T, se.fit=TRUE)
ages <- predict(fit_CD, newdata=ages, type="hazard", grid=F, full=T, se.fit=TRUE)

# Cleaning & Exporting
rm(test, fit_CD, fit_OD)
names(ages) <- c('AGE', 'BIRTHYEAR', 'RACE', 'HIST', 'YEARS', 'OD_HAZARD', 'OD_LOWER', 'OD_UPPER', 'CD_HAZARD', 'CD_LOWER', 'CD_UPPER')
ages <- ages %>% mutate(YEARS = (YEARS-1) %/% 2) %>% group_by(AGE, BIRTHYEAR, RACE, HIST, YEARS) %>% summarise(OD_HAZARD = mean(OD_HAZARD), OD_LOWER = mean(OD_LOWER), 
OD_UPPER = mean(OD_UPPER), CD_HAZARD = mean(CD_HAZARD), CD_LOWER = mean(CD_LOWER), CD_UPPER = mean(CD_UPPER))

# Exporting
write.csv(ages, "W:\\Matt P\\Projects\\202201 CISNET\\Hazard Functions\\EIN_AGE_BIRTH_RACE_HIST.csv", row.names = F)



# Uterine with Birth Cohort
test <- df %>% filter(HIST %in% c('EM', 'Non-EM')) %>% select(AGE, BIRTHYEAR, RACE, HIST, AJCC, MONTHS, EVENT)
test <- test %>% mutate(AGE = cut(AGE, c(0, 29, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84))) # Binning ages
test <- test %>% mutate(BIRTHYEAR = cut(BIRTHYEAR, c(1900,1910,1920,1930,1940,1950,1960,1970,1980,1990), dig.lab=5)) # Binning cohorts
test <- test %>% mutate(EVENT = ifelse(MONTHS > 120, 0, EVENT), MONTHS = ifelse(MONTHS > 120, 120, MONTHS))  # max follow up to 10 years
test <- test %>% mutate(YEARS = 1+MONTHS %/% 6) %>% select(-MONTHS)  # Binning into years
test$AJCC <- factor(test$AJCC, levels = c('I','II','III','IV'))
test <- test %>% filter(!is.na(BIRTHYEAR))

fit_OD <- stpm2(Surv(YEARS, EVENT==1)~AGE+BIRTHYEAR+RACE+HIST+AJCC, data = test, df=3)  # Creating model
fit_CD <- stpm2(Surv(YEARS, EVENT==2)~AGE+BIRTHYEAR+RACE+HIST+AJCC, data = test, df=3)

ages <- data.frame(expand.grid(unique(test$AGE), unique(test$BIRTHYEAR), unique(test$RACE), unique(test$HIST), unique(test$AJCC), unique(test$YEARS)))
names(ages) <- c('AGE', 'BIRTHYEAR', 'RACE', 'HIST', 'AJCC', 'YEARS')
ages <- ages %>% arrange(AGE, BIRTHYEAR, RACE, HIST, AJCC, YEARS)

ages <- predict(fit_OD, newdata=ages, type="hazard", grid=F, full=T, se.fit=TRUE)
ages <- predict(fit_CD, newdata=ages, type="hazard", grid=F, full=T, se.fit=TRUE)

# Cleaning & Exporting
rm(test, fit_CD, fit_OD)
names(ages) <- c('AGE', 'BIRTHYEAR', 'RACE', 'HIST', 'AJCC', 'YEARS', 'OD_HAZARD', 'OD_LOWER', 'OD_UPPER', 'CD_HAZARD', 'CD_LOWER', 'CD_UPPER')
ages <- ages %>% mutate(YEARS = (YEARS-1) %/% 2) %>% group_by(AGE, BIRTHYEAR, RACE, HIST, AJCC, YEARS) %>% summarise(OD_HAZARD = mean(OD_HAZARD), OD_LOWER = mean(OD_LOWER), 
OD_UPPER = mean(OD_UPPER), CD_HAZARD = mean(CD_HAZARD), CD_LOWER = mean(CD_LOWER), CD_UPPER = mean(CD_UPPER))

# Exporting
write.csv(ages, "W:\\Matt P\\Projects\\202201 CISNET\\Hazard Functions\\UTERINE_AGE_BIRTH_RACE_HIST_AJCC.csv", row.names = F)




# Plotting
df <- read.csv("W:\\Matt P\\Projects\\202201 CISNET\\Hazard Functions\\EIN_AGE_RACE_HIST.csv")
df %>% filter(HIST == 'EM', RACE == 'NH Black') %>% ggplot(aes(x=12*YEARS, y=CD_HAZARD, colour=AGE)) + geom_line() +
  xlab('Time from Diagnosis (Months)') + ylab('Probability of Cancer Death') + ggtitle("NH Black, EM Cancer Death Hazard")

df %>% filter(HIST_GRP == 'EM', RACE == 'NH Black') %>% ggplot(aes(x=AGE,y=RATE, colour=BIRTHYEAR)) + geom_line()  +
  xlab('Age (Years)') + ylab('Incidence Rate (per 100k)') + ggtitle("NH Black, EM Cancer Incidence by Cohort")

df %>% filter(HIST_GRP == 'Non-EM', RACE == 'NH Black') %>% ggplot(aes(x=AGE,y=RATE, colour=BIRTHYEAR)) + geom_line()  +
  xlab('Age (Years)') + ylab('Incidence Rate (per 100k)') + ggtitle("NH Black, Non-EM Cancer Incidence by Cohort")



# Bucketing Cohorts for Incidence
template <- read.csv("W:\\Matt P\\Projects\\202201 CISNET\\Uterine\\Imputation\\Template.csv")
template <- template %>% filter(RACE %in% c("White","Black"), HIST_GRP %in% c("EM","Non-EM"), AJCC == "I")
template$BIRTHYEAR <- template$YEAR - template$AGE
template <- template %>% mutate(BIRTHYEAR = cut(BIRTHYEAR, c(1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000), dig.lab =5)) %>% filter(!is.na(BIRTHYEAR))
template$RACE <- ifelse(template$RACE == "White", "NH White", 'NH Black')
template <- template %>% select(AGE, BIRTHYEAR, RACE, HIST_GRP, POP)

# EIN
inc <- read.csv("W:\\Matt P\\Projects\\202201 CISNET\\Calibration Data\\Uterine\\Common Data and Calibration Targets\\Other Data\\Incidence Covariates\\AGE_BIRTH_RACE_HIST.csv")
inc <- inc %>% filter(RACE %in% c("NH White","NH Black"), HIST_GRP %in% c("EM","Non-EM"))
inc <- inc %>% mutate(BIRTHYEAR = cut(BIRTHYEAR, c(1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000), dig.lab =5)) %>% filter(!is.na(BIRTHYEAR))

df <- merge(template, inc, by=c("AGE","BIRTHYEAR","RACE","HIST_GRP"), all=T)
df <- df %>% group_by(AGE, BIRTHYEAR, RACE, HIST_GRP) %>% summarise(RATE = weighted.mean(RATE,POP), LOWER = weighted.mean(LOWER,POP), UPPER = weighted.mean(UPPER,POP))
rm(inc, template)

write.csv(df, "W:\\Matt P\\Projects\\202201 CISNET\\Hazard Functions\\EIN_COHORT_INC.csv", row.names = F)

# Uterine
# Bucketing Cohorts for Incidence
template <- read.csv("W:\\Matt P\\Projects\\202201 CISNET\\Uterine\\Imputation\\Template.csv")
template <- template %>% filter(HIST_GRP %in% c("EM","Non-EM"))
template$BIRTHYEAR <- template$YEAR - template$AGE
template <- template %>% mutate(BIRTHYEAR = cut(BIRTHYEAR, c(1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000), dig.lab =5)) %>% filter(!is.na(BIRTHYEAR))
races <- c("NH White","NH Black","NH AI/AN","NH AAPI","Hispanic")
names(races) <- c("White","Black","AI/AN","AAPI","Hispanic")
template$RACE <- races[template$RACE]
rm(races)
template <- template %>% select(AGE, BIRTHYEAR, RACE, HIST_GRP, AJCC, POP)

# EIN
inc <- read.csv("W:\\Matt P\\Projects\\202201 CISNET\\Calibration Data\\Uterine\\Common Data and Calibration Targets\\Other Data\\Incidence Covariates\\AGE_BIRTH_RACE_HIST_AJCC.csv")
inc <- inc %>% filter(HIST_GRP %in% c("EM","Non-EM"))
inc <- inc %>% mutate(BIRTHYEAR = cut(BIRTHYEAR, c(1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000), dig.lab =5)) %>% filter(!is.na(BIRTHYEAR))

df <- merge(template, inc, by=c("AGE","BIRTHYEAR","RACE","HIST_GRP","AJCC"), all=T)
df <- df %>% group_by(AGE, BIRTHYEAR, RACE, HIST_GRP, AJCC) %>% summarise(RATE = weighted.mean(RATE,POP), LOWER = weighted.mean(LOWER,POP), UPPER = weighted.mean(UPPER,POP))
rm(inc, template)

write.csv(df, "W:\\Matt P\\Projects\\202201 CISNET\\Hazard Functions\\UTERINE_COHORT_INC.csv", row.names = F)


# Future Survival Trends
test <- df %>% filter(HIST %in% c('EM', 'Non-EM')) %>% select(AGE, BIRTHYEAR, RACE, HIST, AJCC, MONTHS, EVENT)
test <- test %>% mutate(AGE = cut(AGE, c(0, 29, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84))) # Binning ages
test <- test %>% mutate(BIRTHYEAR = cut(BIRTHYEAR, c(1900,1910,1920,1930,1940,1950,1960,1970,1980,1990), dig.lab=5)) # Binning cohorts
test <- test %>% mutate(EVENT = ifelse(MONTHS > 120, 0, EVENT), MONTHS = ifelse(MONTHS > 120, 120, MONTHS))  # max follow up to 10 years
test <- test %>% mutate(YEARS = 1+MONTHS %/% 6) %>% select(-MONTHS)  # Binning into years
test$AJCC <- factor(test$AJCC, levels = c('I','II','III','IV'))
test <- test %>% filter(!is.na(BIRTHYEAR))

# Converting birth year
cohorts <- c(5,15,25,35,45,55,65,75,85)
names(cohorts) <- c('(1900,1910]', '(1910,1920]', '(1920,1930]', '(1930,1940]', '(1940,1950]', '(1950,1960]', '(1960,1970]', '(1970,1980]', '(1980,1990]')
test$BIRTHYEAR <- cohorts[test$BIRTHYEAR]
rm(cohorts)

fit_OD <- stpm2(Surv(YEARS, EVENT==1)~AGE+BIRTHYEAR+RACE+HIST+AJCC, data = test, df=3)  # Creating model
fit_CD <- stpm2(Surv(YEARS, EVENT==2)~AGE+BIRTHYEAR+RACE+HIST+AJCC, data = test, df=3)

ages <- data.frame(expand.grid(unique(test$AGE), c(5,15,25,35,45,55,65,75,85,95,105,115,125,135,145), unique(test$RACE), unique(test$HIST), unique(test$AJCC), unique(test$YEARS)))
names(ages) <- c('AGE', 'BIRTHYEAR', 'RACE', 'HIST', 'AJCC', 'YEARS')
ages <- ages %>% arrange(AGE, BIRTHYEAR, RACE, HIST, AJCC, YEARS)

ages <- predict(fit_OD, newdata=ages, type="hazard", grid=F, full=T, se.fit=TRUE)
ages <- predict(fit_CD, newdata=ages, type="hazard", grid=F, full=T, se.fit=TRUE)

# Cleaning & Exporting
rm(test, fit_CD, fit_OD)
names(ages) <- c('AGE', 'BIRTHYEAR', 'RACE', 'HIST', 'AJCC', 'YEARS', 'OD_HAZARD', 'OD_LOWER', 'OD_UPPER', 'CD_HAZARD', 'CD_LOWER', 'CD_UPPER')
ages <- ages %>% mutate(YEARS = (YEARS-1) %/% 2) %>% group_by(AGE, BIRTHYEAR, RACE, HIST, AJCC, YEARS) %>% summarise(OD_HAZARD = mean(OD_HAZARD), OD_LOWER = mean(OD_LOWER), 
OD_UPPER = mean(OD_UPPER), CD_HAZARD = mean(CD_HAZARD), CD_LOWER = mean(CD_LOWER), CD_UPPER = mean(CD_UPPER))
rm(df)
ages$BIRTHYEAR <- ages$BIRTHYEAR + 1900
ages <- ages %>% mutate(BIRTHYEAR = cut(BIRTHYEAR, c(1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000,2010,2020,2030,2040,2050), dig.lab =5)) %>% filter(!is.na(BIRTHYEAR))
ages$OD_UPPER <- ifelse(ages$OD_UPPER > 1.0, 1.0, ages$OD_UPPER)
ages$CD_UPPER <- ifelse(ages$CD_UPPER > 1.0, 1.0, ages$CD_UPPER)
write.csv(ages, "W:\\Matt P\\Projects\\202201 CISNET\\Hazard Functions\\UTERINE_FUTURE_HAZARDS.csv", row.names = F)


ages %>% filter(HIST == 'EM', RACE == 'NH Black', AJCC == 'I', AGE == '(59,64]') %>% ggplot(aes(x=YEARS,y=CD_HAZARD,colour=BIRTHYEAR)) + 
  geom_line() + ggtitle("NH Black, EM Stage I, Age 60-64, Cancer Death")

df <- ages %>% filter(YEARS < 6) %>% group_by(AGE, BIRTHYEAR, RACE, HIST, AJCC) %>% summarise(CD_HAZARD = sum(CD_HAZARD))
blk <- df %>% filter(RACE == 'NH Black')
wht <- df %>% filter(RACE == 'NH White')

df <- merge(blk, wht, by=c("AGE","BIRTHYEAR","HIST","AJCC"), all=T)
# I want cumulative to 5 year cancer death, then ratio black/white
df <- df %>% mutate(HR = CD_HAZARD.x/CD_HAZARD.y)

df %>% filter(HIST == 'EM', BIRTHYEAR == "(2040,2050]") %>% ggplot(aes(x=AGE, group=AJCC)) + 
  geom_line(aes(y=CD_HAZARD.x)) + geom_line(aes(y=CD_HAZARD.y)) + ggtitle("NH Black, EM Stage I, Age 60-64, Cancer Death")



df <- read.csv("W:\\Matt P\\Projects\\202201 CISNET\\Hazard Functions\\UTERINE_AGE_BIRTH_RACE_HIST_AJCC.csv")
# Converting birth year
cohorts <- c(5,15,25,35,45,55,65,75,85)
names(cohorts) <- c('(1900,1910]', '(1910,1920]', '(1920,1930]', '(1930,1940]', '(1940,1950]', '(1950,1960]', '(1960,1970]', '(1970,1980]', '(1980,1990]')
df$BIRTHYEAR <- cohorts[df$BIRTHYEAR]
rm(cohorts)
df <- df %>% filter(YEARS < 6) %>% group_by(AGE, BIRTHYEAR, RACE, HIST, AJCC) %>% summarise(CD_HAZARD = sum(CD_HAZARD))

mod1 <- lm(CD_HAZARD~AGE+BIRTHYEAR+RACE+HIST+AJCC, data = df)
ages <- data.frame(expand.grid(unique(df$AGE), c(5,15,25,35,45,55,65,75,85,95,105,115,125,135,145), unique(df$RACE), unique(df$HIST), unique(df$AJCC)))
names(ages) <- c('AGE', 'BIRTHYEAR', 'RACE', 'HIST', 'AJCC')
ages <- ages %>% arrange(AGE, BIRTHYEAR, RACE, HIST, AJCC)

op <- predict(mod1, newdata=ages, grid=F, full=T, se.fit=TRUE)
ages$CD <- op$fit
ages$err <- op$se.fit
rm(op, mod1, df)
ages$CD <- ifelse(ages$CD <= 0.0, 0.0, ifelse(ages$CD >= 1.0, 1.0, ages$CD))

blk <- ages %>% filter(RACE == 'NH Black')
wht <- ages %>% filter(RACE == 'NH White')

df <- merge(blk, wht, by=c("AGE","BIRTHYEAR","HIST","AJCC"), all=T)
# I want cumulative to 5 year cancer death, then ratio black/white
df <- df %>% mutate(HR = CD.x/CD.y)
rm(blk, wht)
df$BIRTHYEAR <- 1900+df$BIRTHYEAR
df <- ungroup(df)

test <- df %>% filter(HIST == 'EM', AGE == "(74,79]") %>% select(BIRTHYEAR, HR, AJCC)
test %>% ggplot(aes(x=BIRTHYEAR, y=HR, colour=AJCC)) + geom_line() +
  ggtitle("Racial Disparity (Black/White), EM Cancer, Age 74-79, 5-Year Cancer Death HR") + ylim(0.0, 7.5)

# Population level mortality
df <- read.csv("C:\\Users\\mp4090\\Python\\GIT\\Columbia Projects\\Uterine Primary\\pop85.csv")
df <- df %>% filter(!is.na(`NH.White`))

df %>% ggplot(aes(x = Year)) + geom_line(aes(y = `NH.White`), colour='red') + geom_line(aes(y = `NH.Black`), colour='blue') + 
  ylim(0, 100000) + ylab("Proportion Alive at age 85") + ggtitle("Population Mortality")


# Gastric
df <- read.csv("W:\\Matt P\\Projects\\202201 CISNET\\Hazard Functions\\Gastric CL.csv")
df <- df %>% select(AGE,SEX,RACE,STAGE,HIST_GRP,Survival,STATUS) %>% filter(STATUS != "Unknown Death", HIST_GRP != "Other", RACE %in% c("NH White","NH Black"))
events <- c(0,1,2)  # Remapping event variable to numeric
names(events) <- c("Alive", "Other Death", "Cancer Death")
df$STATUS <- events[df$STATUS]
rm(events)
df$STAGE <- factor(df$STAGE, levels=c("Stage I","Stage II","Stage III","Stage IV"))

df <- df %>% mutate(AGE = cut(AGE, c(0, 29, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84))) # Binning ages
df <- df %>% mutate(STATUS = ifelse(Survival > 120, 0, STATUS), Survival = ifelse(Survival > 120, 120, Survival))  # max follow up to 10 years
df <- df %>% mutate(MONTHS = 1+Survival) %>% select(-Survival)  # Binning into years

fit_OD <- stpm2(Surv(MONTHS, STATUS==1)~AGE+SEX+RACE+STAGE+HIST_GRP, data = df, df=4)  # Creating model
fit_CD <- stpm2(Surv(MONTHS, STATUS==2)~AGE+SEX+RACE+STAGE+HIST_GRP, data = df, df=4)


ages <- data.frame(expand.grid(unique(df$AGE), unique(df$SEX), unique(df$RACE), unique(df$STAGE), unique(df$HIST_GRP), unique(df$MONTHS)))
names(ages) <- c('AGE', 'SEX', 'RACE', 'STAGE', 'HIST_GRP', 'MONTHS')
ages <- ages %>% arrange(AGE, SEX, RACE, STAGE, HIST_GRP, MONTHS)
ages <- predict(fit_OD, newdata=ages, type="hazard", grid=F, full=T, se.fit=TRUE)
ages <- predict(fit_CD, newdata=ages, type="hazard", grid=F, full=T, se.fit=TRUE)

# Cleaning & Exporting
names(ages) <- c('AGE', 'SEX', 'RACE', 'STAGE', 'HIST_GRP', 'YEARS', 'OD_HAZARD', 'OD_LOWER', 'OD_UPPER', 'CD_HAZARD', 'CD_LOWER', 'CD_UPPER')
ages <- ages %>% mutate(YEARS = (YEARS-1) %/% 12) %>% group_by(AGE, SEX, RACE, STAGE, HIST_GRP, YEARS) %>% summarise(OD_HAZARD = mean(OD_HAZARD), OD_LOWER = mean(OD_LOWER), 
OD_UPPER = mean(OD_UPPER), CD_HAZARD = mean(CD_HAZARD), CD_LOWER = mean(CD_LOWER), CD_UPPER = mean(CD_UPPER))
rm(df, fit_CD, fit_OD)

write.csv(ages, "W:\\Matt P\\Projects\\202201 CISNET\\Hazard Functions\\GASTRIC_AGE_SEX_RACE_STAGE_HIST.csv", row.names = F)

ages %>% filter(STAGE == "Stage IV", HIST_GRP == "Intestinal") %>% ggplot(aes(x=YEARS, y=CD_HAZARD, colour=AGE)) + 
  geom_line() + ggtitle("Stage IV, Intestinal GC") + facet_wrap(vars(SEX, RACE))




df <- read.csv("W:\\Matt P\\Projects\\202201 CISNET\\Calibration Data\\New Uterine\\Common Data and Calibration Targets\\SEER & CDC Data\\Incidence Joint Distribution.csv")
df <- df %>% select(YEAR,AGE,BIRTHYEAR,RACE,POP) %>% distinct()
write.csv(df, "W:\\Matt P\\Projects\\202201 CISNET\\Calibration Data\\New Uterine\\Common Data and Calibration Targets\\SEER & CDC Data\\SEER Population Counts.csv")





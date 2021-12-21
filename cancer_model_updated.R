citation()
##########################################################
#                                                        #
#  Group 14 Prediction of Breast Tumor Malignancy        #
#                                                        #
##########################################################

library(tidyverse)
library(InformationValue)

# Read data file
# cancer_data is the original data frame
df <- read.table(file = "data.csv", header = TRUE, sep = ",")

####################### DATA FILE CLEANING #######################

Diagnosis <- as.factor(df$diagnosis)
# Bar Graph of the data
df %>%
  ggplot(mapping = aes(x = Diagnosis,
                       fill = Diagnosis)) +
  geom_bar() +
  ggtitle(label = "Balance of Data") +
  xlab("Diagnosis") +
  ylab("Amount") 

# Convert the diagnosis from M and B to 1 and 0
df$diagnosis <- ifelse(df$diagnosis == "M", 1, 0)

# Creates our data frame with only the columns of interest

drop_col <- c(
  "id", "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
  "smoothness_worst", "compactness_worst", "concavity_worst",
  "concave.points_worst", "symmetry_worst", "fractal_dimension_worst", 
  "radius_se","texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se",
  "concavity_se", "concave.points_se", "symmetry_se","fractal_dimension_se"
)

# Full Data Set
cancer_data <- df[, !(names(df) %in% drop_col)]

# Ratio of malignant to benign tumors in this set
length(cancer_data$diagnosis[cancer_data$diagnosis == 1])/
  length(cancer_data$diagnosis[cancer_data$diagnosis == 0])

### Dividing the data into train/test
  
set.seed(666)

n <- as.vector(1:(nrow(cancer_data)))
n_test <- nrow(cancer_data)*.3
n_train <- nrow(cancer_data) - n_test

# Generate n_train random numbers without replacement
train_index <- sort(sample(n, n_train))
test_index <- sort(setdiff(n, train_index))

# Create training and test dataframes
train <- cancer_data[train_index, ]
test <- cancer_data[test_index, ]


####################### MODEL BUILDING #######################
attach(train)

# According to research, compactness, concavity, and smoothness are indicators
# Forwards Model Building

cor(train)

### Single variables

Con1 <- glm(diagnosis ~ concavity_mean, family = binomial(link = "logit"))
summary(Con1)

# Better concavity predictor, keep it
Con <- glm(diagnosis ~ concave.points_mean, family = binomial(link = "logit"))
summary(Con)

# Better "smoothness" (texture) predictor, keep it
texture <- glm(diagnosis ~ texture_mean, family = binomial(link = "logit"))
summary(texture)

smooth <- glm(diagnosis ~ smoothness_mean, family = binomial(link = "logit"))
summary(smooth)

Frac <- glm(diagnosis ~ fractal_dimension_mean, family = binomial(link = "logit"))
summary(Frac)

# Also wanted to see if the size of the tumor is a good predictor, since the
# correlations were high for those. I'll try perimeter first since the cor is highest.

perm <- glm(diagnosis ~ perimeter_mean, family = binomial(link = "logit"))
summary(perm)

# Higher AIC
rad <- glm(diagnosis ~ radius_mean, family = binomial(link = "logit"))
summary(rad)

# Also higher AIC
area <- glm(diagnosis ~ area_mean, family = binomial(link = "logit"))
summary(area)

# Concavity and Compactness are highly correlated, so since 
# AIC(Concavity) < AIC(compactness), leave out compactness to avoid 
# multi-collinearity, which can wreck the model 

Comp <- glm(diagnosis ~ compactness_mean, family = binomial(link = "logit"))
summary(Comp)


### Two variables

ConText <- glm(diagnosis ~ concave.points_mean + texture_mean, family = binomial(link = "logit"))
summary(ConText)
anova(Con, ConText, test = "Chisq") # Significant, nice

# AIC is way higher. Scrap. 
CompText <- glm(diagnosis ~ compactness_mean + texture_mean, family = binomial(link = "logit"))
summary(CompText)
anova(texture, CompText, test = "Chisq") # Significant, but still scrap

# Try perimeter and texture
PermText <- glm(diagnosis ~ perimeter_mean + texture_mean, family = binomial(link = "logit"))
summary(PermText)
anova(perm, PermText, test = "Chisq") # Significant, but still scrap

# Compactness and Perimeter
CompPerm <- glm(diagnosis ~ perimeter_mean + compactness_mean, family = binomial(link = "logit"))
summary(CompPerm)

anova(perm, CompPerm, test = "Chisq") # Significant, but still scrap.
                                      # AIC is much higher than concavity + texture


####################### TESTING #######################

detach(train)
attach(test)

# The best model
ConText <- glm(diagnosis ~ concave.points_mean + texture_mean, family = binomial(link = "logit"))
summary(ConText)

predicted <- predict(ConText,type="response")
predicted

# ROC curve
plotROC(actuals = diagnosis, predictedScores = predicted)

# Find the optimal cutoff for this set
optimal <- optimalCutoff(actuals = diagnosis, predictedScores = predicted)

# Actual: top-bottom, Predicted: left-right
confusionMatrix(actuals = diagnosis, predictedScores = predicted, threshold = optimal)

predicted <- (predict(ConText,type="response")>optimal)*1


# Sensitivity P(Y_hat=1|Y=1)
sum((predicted==1 & diagnosis==1))/sum(diagnosis==1)

# Specificity 
sum((predicted==0 & diagnosis==0))/sum(diagnosis==0)

# Accuracy
(sum((predicted==1 & diagnosis==1))+sum((predicted==0 & diagnosis==0)))/length(diagnosis)

# Precision P(Y=1|Y_hat=1)
sum((predicted==1 & diagnosis==1))/sum(predicted==1)

# Precision, but for 0s P(Y=0|Y_hat=0)
sum((predicted==0 & diagnosis==0))/sum(predicted==0)
# Heart-Disease-Prediction-Using-a-Random-Forrest-Model-in-R

When I first began learning about data science, machine learning seemed unapproachable. I thought that to understand machine learning algorithims I would to master statistics and linear aglebra and to implemnt them I would need to be a brilliant programmer. As my math skills improved, I began to grow more comfortable with model fitting and while I could easiy build a linear regression model in R, I had no idea where to begin as far as implementing more advanced models.

That all changed after I attended this years (virtual) Open Data Science conference and participated in an R machine learning training using TidyModels. I saw how these ingenious packages make it easy to fit, test, tune and evaluate a wide variety of diverse model types without getting bogged down in disparate syntax. 

After attending some hands on tutorials using various types of decision tree models, I decided to try my hand at using a random forest classification model. The University of California at Irvine curates excellent real world data sets, and I decided to use their heart disease dataset to predict if individuals had heart disease. Here is the url: https://archive.ics.uci.edu/ml/datasets/Heart+Disease. 

You will see that the UCI has data from four different hospitals however a combined version was uploaded to kaggle. In this version the target (outcome) variable was converted into 0 or 1 (pressence of heart disease or not) instead of 0 - 4. Here is the url to this version of the data which I used: https://www.kaggle.com/ronitf/heart-disease-uci.

Random forest models are based on decision tree models, a simple model type of model where each data point is evaluated on whether or not it passes a logical test based on one or more variables. The result of the test determines which branch of the tree that data point will continue down, determining the next logical test it will encounter. After passing through enough nodes to catagorize the data point with enough specificity to assign it a discrete classification its placed in a 'bin'. In a classification model, each bin represents one of the possible outcome catagories.

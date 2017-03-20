# Movie-Revenue-And-Rating-Predictor
We are making an app for predicting imdb rating for a new movie before its release. This app can be used by movie producers to choose directors, actors, etc. for their coming movies based on the prediction by the app.

We will be prediciting the rating by following two approaches :
1.) Clasification 2.) Regression

In classification apporach, we assume the rating to be a discrete variable that can take 101 values from the set {0.0, 0.1, 0.2, .. 1.1, 1.2, .. , 2.1, 2.2, .. ,9.8, 9.9, 10.0}. [Classification algorithms used will be updated as soon as it is implemented].
  In regression approach, we assume the movie rating to be a continuous variable taking values from [0,10] with precision upto one decimal point. [Regression algorithms used will be updated as soon as it is implemented].

# Preprocessing:
Some of fields are empty in the dataset. These fields are empty because of unavailability of the data. If director name, actor name or any other attribute which is text, we are deleting the complete row from the dataset. If we don't do so and try to fill this by randomly selecting a director name or actor name from the available names in the dataset, it will lead to error in calculation of the weight for the movie whose director name has been randomly selected. We can't assign a hypothetical name (say D_NAME) for unfilled entry as this will make all these data points a different cluster which is unrelated to the rest of the dataset.Hence, not a meaning ful decision. For numerical attributes which are unfilled, we are filling it with the average of all the other entries. We are converting text field to number by assigning a number to each unique name i.e. if there are 100 unique directors, we assign it numbers from 0 to 99.

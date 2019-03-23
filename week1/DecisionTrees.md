### Decision Trees

I would like to share the results of decision tree analysis which was performed on a portion of the GapMinder data that includes one year of numerous country-level indicators of health, wealth and development. 

The following explanatory variables were included to perform classification:
  - incomeperperson (GDP per capita)
  - alcconsumption (alcohol consumption per adult age 15+)
  - armedforcesrate (armed forces personnel)
  - breastcancerper100th (breast cancer new cases per 100,000 female)
  - co2emissions (cumulative CO2 emission)
  - femaleemployrate (female employees age 15+)
  - hivrate (estimated HIV Prevalence %)
  - lifeexpectancy,	oilperperson (oil consumption per capita)
  - polityscore (democracy score)
  - relectricperperson (residential electricity consumption)
  - suicideper100th (Suicide, age adjusted, per 100 000)
  - employrate (total employees age 15+)
  - urbanrate (urban population)

Variable 'internetuserate' (Internet users per 100 people) was transformed to response variable in the following way: if value is greater or equal 70, then it is classified as 1, if value is less than 70, then it is classified as 0.

There were 2 iterations of algorithm, the first one when all not available values were excluded and the second one when all not available values were replaced with mean, hence 2 decision trees.

In the first run [MyGap.png](https://github.com/kkrasilschikova/ml-for-data-analysis/blob/master/week1/MyGap.png) 213 samples were initially split into two subgroups by 'incomeperperson' variable, 25 people with income less or equal 15917.129 were not marked as Internet users. The further subdivision was made by 'relectricperperson' variable, one person with income more than 15917.129 and electricity consumption less or equal 5.472 was marked as not Internet user, while 7 people with income more than 15917.129 and electricity consumption more than 5.472 were marked as Internet users. Accuracy score is 0.82, 19 samples were defined correctly while 4 were marked incorrectly according to confusion matrix.

The second run [MyGapMean.png](https://github.com/kkrasilschikova/ml-for-data-analysis/blob/master/week1/MyGapMean.png) where all not available values were transformed to mean shows a different decision tree for 56 samples. There are much more splits of samples however the first one was also made on 'incomeperperson' variable (less or equal 20919.645). Accuracy score is 0.8, 69 samples were defined correctly while 17 were marked incorrectly according to confusion matrix.

Python programme may be found here https://github.com/kkrasilschikova/ml-for-data-analysis/blob/master/week1/decision-tree.py

# Lab---Task-1---Box-Office-Revenue-Prediction-
*submitted by: Shani Pais @shanipa and Liam Hazan @liamhazan*
## Files:
* `labHW1.html` is our lab report describing the EDA process we have made and the feature engeneering, model selction, and hyper-tuning process we have preformed. `labHW1.ipynb` is the Jupyter notebook we wrote in order to build it.
* `train.ipynb` is the training pipline we did in order to recive our final prediction model
* `train.tsv`, `test.tsv` - train and test data.
### In order to make prediction on new data you will need:
1. Install all the packeges and dependncies as decribed in `environment.yml`
2. use `predict.py` to predict the revanue of your desired moives: <br>
 ``` python predict.py test.tsv ``` <br>
 the file will need `our_utils.py` for the preprocessing, and `extrees.pkl` to load the trained model and the transformations pipline
 3. you will find your prediction per movie id in a newly created file: `prediction.csv`

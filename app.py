from flask import Flask,url_for,redirect,request,render_template
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application = Flask(__name__)

app = application
## when the webpage loads the home page
@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict_data',methods=['GET','POST'])
def predict_data_section():
    data = CustomData(
        gender = request.form.get('gender'),
        race_ethnicity = request.form.get('race_ethnicity'),
        parental_level_of_education = request.form.get('parental_level_of_education'),
        lunch = request.form.get('lunch'),
        test_preparation_course = request.form.get('test_preparation_course'),
        writing_score = request.form.get('writing_score'),
        reading_score = request.form.get('reading_score'),
        
    )
    predict_df = data.getdata_as_dataframe()
    print(predict_df)

    pred_pipeline = PredictPipeline()
    result = pred_pipeline.predict(features=predict_df)
    return render_template('home.html',result = result[0])
if __name__=='__main__':
    app.run(debug=True)

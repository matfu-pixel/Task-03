from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed
from wtforms import StringField, IntegerField, SelectField, SubmitField, FloatField
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms.validators import DataRequired, Optional, NumberRange


class FileForm(FlaskForm):
    name = StringField("Dataset's name:", validators=[DataRequired()])
    data_train = FileField('Load train dataset:', validators=[
        FileRequired(),
        FileAllowed(['csv'], 'Only .csv files!')
    ])
    data_val = FileField('Load validation dataset:', validators=[
        Optional(),
        FileAllowed(['csv'], 'Only .csv files!')
    ])
    target_name = StringField('Target name:', validators=[
        DataRequired()
    ])
    submit = SubmitField('Add data')

class SelectModelForm(FlaskForm):
    types = [
        ('RandomForest', 'Random Forest'),
        ('GradientBoosting', 'Gradient Boosting')
    ]
    name = StringField('Choose model name:', validators=[DataRequired()])
    type = SelectField('Choose model type:', choices=types)
    submit = SubmitField('Choose model')

class SelectHyperForm(FlaskForm):
    n_estimators = IntegerField('Number of trees in ensemble:',
                                validators=[DataRequired(), NumberRange(min=1, max=10000)], default=100)
    feature_subsample_size = IntegerField('Maximum number of features:',
                                           validators=[DataRequired(), NumberRange(min=1, max=1000)], default=10)
    max_depth = IntegerField('Maximum depth:',
                             validators=[DataRequired(), NumberRange(min=1, max=100)], default=10)
    learning_rate = FloatField('Learning rate:',
                               validators=[DataRequired(), NumberRange(min=0.00001, max=1)], default=0.1)
    submit = SubmitField('Add model')

class ChooseModelAndDataForm(FlaskForm):
    data = SelectField('Choose data:', validators=[DataRequired()])
    model = SelectField('Choose model: ', validators=[DataRequired()])
    submit = SubmitField('Next')

class ShowInfoForm(FlaskForm):
    submit = SubmitField('Train')

class PlotPredictForm(FlaskForm):
    test = FileField('Test dataset:',
                     validators=[FileRequired(),
                     FileAllowed(['csv'], 'Only .csv files!')])
    submit = SubmitField('Predict')

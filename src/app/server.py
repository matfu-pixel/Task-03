from flask import Flask, request, url_for, render_template, redirect, Response, send_from_directory
from flask_forms import FileForm, SelectModelForm, SelectHyperForm, ChooseModelAndDataForm, ShowInfoForm, PlotPredictForm
from dataset import Dataset
from model import Model
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io, os


app = Flask(__name__)
app.config['SECRET_KEY'] = 'matfu21'

menu = [{'name': 'Main', 'url': '/index'},
        {'name': 'Add model', 'url': '/add_model'},
        {'name': 'Add data', 'url': '/load_dataset'},
        {'name': 'Model training', 'url': '/choose_model_data'},
        {'name': 'About me', 'url': '/index'}]

models = {}
datasets = {}

@app.route('/')
@app.route('/index')
def get_index():
    """
    main page
    """
    return render_template('index.html', menu=menu, page_title='Main page')

@app.route('/load_dataset', methods=['GET', 'POST'])
def add_data():
    """
    page for load data
    """
    file_form = FileForm()
    if file_form.validate_on_submit(): # check csrf_token
        data_train = pd.read_csv(file_form.data_train.data)
        app.logger.info(file_form.data_val.data)
        if file_form.data_val.data is not None:
            data_val = pd.read_csv(file_form.data_val.data)
        else:
            data_val = None
        target_name = file_form.target_name.data
        datasets[file_form.name.data] = Dataset(target_name, data_train, data_val)
        return redirect(url_for('add_data'))

    return render_template('add_data.html', form=file_form, menu=menu, page_title='Load dataset')

@app.route('/add_model', methods=['GET', 'POST'])
def add_model():
    """
    select model for fitting
    """
    add_model_form = SelectModelForm()
    if add_model_form.validate_on_submit():
        name = add_model_form.name.data
        type = add_model_form.type.data
        return redirect(url_for('set_hyper', name=name, type=type))

    return render_template('add_ensemble.html', form=add_model_form, menu=menu, page_title="Add model")

@app.route('/hyperparameters/<type>/<name>', methods=['GET', 'POST'])
def set_hyper(type=None, name=None):
    """
    set hyperparameters for model with name - name and with type - type
    """
    hyper_form = SelectHyperForm()

    if hyper_form.validate_on_submit():
        n_estimators = hyper_form.n_estimators.data
        feature_subsample_size = hyper_form.feature_subsample_size.data
        max_depth = hyper_form.max_depth.data
        learning_rate = hyper_form.learning_rate.data
        models[name] = Model(type, n_estimators, feature_subsample_size, max_depth, learning_rate)
        return redirect(url_for('add_model'))

    return render_template('hyperparameters.html', form=hyper_form, menu=menu, page_title='Choose parameters for ' + type, type=type, name=name)

@app.route('/choose_model_data', methods=['GET', 'POST'])
def choose_model_data():
    form = ChooseModelAndDataForm(meta={'csrf': False})
    form.data.choices = list(datasets.keys())
    form.model.choices = list(models.keys())

    if form.validate_on_submit():
        data_name = form.data.data
        model_name = form.model.data
        return redirect(url_for('show_info', data_name=data_name, model_name=model_name))

    return render_template('choose_model_data.html', form=form, menu=menu, page_title="Choose model and data")

@app.route('/show_info/<data_name>/<model_name>', methods=['GET', 'POST'])
def show_info(data_name=None, model_name=None):
    form = ShowInfoForm()

    if form.validate_on_submit():
        return redirect(url_for('plot_predict', data_name=data_name, model_name=model_name))
    return render_template('show_info.html', form=form, menu=menu, page_title="Information about model", data_name=data_name, model_name=model_name, model=models[model_name])

@app.route('/plot_predict/<data_name>/<model_name>', methods=['GET', 'POST'])
def plot_predict(data_name=None, model_name=None):
    form = PlotPredictForm()
    models[model_name].fit(datasets[data_name].data_train,
                           datasets[data_name].data_val,
                           datasets[data_name].target_name)

    if form.validate_on_submit():
        test_data = pd.read_csv(form.test.data)
        path_to_tmp = os.path.join(os.getcwd(), 'tmp/')
        if not os.path.exists(path_to_tmp):
            os.mkdir(path_to_tmp)
        file_name = form.test.name + '_pred.csv'
        pred = models[model_name].predict(test_data)
        pd.DataFrame({datasets[data_name].target_name: pred}).to_csv(os.path.join(path_to_tmp, file_name), index=False)
        return send_from_directory(path_to_tmp, file_name, as_attachment=True)


    return render_template('plot_predict.html', form=form, menu=menu, page_title="Make prediction", data_name=data_name, model_name=model_name)

@app.route('/plot_img/<model_name>')
def plot_img(model_name):
    score_train = models[model_name].history['score_train']
    if 'score_val' in models[model_name].history:
        score_val = models[model_name].history['score_val']
    else:
        score_val = None

    fig, ax = plt.subplots(figsize=(6, 4), dpi=700)
    ax.set_title('RMSE on number of trees')
    ax.set_xlabel('Number of trees')
    ax.set_ylabel('RMSE')
    ax.plot(score_train, label='train')
    if score_val is not None:
        ax.plot(score_val, label='val')
    ax.legend()
    ax.grid()
    fig.tight_layout()
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)

    return Response(output.getvalue(), mimetype='img/png')
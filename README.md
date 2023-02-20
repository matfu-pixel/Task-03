# Веб-сервис "Ансамбли для решения задачи регрессии"

## Запуск
**1 способ:** В корне проекта собрать докер образ с помощью ```scripts/build.sh```, запустить с помощью ```scripts/run.sh```  

**2 способ:** Скачать докер образ с docker.hub с помощью ```docker pull matfu21/ensemble_server```, запустить с помощью ```docker run --rm -p 5000:5000 matfu21/ensemble_server```

## Работа с сервисом

### 0. Главная
![alt text](https://github.com/matfu-pixel/Task-03/blob/description/images/0.png)

### 1. Загрузка модели
![alt text](https://github.com/matfu-pixel/Task-03/blob/description/images/1.png)
![alt text](https://github.com/matfu-pixel/Task-03/blob/description/images/2.png)
Поддерживаются два ансамбля: Случайный лес и Градиентный бустинг. Ограничения на параметры:
* 1 <= n_estimators <= 10000
* 1 <= max_depth <= 1000
* 1 <= feature_subsample_size <= 1000
* 0.00001 <= learning_rate <= 1

### 2. Загрузка данных
![alt text](https://github.com/matfu-pixel/Task-03/blob/description/images/3.png)
Для данных должно выполняться:
* Все признаки числовые (предобработка категориальных не предусмотрена)
* Согласованость тренировочной, валидационной и тестовой выыборок
* Данные для тестирования расположены в папке /data

### 3. Обучение модели, мониторинг и предсказание
![alt text](https://github.com/matfu-pixel/Task-03/blob/description/images/4.png)
![alt text](https://github.com/matfu-pixel/Task-03/blob/description/images/5.png)
![alt text](https://github.com/matfu-pixel/Task-03/blob/description/images/6.png)

* Можно добавлять различное число разных моделей и разных данных, далее можно выбрать нужную пару модель-данные и на этих данных обучить эту модель
* Естественно данные и модель должны согласоваться
* На 2 скриншоте показаны параметры выбранной модели. После того, как параметры были проверены, можно начать обучение модели
* В качестве результата обучения выводится график зависимости RMSE от числа базовых алгоритмов в ансамбле, а также есть возможность загрузить датасет, совпадающий по формату с обучающей выборкой, на котором модель может сделать предсказание

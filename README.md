#  Hyperparameter Tuning con Keras Tuner

Este proyecto demuestra cómo usar **Keras Tuner** para encontrar automáticamente los mejores hiperparámetros en una red neuronal construida con Keras, utilizando el dataset MNIST como ejemplo.

---

##  Requisitos

Este proyecto ha sido probado con:

- `Python 3.10+`
- `TensorFlow 2.16.2`
- `Keras Tuner 1.4.7`
- `NumPy < 2.0.0`
- `scikit-learn`

Puedes instalar las dependencias con:

```bash
pip install -r requirements.txt
```

---

##  Clonar y ejecutar

```bash
git clone https://github.com/jondalar24/tuner.git
cd tuner
python tuner_script.py
```

---

##  Personalización y mejoras

Puedes mejorar este script fácilmente. Aquí van algunas ideas:

### 1.  Probar otros algoritmos de búsqueda

Por defecto, se usa `RandomSearch`, pero puedes sustituirlo por:

```python
from keras_tuner import Hyperband

tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    directory='my_dir',
    project_name='kt_hyperband'
)
```

O también:

```python
from keras_tuner import BayesianOptimization

tuner = BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    directory='my_dir',
    project_name='kt_bayes'
)
```

---

### 2.  Ajustar más hiperparámetros

Dentro de tu función `build_model(hp)` puedes explorar nuevos parámetros:

```python
hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])
hp.Boolean('use_dropout')
```

Y adaptar la arquitectura:

```python
if hp.Boolean('use_dropout'):
    model.add(Dropout(0.3))
```

---

### 3.  Cambiar el dataset

Este script utiliza MNIST, pero puedes adaptarlo a cualquier otro dataset disponible en `tf.keras.datasets` o tus propios datos.

---

 ¡Explora, ajusta y mejora tu modelo sin miedo a los hiperparámetros!


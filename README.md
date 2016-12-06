Reconocimiento visual empleando técnicas de deep learning
---------------------------------------------------------

Trabajo final de grado para la obtencion del titulo Licenciatura en Ciencias de la Computacion

Este repositorio cuenta con todo el codigo necesario para reproducir los resultados de mi tesis de Licenciatura.

Resumen
-------
 En los últimos años ha habido un resurgimiento en el campo de la Inteligencia Artificial (IA) tanto en la academia como en la industria. Esto se debe, por un lado, al abaratamiento de hardware con gran poder de cómputo y, por el otro, al incremento en el volumen de datos disponible. A su vez, Grandes empresas como Google, Facebook o Microsoft han creado sus propios equipos de investigación en IA obteniendo resultados de vanguardia.

 Dentro del campo de la IA, una clase de técnicas conocidas como Deep Learning (DL) han cobrado particular relevancia, ya que mediante su utilización se han conseguido mejoras muy significativas respecto de métodos tradicionales. Una desventaja de los modelos basados en DL, es que para su entrenamiento es necesario contar con miles o millones de datos anotados. En el caso particular de la clasificación de imágenes por contenido, si bien existen grandes conjuntos de datos anotados disponibles (ImageNet o LFW, entre otros), su generación para problemas no contemplados en los mismos es muy costosa. Por ejemplo, si deseáramos generar un modelo para clasificar imágenes del “arco de Córdoba”, muy dificilmente dicha categoría se encuentre representada en los conjuntos anteriores.

 [Agrawal et al.](https://arxiv.org/pdf/1505.01596v2.pdf)  proponen una manera alternativa al entrenamiento de esta clase de modelos inspirada en cómo los organismos vivientes desarrollan habilidades de percepción visual: moviéndose e interactuando con el mundo que los rodea. Partiendo de la hipótesis de que se puede usar la información del movimiento propio (rotación y traslación en los ejes X,Y,Z) como método de supervisión ellos demostraron que es posible obtener buenos resultados entrenando con menos imágenes anotadas que lo usual.


Estructura
----------

- `report` contiene el reporte final de la tesis.
- `experiments` contiene todos los scripts, datos y codigo relevante a la reproduccion de los experimentos.

Requerimientos
--------------

- [Caffe](http://caffe.berkeleyvision.org/)
- `sudo ./setup.sh` (principalmente dependencias para compilar el reporte a LaTeX)
- `pip install -r requirements.txt`

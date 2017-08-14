# Entrenamiento de modelos de aprendizaje profundo mediante autosupervisión 

Trabajo final de grado para la obtencion del título Licenciatura en Ciencias de la Computacion de la Facultad de Matemática, Astronomía, Física y Computación, Universidad Nacional de Córdoba.


Este repositorio cuenta con todo el codigo necesario para reproducir los resultados de mi tesis de Licenciatura.

Este repositorio está bajo licencia [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).

## Resumen
 En los últimos años ha habido un resurgimiento en el campo de la Inteligencia Artificial (IA) tanto en la academia como en la industria. Esto se debe, por un lado, al abaratamiento de hardware con gran poder de cómputo y, por el otro, al incremento en el volumen de datos disponible. A su vez, Grandes empresas como Google, Facebook o Microsoft han creado sus propios equipos de investigación en IA obteniendo resultados de vanguardia.

 Dentro del campo de la IA, una clase de técnicas conocidas como Deep Learning (DL) han cobrado particular relevancia, ya que mediante su utilización se han conseguido mejoras muy significativas respecto de métodos tradicionales. Una desventaja de los modelos basados en DL, es que para su entrenamiento es necesario contar con miles o millones de datos anotados. En el caso particular de la clasificación de imágenes por contenido, si bien existen grandes conjuntos de datos anotados disponibles (ImageNet o LFW, entre otros), su generación para problemas no contemplados en los mismos es muy costosa. Por ejemplo, si deseáramos generar un modelo para clasificar imágenes del “arco de Córdoba”, muy dificilmente dicha categoría se encuentre representada en los conjuntos anteriores.

 [Agrawal et al.](https://arxiv.org/pdf/1505.01596v2.pdf)  proponen una manera alternativa al entrenamiento de esta clase de modelos inspirada en cómo los organismos vivientes desarrollan habilidades de percepción visual: moviéndose e interactuando con el mundo que los rodea. Partiendo de la hipótesis de que se puede usar la información del movimiento propio (rotación y traslación en los ejes X,Y,Z) como método de supervisión ellos demostraron que es posible obtener buenos resultados entrenando con menos imágenes anotadas que lo usual.


## Estructura

- `report` contiene el reporte final de la tesis.
- `experiments` contiene todos los scripts, datos y codigo relevante a la reproduccion de los experimentos.

## Requerimientos

- [Caffe](http://caffe.berkeleyvision.org/)
- `sudo ./setup.sh` (principalmente dependencias para compilar el reporte a LaTeX)
- `pip install -r requirements.txt`


## Experimentos

### 1. Prueba de concepto con MNIST

Para explorar la feasibilidad de la idea y chequear detalles de su implementacion se procedio a realizar una prueba de concepto utilizando el conjunto de datos [MNIST](http://yann.lecun.com/exdb/mnist/).

#### 1.1 Conjunto de datos
El conjunto de datos MNIST cuenta con 60000 imágenes de caracteres numéricos manuscritos para entrenamiento, más 10000 imágenes para evaluación. La dimensión de cada imagen es 1 x 28 x 28.  Para el entrenamiento mediante *automovimiento* se crearon pares de imágenes siguiendo los lineamientos del paper. Esto significa que cada par está compuesto de la imagen original y la imagen con transformaciones en los ejes X, Y, Z. Las
transformaciones en X e Y son traslaciones de 3 píxeles, mientras que la rotación en Z varía entre los -30° y los 30°. Tanto las rotaciones como las traslaciones son números enteros. Para cada par creado las
transformaciones se eligen de manera aleatoria uniforme.


#### 1.2 Arquitectura de la red
La arquitectura utilizada para las BCNN fue C96-P-C256-P, donde C96 significa una capa convolucional de 96 filtros y P significa una capa de MAX-POOLING. Para la
TCNN se eligió F1000-D-Op , donde F es una capa *Fully Connected* con 1000 salidas, D es el dropout y Op es la salida de la red (una *fully connected* con un clasificador Softmax). Tener en cuenta que para el caso del
*automovimiento* es necesario utilizar una combinación
FC-Softmax para calcular la pérdida en cada una de las
transformaciones.

Para transferencia de aprendizaje se añadió F500-D-F10-Softmax a una BCNN.

Para mas informacion respecto a la arquitectura de las redes y la nomenclatura leer la Sección 3 del [paper original](https://arxiv.org/pdf/1505.01596.pdf).

#### 1.3 Entrenamiento y Resultados

Las redes siamesas se pre-entrenaron durante 40000 iteraciones con una tasa de aprendizaje de 0.01. Siguiendo los lineamientos del paper se utilizaron márgenes *m* de 10 y 100 para SFA por ser los que mejores
resultados lograron. En ambas redes la tasa de aprendizaje se reduce a la mitad cada 10000 iteraciones. El tamaño del *mini batch* fue de 125, lo cual equivale a procesar 5 millones de pares de imágenes
durante las 40000 iteraciones del entrenamiento.

La etapa de transferencia de aprendizaje se hizo con 4000 iteraciones a una tasa de aprendizaje constante de 0.01.

Para evaluar la calidad de las *features* aprendidas por las redes siamesas se estableció la tasa de aprendizaje de las capas convolucionales a cero.

En el siguiente se puede observar la exactitud (*accuracy*) obtenida mediante la transferencia de aprendizaje con 100, 300, 1000 y 10000 imágenes de los dos métodos utilizados (automovimiento y SFA) y una
comparación con un entrenamiento desde cero utilizando esa misma cantidad de imágenes.

| Método         |  100 |  300 | 1000 | 10000 |
|:---------------|-----:|-----:|-----:|------:|
| Desde cero     | 0.42 | 0.70 | 0.82 |  0.97 |
| SFA(m=10)      | 0.52 | 0.71 | 0.77 |  0.82 |
| SFA(m=100)     | 0.58 | 0.73 | 0.80 |  0.88 |
| Automovimiento | 0.75 | 0.90 | 0.92 |  0.99 |


Se puede observar que entrenar mediante automovimiento presenta una performance claramente superior a entrenar una red desde cero con la misma cantidad de imágenes en los casos en los que el conjunto de
datos es relativamente pequeño. Es también superior al entrenamiento utilizando *Slow Feature Analysis*, y dado que no se
modificaron los pesos de las capas convolucionales aprendidas durante el pre-entrenamiento, podemos concluir que las *features* aprendidas son buenas y logran captar las representaciones necesarias
para el domino del problema en cuestión. El siguiente paso es verificar que efectivamente las *features* aprendidas se puedan aplicar a diferentes dominios de problemas y sean lo suficientemente generalizables.


### 2. Pruebas utilizando los conjuntos de datos KITTI, SUN392 e ILSVRC'12

#### 2.1 Conjuntos de datos

El conjunto de datos [KITTI](http://www.cvlibs.net/datasets/kitti/) consiste en 11 secuencias que registran el movimiento de un automóvil en una ciudad. Además de proveer cuadros de video, se encuentra la
información odométrica recolectada por sensores montados en el automóvil. Esa misma información es la que usa Agrawal et al. a la hora de computar las transformaciones en la cámara entre pares de imágenes, y es la que intentaremos reproducir en esta sección.

Se asume que la dirección a la que apunta la cámara es el eje Z y el plano de la imagen es el plano XY (ejes horizontales y verticales). Dado que las transformaciones más significativas de la
cámara ocurren en los ejes Z/X (a medida que el automóvil avanza por la calle) y sobre el eje Y (cuando el automóvil gira), sólo se tomaron en cuenta esas tres dimensiones a la hora de analizar las transformaciones.

Nuevamente, la predicción de transformaciones se establece como una tarea de clasificación, esta vez con 20 clases para las transformaciones en cada eje. Siguiendo los lineamientos originales del paper, los pares de entrenamiento se tomaron de cuadros separados
a lo sumo por 7 cuadros intermedios. Similarmente, para entrenamiento por SFA se consideraron a los cuadros separados por 7 cuadros intermedios como similares.

Finalmente, las redes siamesas fueron entrenadas a partir de parches de 227 x 227 extraídos aleatoriamente de las imágenes originales de 1241 x 376 píxeles. No se aplicaron transformaciones extras más allá de las otorgadas por el movimiento de la cámara.

El conjunto de datos [SUN-397](http://vision.princeton.edu/projects/2010/SUN/) consiste de 397 categorías de paisajes interiores y exteriores y además provee 10 particiones del
dataset para hacer *cross-validation*, pero debido a lo costoso que es entrenar redes neuronales convolucionales solo se utilizaron tres particiones.

El conjunto de datos [ILSVRC'12](http://image-net.org/challenges/LSVRC/2012/results.html) cuenta con 1000 clases de objetos distintos.

#### 2.2 Arquitectura de la red 

La red utilizada como base de las BCNN está inspirada en las primeras 5 capas convolucionales de [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), es 
decir, C96-P-C256-P-C384-C384-C256-P. La TCNN fue definida como C256-C128-F500-D-Op, con filtros convolucionales de 3 x 3.


#### 2.3 Evaluación de las features utilizando el conjunto de datos SUN397

La evaluación se hizo midiendo la exactitud de clasificadores *Softmax* utilizando las *features* obtenidas de las
salidas de las primeras 5 capas convolucionales (nombradas L1-L5). Los resultados pueden verse en el siguiente cuadro, donde *#preentr.* es la cantidad de datos utilizandos en 
el pre-entrenamiento de las redessiamesas (excepto para el caso ALEX-1000 y ALEX-20 que fueron entrenadas de cero con dicha cantidad de datos). *#finet.* es la cantidad de datos de SUN397 utilizados durante la 
transferencia de aprendizaje. 

| Método    | #preentr. | #finet. |   L1 |   L2 |   L3 |   L4 |    L5 | #finet. |   L1 |    L2 |    L3 |    L4 |    L5 |
|:----------|:---------:|--------:|-----:|-----:|-----:|-----:|------:|--------:|-----:|------:|------:|------:|------:|
| ALEX-1000 | 1M        |       5 | 3.73 | 5.07 | 5.07 | 8.53 | 10.40 |      20 | 9.07 | 12.53 | 16.27 | 17.60 | 10.67 |
| ALEX-20   | 20K       |       5 | 2.93 | 1.87 | 3.73 | 5.07 |  3.20 |      20 | 6.13 |  5.33 |  5.33 |  4.53 |  5.07 |
| KITTI-SFA | 20.7K     |       5 | 2.13 | 3.20 | 2.40 | 1.60 |  1.87 |      20 | 4.53 |  3.73 |  2.13 |  2.40 |  2.93 |
| KITTI-EGO | 20.7K     |       5 | 2.93 | 1.87 | 3.20 | 5.87 |  1.33 |      20 | 6.67 |  7.47 |  9.87 |  9.33 |  4.00 |

Se puede observar que el pre-entrenamiento con automoviento no supera la performance que se obtiene al pre-entrenar la misma red con ILSVRC'12 con 1000 imagenes por clase. Sin embargo, sí supera la performance de pre-entrenar la red con 20 imágenes por clase en
ILSVRC'12. No solamente eso sino que también supera la performance de la red siamesa entrenada con SFA. Esto nos indica que cuando se tiene un conjunto acotado de datos de entrenamiento, se puede lograr
entrenar un modelo que logre resultados similares al estado del arte si utilizamos redes siamesas entrenadas con automovimiento.


#### 2.4 Evaluación de las features utilizando el conjunto de datos ILSVRC'12

Para evaluar que los filtros aprendidos mediante automovimiento son buenos para tareas de clasificación, se procedió a realizar transferencia de aprendizaje en todas las capaz convolucionales (es decir, reentrenar toda la red utilizando un nuevo conjunto de datos).

Se evaluó la exactitud de una red preentrenada con automovimiento contra una entrenada con *SFA* y una con pesos inicializados aleatoriamente. Para ello se utilizó el conjunto de datos ILSVRC 2012 utilizado en la competencia de Imagenet. Dicho conjunto de datos
cuenta con 1000 clases de objetos. Para el entrenamiento de las redes se utilizaron subconjuntons de todas las clases con 1, 5, 10, 20 y 1000 elementos por cada una.  Los resultados se muestran en el siguiente cuadro. Se puede observar que los pesos de una red pre-entrenada con automovimiento (KITTI-EGO) supera en todos los casos a los pesos inicializados aleatoriamente (ALEXNET), mientras que los pesos aprendidos mediante \textit{SFA} (KITTI-SFA) presentan un rendimiento incluso peor que el de ALEXNET.

| Método    |    1 |    5 |   10 |   20 |  1000 |
|:----------|-----:|-----:|-----:|-----:|------:|
| KITTI-EGO | 0.49 | 1.27 | 2.14 | 4.13 |  20.8 |
| KITTI-SFA | 0.35 | 0.75 | 1.34 | 2.64 | 11.83 |
| ALEXNET   | 0.45 | 0.95 | 1.91 | 3.69 | 18.35 |


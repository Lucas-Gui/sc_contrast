import tensorflow as tf
import tensorflow.keras.layers as layers 

## DEEPLIFT MODEL LOADING

def make_model(X, shape=(40,40), activation='sigmoid', n_components_pca=40, n_classes=2):
    pc_mat = PCA(n_components_pca).fit(X).components_.T
    layer_pca = layers.Dense(n_components_pca, use_bias=False, activation=None)
    layer_pca.trainable=False #freezing the layer
    layer_pca(X[0:1]) #initializing the layer
    layer_pca.set_weights((pc_mat,))
    
    model = tf.keras.Sequential(
        [
        layers.InputLayer((X.shape[1])), #unknown for deeplift...
        layer_pca,
        ]
    )
    for s in shape:
        model.add(layers.Dense(s,activation=activation))
    model.add(layers.Dense(n_classes, activation='softmax')) #the choice of softmax is debatable
    opt = tf.keras.optimizers.Adam(learning_rate=1e-2)
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(opt, loss, metrics=[tf.keras.metrics.CategoricalAccuracy()])

    return model

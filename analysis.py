import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input

spreadsheet = 'clean_analise.xlsx'

df = pd.read_excel(
    spreadsheet,
    engine='openpyxl'
)

num_col = [
    'CUSTOEST',
    'QTD',
    'PRECONORMAL',
    'TOTALPRECONORMAL',
    'CUSTO',
    'VENDA',
    'TOTALPROMOCAO',
    'PART_NORMAL',
    'PART_PROMO',
    'PART_VENDA',
    'CUSTONORMAL',
    'CUSTOPROMOCIONAL',
    'MVB',
    'MVN',
    'MVP',
    'MVPTOT',
    'MVPINDIV',
    'MARGEM',
    'MARGEM_PROMOCAO',
    'MARGEM_NORMAL'
]

for col in num_col:
    if col not in df.columns:
        continue

    if df[col].dtype == 'object':
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(",", ".", regex=False)
        )

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    ).astype('float32')

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

X = df[num_col].copy()

X = X.fillna(
    X.median()
)

for col in X.columns:
    q01 = X[col].quantile(0.01)
    q99 = X[col].quantile(0.99)

    X[col] = X[col].clip(
        lower=q01,
        upper=q99
    )

scaler = RobustScaler()

X_scaled = scaler.fit_transform(
    X
)

X_scaled = np.nan_to_num(
    X_scaled,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
).astype(np.float32)

input_dim = X_scaled.shape[1]

inputs = Input(
    shape=(input_dim,),
    name="input",
    dtype="float32"
)

x = Dense(
    32,
    activation='relu'
)(inputs)

x = Dropout(
    0.10
)(x)

x = Dense(
    16,
    activation='relu'
)(x)

latent = Dense(
    8,
    activation='linear',
    name='latent_space'
)(x)

x = Dense(
    16,
    activation='relu'
)(latent)

x = Dense(
    32,
    activation='relu'
)(x)

outputs = Dense(
    input_dim,
    activation='linear'
)(x)

autoencoder = Model(
    inputs,
    outputs,
    name='Profit_Autoencoder'
)

encoder = Model(
    inputs,
    latent,
    name='Profit_Encoder'
)

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss='mse'
)

autoencoder.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=12,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

history = autoencoder.fit(
    X_scaled,
    X_scaled,
    epochs=64,
    batch_size=10,
    validation_split=0.10,
    shuffle=True,
    callbacks=[
        early_stop,
        reduce_lr
    ]
)

X_latent = encoder.predict(
    X_scaled,
    batch_size=10
)

X_latent = np.asarray(
    X_latent,
    dtype=np.float32
)

print(f"Original Format: {X_scaled.shape}")
print(f"Latent Format: {X_latent.shape}")

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(
    X_latent
)

silhouette = silhouette_score(
    X_latent,
    clusters
)

print(f"Silhouette Score: {silhouette:.4f}")

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_latent
).astype(np.float32)

X_pca_df = pd.DataFrame(
    X_pca,
    columns=["PCA1", "PCA2"]
)

X_pca_df["CLUSTER"] = clusters

df["CLUSTER"] = clusters

plt.figure(
    figsize=(12,8)
)

sns.scatterplot(
    data=X_pca_df,
    x="PCA1",
    y="PCA2",
    hue="CLUSTER",
    palette="tab10",
    alpha=0.65,
    s=25
)

plt.title("Cluster de Produtos - Espaço Latente")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend(title="Cluster")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("cluster_produtos.png", dpi=200)
plt.show()

df.to_excel(
    "resultado_clusters.xlsx",
    index=False
)

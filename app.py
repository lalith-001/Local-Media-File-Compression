import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import requests
import os



def residual_block(x, filters):
    r = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    r = layers.Conv2D(filters, 3, padding="same")(r)
    return layers.Add()([x, r])


class ConvAutoencoder(keras.Model):

    def __init__(self, latent_dim=256):
        super().__init__()

        enc_in = keras.Input(shape=(128,128,3))

        x = layers.Conv2D(64,3,strides=2,padding="same",activation="relu")(enc_in)
        s1 = x

        x = layers.Conv2D(128,3,strides=2,padding="same",activation="relu")(x)
        s2 = x

        x = layers.Conv2D(256,3,strides=2,padding="same",activation="relu")(x)
        s3 = x

        x = layers.Conv2D(512,3,strides=2,padding="same",activation="relu")(x)

        x = residual_block(x,512)
        x = residual_block(x,512)

        x = layers.Flatten()(x)
        enc_out = layers.Dense(latent_dim)(x)

        self.encoder = keras.Model(enc_in,[enc_out,s1,s2,s3])

        lat_in = keras.Input(shape=(latent_dim,))
        s1_in = keras.Input(shape=(64,64,64))
        s2_in = keras.Input(shape=(32,32,128))
        s3_in = keras.Input(shape=(16,16,256))

        x = layers.Dense(8*8*512,activation="relu")(lat_in)
        x = layers.Reshape((8,8,512))(x)

        x = residual_block(x,512)

        x = layers.Conv2DTranspose(256,3,strides=2,padding="same",activation="relu")(x)
        x = layers.Add()([x,s3_in])
        x = layers.Conv2D(256,3,padding="same",activation="relu")(x)

        x = layers.Conv2DTranspose(128,3,strides=2,padding="same",activation="relu")(x)
        x = layers.Add()([x,s2_in])
        x = layers.Conv2D(128,3,padding="same",activation="relu")(x)

        x = layers.Conv2DTranspose(64,3,strides=2,padding="same",activation="relu")(x)
        x = layers.Add()([x,s1_in])
        x = layers.Conv2D(64,3,padding="same",activation="relu")(x)

        x = layers.Conv2DTranspose(32,3,strides=2,padding="same",activation="relu")(x)

        dec_out = layers.Conv2D(3,3,padding="same",activation="sigmoid")(x)

        self.decoder = keras.Model([lat_in,s1_in,s2_in,s3_in],dec_out)

    def call(self,x):
        encoded,s1,s2,s3 = self.encoder(x)
        return self.decoder([encoded,s1,s2,s3])



@st.cache_resource
def load_model():


    model = ConvAutoencoder(latent_dim=1536)
    dummy = tf.zeros((1,128,128,3))
    model(dummy)

    model.load_weights("model/ae_ld1536_fp16.weights.h5")

    return model


model = load_model()



st.title("Image Compression")

uploaded = st.file_uploader("Upload Image",type=["png","jpg","jpeg"])

if uploaded:

    original = Image.open(uploaded).convert("RGB")

    display_original = original.resize((512,512))

    image = original.resize((128,128))

    img = np.array(image)/255.0
    img = np.expand_dims(img,0)

    reconstructed = model(img,training=False)[0].numpy()


    reconstructed_uint8 = (reconstructed*255).astype(np.uint8)

    display_reconstructed = Image.fromarray(reconstructed_uint8).resize((512,512), Image.BICUBIC)

    col1,col2 = st.columns(2)

    with col1:
        st.subheader("Original (scaled)")
        st.image(display_original)

    with col2:
        st.subheader("Reconstructed (scaled)")
        st.image(display_reconstructed)


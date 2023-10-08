#importation nécessiares
import streamlit as st
from PIL import Image
import numpy as np
import joblib
from skimage import color, transform, feature
import matplotlib.pyplot as plt

# Charger le modèle pré-entrainé
best_model = joblib.load("rf_KhaoulaAmira_model.sav")
# Titre de l'application Streamlit
st.title("Application of Face Detection")
st.markdown("Download an image")

# Sélection de l'image téléchargée par l'utilisateur
uploaded_image = st.file_uploader("Choose an image", type=["png","jpg", "jpeg"])

# Fonction pour générer des fenêtres glissantes (sliding windows) sur une image
def sliding_window(img, patch_size=(60, 45), istep=2, jstep=2, scale=1.0):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Ni, jstep):
            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = transform.resize(patch, patch_size)
            yield (i, j), patch
# Fonction pour détecter les visages dans une image
def detect_face(image, indices, labels):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    Ni, Nj = (42,67)
    indices = np.array(indices)
    for i, j in indices[labels == 1]:
        ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='green', alpha=0.3, lw=2, facecolor='none'))
    return fig

if uploaded_image is not None:
    # Charger l'image téléchargée et la prétraiter
    img = Image.open(uploaded_image)
    img = np.array(img)
    gray_img = color.rgb2gray(img)
    resized_img = transform.rescale(gray_img, 0.5)
    cropped_img = resized_img
     # Afficher l'image téléchargée
    st.image(cropped_img, caption="Image downloaded", use_column_width=True)
    
    # Bouton pour détecter les visages
    if st.button("Detect faces"):
        indices, patches = zip(*sliding_window(cropped_img))
        patches_hog = np.array([feature.hog(patch) for patch in patches])
        labels = best_model.predict(patches_hog)
        fig = detect_face(cropped_img, indices, labels)
        st.pyplot(fig)

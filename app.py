import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# import efficientnet.keras as efn
import streamlit as st
import SessionState
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.transform import resize

import reportgenerator
import style
from keras.models import Model, load_model
st.set_option('deprecation.showPyplotGlobalUse', False)

model = load_model('model.h5')


st.markdown(
    f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {1000}px;
        padding-top: {5}rem;
        padding-right: {0}rem;
        padding-left: {0}rem;
        padding-bottom: {0}rem;
    }}
    .reportview-container .main {{     

    }}   
    [data-testid="stImage"] img {{
        margin: 0 auto;
        max-width: 500px;
    }}
</style>
""",
    unsafe_allow_html=True,
)

# main panel
# logo = Image.open('dss_logo.png')
# st.image(logo, width=None)
style.display_app_header(main_txt='Packages Defective and non Defective Differentiator Systems',
                         sub_txt='An artificial intelligence system for packages defect and non defect determination', is_sidebar=False)

# session state
ss = SessionState.get(page='home', run_model=False)


st.markdown('**Upload package image to analyze**')
st.write('')
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg'])



if uploaded_file is not None:
    # uploaded_file.read()
    image = Image.open(uploaded_file)
    st.image(image, caption='Package image', use_column_width=True)
    im_resized = image.resize((224, 224))
    im_resized = resize(np.asarray(im_resized), (224, 224, 3))
    grayy = resize(np.asarray(im_resized), (224, 224, 1))

    # grid section
    col1, col2, col3 = st.columns(3)
    col1.header('Resized Image')
    col1.image(im_resized, caption='Resized image', use_column_width=False)
    with col2:
        st.header('Gray Image')
        gray_image = rgb2gray(im_resized)
        st.image(gray_image, caption='preprocessed image',
                 use_column_width=False)

    with col3:
        st.header('Denoised Image')
        # sigma = float(sys.argv[2])
        gray_image = rgb2gray(im_resized)
        blur = gaussian(gray_image, sigma=1.5)
        # perform adaptive thresholding
        t = threshold_otsu(blur)
        mask = blur > t
        sel = np.zeros_like(im_resized)
        sel[mask] = im_resized[mask]
        st.image(sel, caption='preprocessed image', use_column_width=False)
    print(grayy.shape)
    pred_proba = model.predict(np.expand_dims(grayy, 0))
    pred_class = model.predict_classes(np.expand_dims(grayy, 0))
    data = list((pred_proba[0]*100).round(2))
    data.append((100-data[0]).round(2))
    data = [data[0],data[1]]
    label = [str(pred_class[0][0]),str(1 - pred_class[0][0])]
    print(label)
    isup_colors = ['lightskyblue','lightcoral']


    col1, col2, = st.columns(2)
    with col1:
        reportgenerator.pieChart(data, label=label, colors=isup_colors,
                                 title='Prediction Distribution', startangle=45)
    with col2:
        reportgenerator.visualize_confidence_level(data, label=label, ylabel='Package Box Classes',
                                                   title='Prediction Bar Chat')
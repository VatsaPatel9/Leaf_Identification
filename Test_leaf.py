#This is used to test the model from random images taken from google 

#!/usr/bin/env python
# coding: utf-8

# In[11]:


import h5py
import os 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

categories = ["acer_campestre", "betula_nigra", "carya_cordiformis", "eucommia_ulmoides", "fraxinus_americana", "gleditsia_triacanthos",
             "juglans_nigra","liriodendron_tulipifera"]
print(len(categories))

os.chdir(r"path")
testing = os.getcwd() #returns current working directory 
print(testing)

# test_dir = os.path.join(testing, 'test_leaf')

print('total test image :', len(os.listdir(testing))) 


test_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_generator = test_datagen.flow_from_directory(testing,class_mode='binary',target_size=(128, 128))


print(test_generator)

new_model = tf.keras.models.load_model('leaf_dl_vgg16_aug_mine.h5') #load the model

predictions = new_model.predict(test_generator)

a=predictions[0]
print(a)
index_max = np.argmax(a)
print(index_max)
print(categories[index_max])


# In[12]:


sample_training_images, _ = next(test_generator)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 2, figsize=(5,5))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    

plotImages(sample_training_images[:])

thisdict = {0:"Name : Acer \nFamily: Campestre \nType: Deciduous \nEdible uses: none \nMaterial uses: Used for packing fruits \nMedicinal uses: none \nEnvironmental Tolerances: Strong Wind \nLife Cycle: Perennial", 
             1:"Name : Betula nigra \nFamily : Betulaceae \nType: Deciduous \nEdible uses: Sap - raw or cooked \nMaterial uses: sometimes used for furniture \nMedicinal uses: Chewed, or used as an infusion,  treatment of dysentery \nEnvironmental Tolerances: sensitive \nLife Cycle: Perennial",
              2:"Name : Carya cordiformis \nFamily : Juglandaceae \nType: Deciduous \nEdible uses: raw or cooked \nMaterial uses: Used as an illuminant in oil lamps \nMedicinal uses: Treatment of rheumatism \nEnvironmental Tolerances: sensitive \nLife Cycle: Perennial",
               3: "Name : Eucommia ulmoides \nFamily : Eucommiaceae \nType: Deciduous \nEdible uses: Young leaves \nMaterial uses: The leaves contain 3% dry weight of gutta-percha, a non-elastic rubber, used for insulation of electrical wires \nMedicinal uses: Treatment of reduction in blood pressure \nEnvironmental Tolerances: sensitive \nLife Cycle: Perennial",
               4: "Name : Fraxinus american \nFamily : Fraxinus \nType: Deciduous \nEdible uses: Yes \nMaterial uses: none \nMedicinal uses: Decoction of the leaves as a laxative and general tonic for women after childbirth \nEnvironmental Tolerances: sensitive \nLife Cycle: Perennial",
               5: "Name : Gleditsia triacanthos \nFamily : Leguminosae \nType: Deciduous \nEdible uses: none, The plant contains potentially toxic compounds \nMaterial uses: none \nMedicinal uses: The juice of the pods is antiseptic \nEnvironmental Tolerances: Salinity and Drought \nLife Cycle: Perennial",
               6: "Name : Juglans nigra \nFamily : Juglandaceae \nType: Deciduous \nEdible uses: none, The plant has occasionally been known to cause contact dermatitis in humans; toxic \nMaterial uses: none \nMedicinal uses: none \nEnvironmental Tolerances: sensitive \nLife Cycle: Perennial",
               7: "Name : Liriodendron tulipifera \nFamily : Magnoliaceae \nType: Deciduous \nEdible uses: none \nMaterial uses: none \nMedicinal uses: A gold-coloured dye is obtained from it \nEnvironmental Tolerances: sensitive \nLife Cycle: Perennial",}
print(thisdict[index_max])


# In[ ]:





###############################################################
# AutoWiki                                                    #
# Created by: Eszes Balint (ebalint96) 2018                   #
###############################################################

from keras.models import load_model
from sklearn.externals import joblib
import sys

#Loading BoW and encoder
vectorizer = joblib.load('vectorizer.pkl')
encoder = load_model('encoder.h5')

import matplotlib
import matplotlib.pyplot as plt
import mpld3

xs = []
ys = []
labels = []

for i in range(5000):
	file_id = i + 2000
	
	file = open("title_{0}".format(file_id),'r',encoding='utf8')
	labels.append(file.read())

	v = encoder.predict(vectorizer.transform(["text_{0}".format(file_id)]).toarray())
	xs.append(v[0][0])
	ys.append(v[0][1])

fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))

scatter = ax.scatter(xs,ys,s=5)
ax.grid(color='white', linestyle='solid')

tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
mpld3.plugins.connect(fig, tooltip)

mpld3.save_html(fig,'wiki.html')


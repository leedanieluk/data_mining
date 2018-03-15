import os
import html2text
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS

def tokenize_and_stem(document):
    tokens = word_tokenize(document)
    tokens = [token for token in tokens if len(token) > 2]
    stemmer = SnowballStemmer('english')
    stems = [stemmer.stem(t) for t in tokens]
    return stems

def tokenize_only(document):
    tokens = word_tokenize(document)
    tokens = [token for token in tokens if len(token) > 2]
    return tokens

rootdir = './data'

words_freq = {}
documents = []
documents_titles = []
counter = 0
labels = []

for subdir, dirs, files in os.walk(rootdir):
    counter += 1
    # if counter > 5:
    #     continue

    # skip root dir
    if subdir == './data':
        continue

    # for text n
    print("For text: %s" % subdir)
    
    documents_titles.append(str(subdir))
    document = []
    labels.append(str(subdir))
    # do page analysis
    for file in files:

        # for page n
        page_html = open(str(os.path.join(subdir, file)))
        # print(str(os.path.join(subdir, file)))
        # print("For text: %s" % str(os.path.join(subdir, file)))

        # 1. html to string
        page_str = html2text.html2text(page_html.read()) # html to string
        # print(page_str)

        # 2. remove everything except for letters a-z, A-Z
        # letters_list = set('abcdefghijklmnopqrstuvwxy ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        # page_only_words = ''.join(filter(letters_list.__contains__, page_str))
        page_only_words = re.sub('[^A-Z^a-z]+', ' ', page_str)

        # 3. attach page to document
        document = "%s %s" % (document, page_only_words)

    documents.append(document)
    
# tokenized and stemmed
totalvocab_stemmed = []
totalvocab_tokenized = []

for i in documents:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')



# define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                min_df=0.1, stop_words='english',
                                use_idf=True, tokenizer=tokenize_and_stem)

tfidf_matrix = tfidf_vectorizer.fit_transform(documents) #fit the vectorizer to synopses
terms = tfidf_vectorizer.get_feature_names()

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)
print(dist)

# cluster
num_clusters = 5

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()
joblib.dump(km,  'doc_cluster.pkl')

km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

the_docs = { 'title': documents_titles, 'document': documents, 'cluster': clusters}

frame = pd.DataFrame(the_docs, index = [clusters] , columns = ['title', 'cluster'])

print(frame.head())

print("Top terms per cluster:")
print('\n')

#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print('\n')
    print('\n')
    
    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print('\n')
    print('\n')
print('\n')
print('\n')



MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)


plt.scatter(pos[:, 0], pos[:, 1], color='turquoise',label='MDS')

for label, x, y in zip(labels, pos[:, 0], pos[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

plt.show()
xs, ys = pos[:, 0], pos[:, 1]
print(xs)
print(ys)
print()
print()

# from scipy.cluster.hierarchy import ward, dendrogram

# linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

# fig, ax = plt.subplots(figsize=(15, 20)) # set size
# ax = dendrogram(linkage_matrix, orientation="right", labels=documents_titles)

# plt.tick_params(\
#     axis= 'x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='off',      # ticks along the bottom edge are off
#     top='off',         # ticks along the top edge are off
#     labelbottom='off')

# plt.tight_layout() #show plot with tight layout

# #uncomment below to save figure
# plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters




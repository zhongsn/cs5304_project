import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import pickle


def preprocess(path):
	df = pd.read_csv(path)
	print(df.shape)
	print(df.columns)

	movies = df[['MovieID', 'Title', 'Genre']]
	print(movies.shape)
	print(movies.columns)

	movies = movies.drop_duplicates(subset ='MovieID') 
	print(movies.shape)
	print(movies.columns)

	vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
	genre_vectors = vectorizer.fit_transform(movies['Genre'])

	print(vectorizer.get_feature_names())
	print(genre_vectors.shape)

	# movies.to_csv('movies.csv')

	return movies, genre_vectors

def cos_similarity(vector_matrix):
	cos_similarity_matrix = linear_kernel(vector_matrix, vector_matrix)
	print(cos_similarity_matrix.shape)
	# output_file = open('cos_similarity_matrix', 'w')
	# output_file.write(str(cos_similarity_matrix))
	return cos_similarity_matrix



if __name__ == '__main__':
	path = '../M.csv'
	movies, genre_vectors = preprocess(path)
	cos_similarity_matrix = cos_similarity(genre_vectors)

	output = {'movies': movies, 'cos_similarity_matrix': cos_similarity_matrix}
	with open('model.pkl', 'wb') as outfile:
		pickle.dump(output, outfile)



import pandas as pd
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))
movies = model['movies']
cos_similarity_matrix = model['cos_similarity_matrix']
title_to_index_dict = dict(zip(movies['Title'], movies.index))

print(movies.shape)
print(cos_similarity_matrix.shape)

def recommend_from_one_movie(title):
	# Extract top 20 similar movies except the input movie itself
	top20 = np.argsort(np.array(cos_similarity_matrix[title_to_index_dict[title]]))[-21:-1][::-1]
	movies.iloc[top20].to_csv('test_ouput_for_one_movie.csv')


def recommend_for_one_user(test_data_path):
	test_df = pd.read_csv(test_data_path)
	# recommend according to the highest rated movies. Would add more algorithm later
	high_rated_movies = test_df[test_df['Rating'] == 5]

	scores = np.zeros(movies.shape[0])
	for title in high_rated_movies['Title']:
		scores += cos_similarity_matrix[title_to_index_dict[title]]

	indices = [title_to_index_dict[title] for title in test_df['Title']]
	np.put(scores, indices, 0)
	top20 = np.argsort(scores)[-20:][::-1]
	movies.iloc[top20].to_csv('test_ouput_for_one_user.csv')

recommend_from_one_movie('Toy Story (1995)')
recommend_for_one_user('test.csv')

import pandas as pd
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))
movies = model['movies']
cos_similarity_matrix = model['cos_similarity_matrix']
title_to_index_dict = dict(zip(movies['Title'], range(len(movies.index))))

print(movies.shape)
print(cos_similarity_matrix.shape)

def recommend_from_one_movie(title):
	# Extract top 20 similar movies except the input movie itself
	top20 = np.argsort(np.array(cos_similarity_matrix[title_to_index_dict[title]]))[-21:-1][::-1]
	movies.iloc[top20].to_csv('test_output_for_one_movie.csv')


def recommend_for_one_user(test_data_path):
	test_df = pd.read_csv(test_data_path)

	# if the user has rated enough movies, extract the last 10 to verify
	verify_df = None
	if len(test_df.index) > 300:
		verify_df = test_df.iloc[-200:]
		test_df = test_df.iloc[:-200]

	# recommend according to the highest/lowest rated movies. Would add more algorithm later
	high_rated_movies = test_df[test_df['Rating'] == 5]
	low_rated_movies = test_df[test_df['Rating'] == 1]
	# high_rated_movies = test_df[test_df['Rating'] >= 4]
	# low_rated_movies = test_df[test_df['Rating'] <= 2]

	scores_for_high_rated = np.zeros(movies.shape[0])
	for _, movie in high_rated_movies.iterrows():
		title = movie['Title']
		weight = 1 if movie['Rating'] == 5 else 0.3
		scores_for_high_rated += cos_similarity_matrix[title_to_index_dict[title]] * weight
	print('\nstart')
	scores_for_high_rated /= len(high_rated_movies.index)

	scores_for_low_rated = np.zeros(movies.shape[0])
	for _, movie in low_rated_movies.iterrows():
		title = movie['Title']
		weight = 1 if movie['Rating'] == 1 else 0.3
		scores_for_low_rated -= cos_similarity_matrix[title_to_index_dict[title]] * weight
	scores_for_low_rated /= len(low_rated_movies.index)

	scores = scores_for_high_rated + scores_for_low_rated


	# we don't want to recommend those movies that the user has rated
	indices = [title_to_index_dict[title] for title in test_df['Title']]
	np.put(scores, indices, np.NINF)

	top20 = np.argsort(scores)[-20:][::-1]
	movies.iloc[top20].to_csv('test_output_for_one_user.csv')

	if verify_df is not None:
		indices = [title_to_index_dict[title] for title in verify_df['Title']]
		verify_df['Score'] = [scores[i] for i in indices]
		verify_df.sort_values('Score', 0,  ascending=False, inplace=True)
		verify_df.to_csv('verify_scores_for_one_user.csv')

recommend_from_one_movie('Toy Story (1995)')
recommend_for_one_user('test.csv')

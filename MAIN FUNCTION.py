# Import all required libraries
import pandas as pd
from numpy import *
import numpy as np
import datetime as dt
from sklearn.cluster import KMeans
import operator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from random import randint

# Convert a given continuous vector into a feature vector
def feature_cont(dataset, column_name):
    # Null treatment
    value = dataset[str(column_name)].isnull().sum()
    if(value > 0 and (type(dataset[column_name][0])!=str)):
        dataset[column_name] = dataset[column_name].replace(np.nan, 0)
    # Normalize all the values in a given continuous vector
    vec = np.array([dataset[str(column_name)]/max(dataset[str(column_name)])])
    return (vec.T)

# Convert a given categorical/ordinal vector into a feature vector
def feature_categ(dataset, column_name):
    # Null treatment
    value = dataset[str(column_name)].isnull().sum()
    if(value > 0 and (type(dataset[column_name][0])!=str)):
        dataset[column_name] = dataset[column_name].replace(np.nan, 0)
    elif(value > 0 and (type(dataset[column_name][0])==str)):
        dataset[column_name] = dataset[column_name].replace(np.nan, 'NA')
    # Split the given column into multiple columns based on the number of categories
    data = list(dataset[str(column_name)].unique())
    no_of_rows = size(dataset[str(column_name)])
    no_of_categ = size(dataset[str(column_name)].unique())
    # Create a matrix of (total number of entries x total number of categories)
    mat = np.zeros(shape=(no_of_rows, no_of_categ))
    for i in range(0, no_of_rows):
        index_val = data.index(dataset[str(column_name)][i])
        mat[i][index_val] = 1
    return mat

# Convert into datetime format and get a year column
def date_convert(dataset, column_name):
    # Set a default date for NA's
    dataset[str(column_name)] = dataset[str(column_name)].replace(np.nan, str("1900-11-11 11:11:11"))
    # Convert to pandas series format
    dataset[str(column_name)] = [dt.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").date() for date in dataset[str(column_name)]]
    # Conver to datetime format
    vec = (pd.to_datetime(dataset[column_name])).dt.year
    # create a column called year
    nw_yr = list(vec)
    # Ordinal variables
    yrs = list(sort(vec.unique()))
    # Setting matrix sizes
    no_of_rows = size(dataset[str(column_name)])
    no_of_categ = size(yrs)
    mat = np.zeros(shape=(no_of_rows, no_of_categ))
    for i in range(0, no_of_rows):
        index_val = yrs.index(vec[i])
        mat[i][index_val] = 1
    return mat

def feature_extraction(dataset_name, column_list):
    # Iterate through the column list and add the features one by one
    n = size(column_list)
    dates_list = ['al_date_created', 'ar_date_created', 'tr_date_created']
    contin_list = ['al_listens', 'al_tracks','tr_duration', 'tr_favorites','tr_interest','tr_listens']
    categ_list = ['al_type' ,'tr_genre_top']
    mat = []
    # Create features
    for i in range(0,n):
        # For datetime variable
        if(column_list[i] in dates_list):
            mat.append(date_convert(dataset_name, column_list[i]))
        # For continuous variable
        elif(column_list[i] in contin_list):
            mat.append(feature_cont(dataset_name, column_list[i]))
        # For categorical/ordinal variable
        elif(column_list[i] in categ_list):
            mat.append(feature_categ(dataset_name, column_list[i]))
    # Create the final feature matrix
    fin_mat = np.concatenate(tuple(mat), axis = 1)
    return fin_mat

# Function to print out the list of songs given by the random generator
def input_playlist(dataset, inp_list):
    output_val = ['Playlist 1', 'Playlist 2', 'Playlist 3']
    count = 0
    j = 0
    # Find track names based on randomly generator
    for i in inp_list:
        if(count%10 == 0):
            if (count!= 0):
                print("+------------------------------------------------------------+")
            print("+------------------------",output_val[j],"------------------------+")
            print("\n")
            j += 1
        print(dataset['tr_title'][i], "\n")
        count += 1
    print("+------------------------------------------------------------+")

# Function to get the input data and classify them into genres (with priority)
def genre_group(dataset, songs_list, feature_matrix):
    # All genres present
    genres = ['Hip-Hop', 'Pop', 'NA', 'Rock', 'Experimental', 'Folk', 'Jazz', 'Electronic',
              'Spoken', 'International', 'Blues', 'Country', 'Old-Time / Historic',
              'Soul-RnB', 'Classical', 'Instrumental', 'Easy Listening']
    genre_dict = {}
    inter_dict = {}
    final_dict = {}
    no_of_rows = size(feature_matrix[0])
    # Set the count for each genre equal to 0
    for i in genres:
        genre_dict[i] = 0
        inter_dict[i] = np.zeros(shape=(1,no_of_rows))
    # Get genre count for all the songs present in the list
    for i in songs_list:
        genre_dict[dataset['tr_genre_top'][i]] += 1
        inter_dict[dataset['tr_genre_top'][i]] += np.array([feature_matrix[i]])
    # Select the genre that is present in the songs list
    mat = dict(sorted(genre_dict.items(), key=operator.itemgetter(1), reverse=True))
    mat = dict((k, v) for k, v in mat.items() if v!=0)
    # Normalize the feature vectors
    for key, value in mat.items():
        final_dict[key] = inter_dict[key]/mat[key]
    return final_dict

# Get the custom playlist list to output to the user
def custom_playlist(kNN_classifier, usr_inpt, dataset, k):
    nrst_songs_list = {}
    i = 0
    # Convert all the track id's to track names
    for key, value in usr_inpt.items():
        temp = []
        sng_lst = kNN_classifier.kneighbors(usr_inpt[key])[1][0]
        for j in range(0,k):
            temp.append(dataset['tr_title'][sng_lst[j]])
        nrst_songs_list[i] = temp
        i += 1
    return nrst_songs_list

# Function to select the relevant songs based on priority
def final_list(nearest_songs, dataset, k):
    dict_len = len(nearest_songs)
    songs_list = []
    count = 0
    # Select the nearest songs based on the genre present
    # If only one genre is present
    if(dict_len == 1):
        songs_list += nearest_songs[0]
    # If more than one genre is present
    else:
        for i in range(0, k):
            for j in range(0, dict_len):
                songs_list.append(nearest_songs[j][i])
                count += 1
                if(count == 10):
                    return songs_list
    return songs_list

def mainCall():
    # Import the file
    #path = '/Users/tst/Desktop/flask_project'
    filename = 'PythonExport.csv'
    inp_file = pd.read_csv(filename)
    print("The file has been imported.....\n")
    # Remove the extra column
    inp_file = inp_file.drop("Unnamed: 0", axis = 1)
    # Remove the irrelevant columns
    inp_file = inp_file.drop(["tr_id", "al_id", "ar_id", "tr_number", "tr_tags", "subset"], axis =  1)
    # Drop column named 'split'
    all_data = inp_file.drop("split", axis = 1)
    print("Uneccessary columns have been dropped.....\n")
    # Split the datasets with selected columns
    all_data = all_data[['al_date_created', 'al_listens', 'al_tracks', 'al_type', 'ar_date_created',
                         'tr_date_created', 'tr_duration','tr_favorites', 'tr_genre_top', 'tr_genres',
                         'tr_genres_all', 'tr_interest', 'tr_listens']]
    # Create a table with only track name and genre
    tracks = inp_file[['tr_title', 'tr_genre_top']]
    # Treat the nulls
    tracks['tr_title'] = tracks['tr_title'].replace(np.nan, 'NA')
    tracks['tr_genre_top'] = tracks['tr_genre_top'].replace(np.nan, 'NA')
    col_names = all_data.columns
    print("Created a database consisting of only track names and its corresponding genre....\n")
    # Call feature extraction function
    feature_matrix = feature_extraction(all_data, col_names)
    print("Created the feature matrix.....\n")
    print("Data is ready for clustering....\n")
    # Elbow test
    # Elbow point to select the optimal value of k in k-means clustering
    print("Performing the elbow point test to find the optimal value of k to be chosen to perform k-means clustering....\n")
    sum_of_squared_dist = []
    for k in range(1,20):
        km = KMeans(n_clusters=k)
        km = km.fit(feature_matrix)
        sum_of_squared_dist.append(km.inertia_)
    print("Elbow point test is complete.....\n")
    print("Ready to run k-means clustering.....\n")
    print("Running k-means clustering.....\n")
    # Number of clusters
    kmeans = KMeans(n_clusters=10)
    # Fitting the input data
    kmeans = kmeans.fit(feature_matrix)
    # Getting the cluster labels
    labels = kmeans.predict(feature_matrix)
    print("k-means clustering is complete....\n")
    # Centroid values
    centroids = kmeans.cluster_centers_
    print("Ready to run k-Nearest Neighbor.....\n")
    # knn
    k = 5
    print("Training k-Nearest Neighbor model....\n")
    kNN_classifier = KNeighborsClassifier(n_neighbors=k)
    kNN_classifier.fit(feature_matrix, labels)
    y_pred = kNN_classifier.predict(feature_matrix)
    print("Training the k-NN model is complete.....\n")
    # User input
    print("Generating 30 random samples (songs)......\n")
    value = [randint(0, 106574) for p in range(0, 30)]
    #value = listArray
    print(value)
    print("The three playlists created randomly are : \n")
    input_playlist(tracks, value)
    print("Identifying and grouping genres of the songs given by user.....\n")
    user_input = genre_group(tracks, value, feature_matrix)
    print("Finding songs that are bound to be liked by the user.....\n")
    all_nearest_songs = custom_playlist(kNN_classifier, user_input, tracks, k)
    print("Identified songs that closely resemble songs liked by the user.....\n")
    final_output = final_list(all_nearest_songs, tracks, k)
    print("Your customized playlist is : \n")
    print("+------------------------------------------------------------+")
    for i in final_output:
        print(i, "\n")
    print("+------------------------------------------------------------+")
    print("Enjoy!!")
    return final_output


# Main function
if __name__ == "__main__":
    # Import the file
    #path = '/Users/tst/Desktop/flask_project/'
    filename = 'PythonExport.csv'
    inp_file = pd.read_csv(filename)
    print("The file has been imported.....\n")
    # Remove the extra column
    inp_file = inp_file.drop("Unnamed: 0", axis = 1)
    # Remove the irrelevant columns
    inp_file = inp_file.drop(["tr_id", "al_id", "ar_id", "tr_number", "tr_tags", "subset"], axis =  1)
    # Drop column named 'split'
    all_data = inp_file.drop("split", axis = 1)
    print("Uneccessary columns have been dropped.....\n")
    # Split the datasets with selected columns
    all_data = all_data[['al_date_created', 'al_listens', 'al_tracks', 'al_type', 'ar_date_created',
                         'tr_date_created', 'tr_duration','tr_favorites', 'tr_genre_top', 'tr_genres',
                         'tr_genres_all', 'tr_interest', 'tr_listens']]
    # Create a table with only track name and genre
    tracks = inp_file[['tr_title', 'tr_genre_top']]
    # Treat the nulls
    tracks['tr_title'] = tracks['tr_title'].replace(np.nan, 'NA')
    tracks['tr_genre_top'] = tracks['tr_genre_top'].replace(np.nan, 'NA')
    col_names = all_data.columns
    print("Created a database consisting of only track names and its corresponding genre....\n")
    # Call feature extraction function
    feature_matrix = feature_extraction(all_data, col_names)
    print("Created the feature matrix.....\n")
    print("Data is ready for clustering....\n")
    # Elbow test
    # Elbow point to select the optimal value of k in k-means clustering
    print("Performing the elbow point test to find the optimal value of k to be chosen to perform k-means clustering....\n")
    sum_of_squared_dist = []
    for k in range(1,20):
        km = KMeans(n_clusters=k)
        km = km.fit(feature_matrix)
        sum_of_squared_dist.append(km.inertia_)
    print("Elbow point test is complete.....\n")
    print("Ready to run k-means clustering.....\n")
    print("Running k-means clustering.....\n")
    # Number of clusters
    kmeans = KMeans(n_clusters=10)
    # Fitting the input data
    kmeans = kmeans.fit(feature_matrix)
    # Getting the cluster labels
    labels = kmeans.predict(feature_matrix)
    print("k-means clustering is complete....\n")
    # Centroid values
    centroids = kmeans.cluster_centers_
    print("Ready to run k-Nearest Neighbor.....\n")
    # knn
    k = 5
    print("Training k-Nearest Neighbor model....\n")
    kNN_classifier = KNeighborsClassifier(n_neighbors=k)
    kNN_classifier.fit(feature_matrix, labels)
    y_pred = kNN_classifier.predict(feature_matrix)
    print("Training the k-NN model is complete.....\n")
    # User input
    print("Generating 30 random samples (songs)......\n")
    value = [randint(0, 106574) for p in range(0, 30)]
    #value = listArray
    print("The three playlists created randomly are : \n")
    input_playlist(tracks, value)
    print("Identifying and grouping genres of the songs given by user.....\n")
    user_input = genre_group(tracks, value, feature_matrix)
    print("Finding songs that are bound to be liked by the user.....\n")
    all_nearest_songs = custom_playlist(user_input, tracks, k)
    print("Identified songs that closely resemble songs liked by the user.....\n")
    final_output = final_list(all_nearest_songs, tracks, k)
    print("Your customized playlist is : \n")
    print("+------------------------------------------------------------+")
    for i in final_output:
        print(i, "\n")
    print("+------------------------------------------------------------+")
    print("Enjoy!!")


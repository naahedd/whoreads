import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


my_books = pd.read_csv("liked_books.csv", index_col=0)
my_books["book_id"] = my_books["book_id"].astype(str)

csv_book_mapping = {}
with open("book_id_map.csv", 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        csv_id, book_id = line.strip().split(",")
        csv_book_mapping[csv_id] = book_id

book_set = set(my_books["book_id"])
overlap_users = {} #{key: user_id, value: # times the user have read a book we liked}
with open("goodreads_interactions.csv") as f:
    while True:
        line = f.readline()
        if not line:
            break
    user_id, csv_id, _, rating, _ = line.strip().split(",")
    book_id = csv_book_mapping.get(csv_id)
    if book_id in book_set:
        if user_id not in overlap_users:
            overlap_users[user_id] = 1
        else:
            overlap_users[user_id] += 1 #keeping a count of how many times a given user that overlaped with the books we had


filtered_overlap_users = set([k for k in overlap_users if overlap_users[k] > my_books.shape[0]/5]) #users that read 20% of the same books as us 

interactions_list = []

with open("goodreads_interactions.csv") as f:
    while True:
        line = f.readline()
        if not line:
            break
        user_id, csv_id, _, rating, _ = line.strip().split(",")
        if user_id in filtered_overlap_users:
            books_id = csv_book_mapping[csv_id]
            interactions_list.append([user_id, book_id, rating])


#user / book matrix : rows:users cols:books cells:ratings

interactions = pd.DataFrame(interactions_list, columns=["user_id", "book_id", "rating"])

# add our own ratings into the matrix

interactions = pd.concat([my_books[["user_id", "book_id", "rating"]], interactions])

interactions["book_id"] = interactions["book_id"].astype(str)
interactions["user_id"] = interactions["user_id"].astype(str)
interactions["rating"] = pd.to_numeric(interactions["rating"])

interactions["user_index"] = interactions["user_id"].astype("category").cat.codes #id to position ##numbers that correspond to the long user id
interactions["book_index"] = interactions["book_id"].astype("category").cat.codes

ratings_mat_coo = coo_matrix((interactions["rating"], (interactions["user_index"], interactions["book_index"])))

ratings_mat = ratings_mat_coo.tocsr()


my_index = 0

#users with similar taste in books - cosine similarity

similarity = cosine_similarity(ratings_mat[my_index,:], ratings_mat).flatten()
indices = np.argpartition(similarity, -15)[15:] #15 users with most similar taste 

similar_users = interactions[interactions["user_index"].isin(indices)].copy()
similar_users = similar_users[similar_users["user_id"] != "-1"]

#create book recommendations
book_recs = similar_users.groupby("book_id").rating.agg(['count', 'mean']) #count of each book appearance and find mean rating
books_titles = pd.read_json("books_titles.json")
books_titles["book_id"] = books_titles["book_id"].astype(str)
books_recs = book_recs.merge(books_titles, how="inner", on="book_id")

#which recommendations are most relevant
book_recs["adjusted_count"] = book_recs["count"] * (book_recs["count"] / book_recs["ratings"])
book_recs["score"] = book_recs["mean"] * book_recs["adjusted_count"]

book_recs = book_recs[~book_recs["book_id"].isin(my_books["book_id"])] #take out any books where book id matches an id of a book we have already read
my_books["mod_title"] = my_books["title"].str.replace("^a-zA-Z0-9", "", regex=True).str.lower()
my_books["mod_title"] = my_books["mod_title"].str.replace("\s+", "", regex=True)
book_recs = book_recs[~book_recs["mod_title"].isin(my_books["mod_title"])]
book_recs = book_recs[book_recs["count"]>2]
book_recs = book_recs[book_recs["mean" > 2.5]]

top_recs = book_recs.sort_values("score", ascending=False)

def make_clickable(val):
    return '<a target=_blank href="{}">Goodreads</a>'.format(val)

def show_image(val):
    return '<img src="{}" width=50></img>'

top_recs.style.format({'url':make_clickable, 'cover_image': show_image})














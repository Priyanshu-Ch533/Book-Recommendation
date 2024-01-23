# Project3-Recommendationsystem
Book Recommendation System
pip install  numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn
# Importing Libraries
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings; warnings.simplefilter('ignore')
# Importing Books Data Set
book_data = pd.read_csv("Books.csv")


# Displaying First 5 rows and colums data in the books.csv data set
book_data.head()
book_data.iloc[237]['Image-URL-L']
# Getting info about books.csv data set

book_data.info()
# shows no.of rows and columns in data set
book_data.shape
# similarly as above dataset applying same to "Users.csv" dataset 
users_data= pd.read_csv('Users.csv')
users_data.head()
users_data.info()
users_data.shape
# similarly as above dataset applying same to "Rating.csv" dataset# 
ratings = pd.read_csv("Ratings.csv")
ratings.head()
ratings.info()
ratings.shape
# Data Preparation (Data Cleaning and Feature Engineering)
# Let's take  book_data dataset first
# droping the url
book_data.drop(['Image-URL-S', 'Image-URL-M'], axis= 1, inplace= True)

# replacing '-' with '_' and features name in lower case
book_data.columns= book_data.columns.str.strip().str.lower().str.replace('-', '_')
users_data.columns= users_data.columns.str.strip().str.lower().str.replace('-', '_')
ratings.columns= ratings.columns.str.strip().str.lower().str.replace('-', '_')
pd.set_option('display.max_colwidth', None)

book_data.info()
## Let's see null values in book_data.
print(book_data.isnull().sum())
# nan values in particular column
book_data.loc[(book_data['book_author'].isnull()),: ]
# nan values in particular column
book_data.loc[(book_data['publisher'].isnull()),: ]
### We pointed null values in 'book_author' and 'publisher' feature.

### Let's look at the unique years to realize the time period as this dataset was created in 2004.
# getting unique value from 'year_of_publication' feature
book_data['year_of_publication'].unique()
###  Let's check at the corresponding rows in the dataframe.
# Extracting and fixing mismatch in feature 'year_of_publication', 'publisher', 'book_author', 'book_title'
book_data[book_data['year_of_publication'] == 'DK Publishing Inc'] 
# Extracting and fixing mismatch in feature 'year_of_publication', 'publisher', 'book_author', 'book_title' 
book_data[book_data['year_of_publication'] == 'Gallimard']
### There has to make some correction in three rows as you can see in above output. Let's fix it.
book_data.loc[221678]
book_data.loc[209538]
book_data.loc[220731]
# function to fix mismatch data in feature 'book_title', 'book_author', ' year_of_publication', 'publisher'
def replace_df_value(df, idx, col_name, val):
    df.loc[idx, col_name] = val
    return df
replace_df_value(book_data, 209538, 'book_title', 'DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)')
replace_df_value(book_data, 209538, 'book_author', 'Michael Teitelbaum')
replace_df_value(book_data, 209538, 'year_of_publication', 2000)
replace_df_value(book_data, 209538, 'publisher', 'DK Publishing Inc')

replace_df_value(book_data, 221678, 'book_title', 'DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)')
replace_df_value(book_data, 221678, 'book_author', 'James Buckley')
replace_df_value(book_data, 221678, 'year_of_publication', 2000)
replace_df_value(book_data, 221678, 'publisher', 'DK Publishing Inc')

replace_df_value(book_data, 220731,'book_title', "Peuple du ciel, suivi de 'Les Bergers")
replace_df_value(book_data, 220731, 'book_author', 'Jean-Marie Gustave Le ClÃ?Â©zio')
replace_df_value(book_data, 220731, 'year_of_publication', 2003)
replace_df_value(book_data, 220731, 'publisher', 'Gallimard')
book_data.loc[209538]

book_data.loc[221678]
book_data.loc[220731]
### Now it is been fixed. We can view perfect matching for their corresponding features.
### We've seen there are two missing values in the 'publisher' column. Let's take care of that. As we have seen
book_data.loc[(book_data['publisher'].isnull()),: ]
# replacing 'Nan' with 'No Mention'
book_data.loc[(book_data['isbn'] == '193169656X'),'publisher'] = 'No Mention'
book_data.loc[(book_data['isbn'] == '1931696993'),'publisher'] = 'No Mention'
### df will be a new DataFrame containing only the column(s) at index 4 from the original book_data DataFrame. If you have more columns to select, you can extend the cols list with additional column indices.
df = pd.DataFrame(book_data)
cols = [4]
df = df[df.columns[cols]]
pd.set_option('display.max_columns', None)  
df.head(5)
book_data[book_data['publisher'] == 'No Mention']


# Let's see now user_data dataset
# users_data size
print(users_data.shape)
# unique value in age
users_data['age'].unique()
### There is NaN value in age. We can replace NaN with mean of 'age'.
# replacing nan with average of 'age'
users_data['age'].fillna((users_data['age'].mean()), inplace=True)
users_data['age'].unique()
# retrieving age data between 5 to 90
users_data.loc[(users_data['age'] > 90) | (users_data['age'] < 5)] = np.nan
users_data['age'].fillna((users_data['age'].mean()), inplace=True)
users_data['age'].unique()
#### As we can see above we don't have any null values for age.
# Now let's take ratings_data dataset
ratings.head()
# finding unique ISBNs from rating and book dataset
unique_ratings = ratings[ratings.isbn.isin(book_data.isbn)]
unique_ratings
print(ratings.shape)
print(unique_ratings.shape)
# unique ratings from 'book_rating' feature
unique_ratings['book_rating'].unique()

# Data Visualizations
## Book_data dataset 
#### Question 1 
##### Which are the top Author with number of books ?
plt.figure(figsize=(12,6))
sns.countplot(y="book_author",palette = 'Paired', data=book_data,order=book_data['book_author'].value_counts().index[0:20])
plt.title("Top 20 author with number of books")
### Agatha Christie is leading at top with more than 600 counts, followed by William Shakespeare. We can plot some hypothesis point :-

It can happen in some possible cases that Agatha Christie is not a best Author, though Agatha Christie has most number of books as compared to others.

William Shakespeare is one of the popular Author in the world. Still he doesn't have highest number of books.

Among all other Authors, it might happen that few of the Author might have some of the best seller books who have millions of copies been sold in world.
#### Question 2 
#### Which are top publishers with published books ?
plt.figure(figsize=(12,6))
sns.countplot(y="publisher",palette = 'Paired', data=book_data,order=book_data['publisher'].value_counts().index[0:20])
plt.title("Top 20 Publishers with number of books published")
# Harlequin has most number of books published, followed by Silhouette. Hypothesis analysis to focus :-

Some of the top Author's had published their books from Harlequin.

We can observe Harlequin publiser's marking better performance than any other publishers.

Penguin Books, Warner Books, Penguin USA, Berkely Publishing Group and many more are among popular publisher's remarking competition with Harlequin.

Though Penguin Books Publisher has less number of books published but it might happen that only top Author's are approaching towards Penguin Books Publisher.
## Question 3 
### Number of Books published in yearly. 
publications = {}
for year in book_data['year_of_publication']:
    if str(year) not in publications:
        publications[str(year)] = 0
    publications[str(year)] +=1

publications = {k:v for k, v in sorted(publications.items())}

fig = plt.figure(figsize =(55, 15))
plt.bar(list(publications.keys()),list(publications.values()), color = 'blue')
plt.ylabel("Number of books published")
plt.xlabel("Year of Publication")
plt.title("Number of books published yearly")
plt.margins(x = 0)
plt.show()
book_data.year_of_publication = pd.to_numeric(book_data.year_of_publication, errors='coerce')

# Checking for 0's or NaNs in Year of Publication
zero_year = book_data[book_data.year_of_publication == 0].year_of_publication.count()
nan_year = book_data.year_of_publication.isnull().sum()

print(f'There are {zero_year} entries as \'0\', and {nan_year} NaN entries in the Year of Publication field')

# Replace all years of zero with NaN
book_data.year_of_publication.replace(0, np.nan, inplace=True)
year = book_data.year_of_publication.value_counts().sort_index()
year = year.where(year>5) 
plt.figure(figsize=(10, 8))
plt.rcParams.update({'font.size': 15}) 
plt.bar(year.index, year.values)
plt.xlabel('Year of Publication')
plt.ylabel('counts')
plt.show()
## So we can see publication years are somewhat between 1950 - 2005 here.The publication of books got vital when it starts emerging from 1950. We can get some hyothesis key points:-

It might happen people starts to understand the importance of books and gradually got productivity habits in their life.

Every user has their own taste to read books based on what particular subject Author uses. The subject of writing books got emerge from late 1940 slowly. Till 1970 it has got the opportunity to recommend books to people or users what they love to read.

The highest peak we can observe is between 1995-2001 year. The user understand what they like to read. Looking towards the raise the recommendation is also increase to understand their interest. 
# User_data Dataset
### Question 4 
Age distributions of users_data
plt.figure(figsize=(10,8))
users_data.age.hist(bins=[10*i for i in range(1, 10)], color = 'cyan')     
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
### Looking towards the users age between 30-40 prefer more and somewhat we can also view between 20-30. Let's make some hypothesis analysis:-

It is obvious that most of the user books are from Age 30 to 40.

It might happen that the users are more interested on that subject what Authors are publishing in the market.

The age group between 20-30 are immensely attracted to read books published by Author.

We can observe same pitch for Age group between 10-20 and 50-60. There are can be lot of different reasons. 
# Ratings_data Dataset
### Analysis No. 5

#### What are top 20 books as per number of ratings ? 
plt.figure(figsize=(12,6))
sns.countplot(y="book_title",palette = 'Paired',data= book_data, order=book_data['book_title'].value_counts().index[0:15])
plt.title("Top 20 books as per number of ratings")
As per ratings "Selected Poems" has been rated most followed by "Little Women".
Selected Poems are most favourable to users as per ratings.

Three of the books 'The Secret Garden', 'Dracula','Adventures of Huckleberry Finn'are struggling to compete with each other.

Similarly, we can observe in 'Masquerade','Black Beauty','Frankenstein'.


plt.figure(figsize=(8,6))
sns.countplot(x="book_rating",palette = 'Paired',data= unique_ratings)
### Firstly the above ratings are unique ratings from 'ratings_data' and 'books_data' dataset. We have to separate the explicit ratings represented by 1–10 and implicit ratings represented by 0. Let's make some hypothesis assumptions :-

This countplot shows users have rated 0 the most, which means they haven't rated books at all.

Still we can see pattern to recognize in ratings from 1-10.

Mostly the users have rated 8 ratings out of 10 as per books. It might happen that the feedback is positive but not extremely positive as 10 ratings (i.e best books ever). 
ratings.head(2)
ratings_new = ratings[ratings.isbn.isin(book_data.isbn)]
ratings.shape,ratings_new.shape
Ratings dataset should have ratings from users which exist in users dataset.
print("Shape of dataset before dropping",ratings_new.shape)
ratings_new = ratings_new[ratings_new['user_id'].isin(users_data['user_id'])]
print("shape of dataset after dropping",ratings_new.shape)
It can be seen that no new user was there in ratings dataset.¶
# Let's see how the ratings are distributed
plt.rc("font", size=15)
ratings_new['book_rating'].value_counts(sort=False).plot(kind='bar')
plt.title('Rating Distribution\n')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()
#Hence segragating implicit and explict ratings datasets
ratings_explicit = ratings_new[ratings_new['book_rating'] != 0]
ratings_implicit = ratings_new[ratings_new['book_rating'] == 0]
print('ratings_explicit dataset shape',ratings_explicit.shape)
print('ratings_implicit dataset',ratings_implicit.shape)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(8, 6))
sns.countplot(data=ratings_explicit , x='book_rating',palette = 'Paired')
It can be observe that higher ratings are more common amongst users and rating 8 has been rated highest number of times
Let's find the top 5 books which are rated by most number of users.
rating_count = pd.DataFrame(ratings_explicit.groupby('isbn')['book_rating'].count())
rating_count.sort_values('book_rating', ascending=False).head()
most_rated_books = pd.DataFrame(['0316666343', '0971880107', '0385504209', '0312195516', '0060928336'], index=np.arange(5), columns = ['isbn'])
most_rated_books_summary = pd.merge(most_rated_books, book_data, on='isbn')
most_rated_books_summary
Collaborative filtering

print(book_data.shape, users_data.shape, ratings.shape, sep='\n')
ratings['user_id'].value_counts()
ratings['user_id'].value_counts().shape
ratings['user_id'].unique().shape
# Lets store users who had at least rated more than 200 books
x = ratings['user_id'].value_counts() > 200
x[x].shape
y= x[x].index
y
ratings = ratings[ratings['user_id'].isin(y)]
ratings.head()
ratings.shape
# Now join ratings with books

ratings_with_books = ratings.merge(book_data, on='isbn')
ratings_with_books.head()
ratings_with_books.shape
number_rating = ratings_with_books.groupby('book_title')['book_rating'].count().reset_index()
number_rating.head()
number_rating.rename(columns={'rating':'num_of_rating'},inplace=True)
number_rating.head()
final_rating = ratings_with_books.merge(number_rating, on='book_title')
final_rating.head()
final_rating.shape
# Lets take those books which got at least 50 rating of user

final_rating = final_rating[final_rating['book_rating_y'] >= 50]
final_rating.head()
final_rating.shape
# lets drop the duplicates
final_rating.drop_duplicates(['user_id','book_title'],inplace=True)
final_rating.shape
# Lets create a pivot table
book_pivot = final_rating.pivot_table(columns='user_id', index='book_title', values= 'book_rating_x')
book_pivot
book_pivot.shape
book_pivot.fillna(0, inplace=True)
book_pivot
Training Model
from scipy.sparse import csr_matrix
book_sparse = csr_matrix(book_pivot)
type(book_sparse)
# Now import our clustering algoritm which is Nearest Neighbors this is an unsupervised ml algo
from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(algorithm= 'brute')
model.fit(book_sparse)
distance, suggestion = model.kneighbors(book_pivot.iloc[237,:].values.reshape(1,-1), n_neighbors=6 )
distance
suggestion
book_pivot.iloc[241,:]
for i in range(len(suggestion)):
    print(book_pivot.index[suggestion[i]])
book_pivot.index[4]
#keeping books name
book_names = book_pivot.index
book_names[3]
np.where(book_pivot.index == '4 Blondes')[0][0]
# final_rating['title'].value_counts()
ids = np.where(final_rating['book_title'] == "Harry Potter and the Chamber of Secrets (Book 2)")[0][0]
final_rating.iloc[ids]['image_url_l']
book_name = []
for book_id in suggestion:
    book_name.append(book_pivot.index[book_id])
    
book_name[0]
ids_index = []
for name in book_name[0]: 
    ids = np.where(final_rating['book_title'] == name)[0][0]
    ids_index.append(ids)
for idx in ids_index:
    url = final_rating.iloc[idx]['image_url_l']
    print(url)
import os
if not os.path.exists('artifacts'):
    os.mkdir('artifacts')
import pickle
pickle.dump(model,open('artifacts/model.pkl','wb'))
pickle.dump(book_names,open('artifacts/book_names.pkl','wb'))
pickle.dump(final_rating,open('artifacts/final_rating.pkl','wb'))
pickle.dump(book_pivot,open('artifacts/book_pivot.pkl','wb'))
Testing model

def recommend_book(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6 )
    
    for i in range(len(suggestion)):
            books = book_pivot.index[suggestion[i]]
            for j in books:
                if j == book_name:
                    print(f"You searched '{book_name}'\n")
                    print("The suggestion books are: \n")
                else:
                    print(j)
book_name = "Harry Potter and the Chamber of Secrets (Book 2)"
recommend_book(book_name)


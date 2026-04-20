import pandas as pd

def main():

    # Retrieiving two datasets
    try:
        listingRatings = pd.read_csv('../data/raw/listings.csv')
        textReviews = pd.read_csv('../data/raw/reviews.csv')
    except:
        listingRatings = pd.read_csv('data/raw/listings.csv')
        textReviews = pd.read_csv('data/raw/reviews.csv')
    
    # Extracting the important features
    listingRatings = listingRatings[['id','review_scores_rating']]
    textReviews = textReviews[['listing_id', 'comments']]
    
    # Joining the two tables together using a left join
    RandR = pd.merge(textReviews, listingRatings, left_on = 'listing_id', right_on = 'id', how='left')
    
    # Save in staged folder
    try:
        RandR.to_csv('../data/staged/ratings_and_reviews.csv', index=False)
        print("Data ingestion complete: ratings_and_reviews.csv created")
    except:
        RandR.to_csv('data/staged/ratings_and_reviews.csv', index=False)
        print("Data ingestion complete: ratings_and_reviews.csv created")
    
if __name__ == "__main__":
    main()

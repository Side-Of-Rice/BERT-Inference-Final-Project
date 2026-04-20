import pandas as pd

def main():

    # Import Raw Dataset
    try:
        RandR = pd.read_csv('../data/staged/ratings_and_reviews.csv')
    except:
        RandR = pd.read_csv('data/staged/ratings_and_reviews.csv')
    
    # Some comments are empty. Replace them with the empty string.
    RandR['comments'] = RandR['comments'].fillna('').astype(str)
    
    # Check for Missing Variables
    if(RandR.isnull().values.any()):
        raise Exception("Table has missing values.")
    
    # Feature Engineering: Binary Target as "Top Performer" or "Under Performer" based on rating (rating >= 4.8)
    RandR['performance'] = 'Under'
    for index, row in RandR.iterrows():
        if(row['review_scores_rating'] >= 4.80):
            RandR.loc[index, 'performance'] = 'Top'
            
    # Save comment and performance to cleaned dataset
    
    try:
        RandR[['comments', 'performance']].to_csv('../data/staged/cleaned_ratings_and_reviews.csv')
        print('Dataset is Valid. Saving...')
    except:
        RandR[['comments', 'performance']].to_csv('data/staged/cleaned_ratings_and_reviews.csv')
        print('Dataset is Valid. Saving...')
        
if __name__ == "__main__":
    main()

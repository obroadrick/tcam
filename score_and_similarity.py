import numpy as np
from util import load_dir


def gen_closeness_rankings(path):
    # Load in a single npy file 
    # (which contains two lists, one for similarities in the same hotel and 
    #      another for similarities for images in all the other hotels)
    # same is idx 0, diff idx 1
    similarities = np.load(path, allow_pickle=True)
    same_similarities = np.array(similarities[0])[:, 0]
    diff_similarities = np.array(similarities[1])[:, 0]
    
    combined_similarities = [(float(s), 1) for s in same_similarities]
    combined_similarities.extend( [(float(d), 0) for d in diff_similarities] )
    combined_similarities.sort(key=lambda x: x[0], reverse=True)

    # print(combined_similarities[:25])

    return combined_similarities

   

#query_hotel_dir = 'datasets/0.1k/naive_search_results_diff_ratio/38889' 
query_hotel_dir = 'datasets/0.1k/naive_search_r2d2_scores/38889'

# query_hotel_dir = 'datasets/0.1k/naive_search_results_more_bins/141527' 
hotel_np_files = load_dir(query_hotel_dir, file_type='npy')
query_rid = ' ' 

similarities = []
for idx, file in enumerate(hotel_np_files):
    similarities.append(gen_closeness_rankings(file))

# find avg occurrences of same hotel in top n (for lowish n) similarities
#print('HID=18470')
print('HID=38889')
for n in range(1,25+1):
    proportions = []
    for idx, simils in enumerate(similarities):
        top_n = simils[0:n]
        # if n==3:
        #     print(top_n)
        tot = 0
        for thing in top_n: 
            #recall that the term "thing" refers to a 2-tuple (score, hotel_indicator_integer_zero_or_one)
            tot += thing[1]
        proportions.append(tot / n)

    print('n={}, avg={}'.format(n, sum(proportions) / len(proportions)) )







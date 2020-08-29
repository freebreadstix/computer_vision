import requests
from requests import exceptions
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True, help="search query for Bing API")
ap.add_argument("-o", "--output", required=True,
                help="path to output directory of images")
args = vars(ap.parse_args())

API_KEY = "9c0576da97994eea84ff48d7db719563"
MAX_RESULTS = 250
GROUP_SIZE = 50
# set the endpoint API URL
URL = "https://api.cognitive.microsoft.com/bing/v5.0/images/search"

EXCEPTIONS = set([IOError, FileNotFoundError,
                  exceptions.RequestException, exceptions.HTTPError,
                  exceptions.ConnectionError, exceptions.Timeout])

# Setup parameters for search
term = args['query']
headers = {"Ocp-Apim-Subscription-Key": API_KEY}
params = {"q": term, "offset": 0, "count": GROUP_SIZE}

# Make request to estimate number of results from search API
print("Searching Bing API for {}".format(term))
search = requests.get(URL, headers=headers, params=params)
search.raise_for_status() # Raise HTTPError if one occurred

results = search.json()
estNumResults = min(results['totalEstimatedMatches'], MAX_RESULTS)
print("{} total results for '{}'".format(estNumResults, term))


total = 0

# Make requests for images in Group Size batches (easier to save to disk)
for offset in range(0, estNumResults, GROUP_SIZE):
    # update the search parameters using the current offset, then
    # make the request to fetch the results
    print("[INFO] making request for group {}-{} of {}...".format(
        offset, offset + GROUP_SIZE, estNumResults))
    params["offset"] = offset
    search = requests.get(URL, headers=headers, params=params)
    search.raise_for_status()
    results = search.json()
    print("[INFO] saving images for group {}-{} of {}...".format(
        offset, offset + GROUP_SIZE, estNumResults))

    for v in results["value"]:
        # try to download the image
        try:
            # make a request to download the image
            print("[INFO] fetching: {}".format(v["contentUrl"])) 
            r = requests.get(v["contentUrl"], timeout=30) # Get specific image
            # build the path to the output image
            ext = v["contentUrl"][v["contentUrl"].rfind("."):] # gets file extension i.e. jpg, png
            p = os.path.sep.join([args["output"], "{}{}".format(
                str(total).zfill(8), ext)]) # make file path w/ term/result#/extension
            # write the image to disk
            f = open(p, "wb")
            f.write(r.content)
            f.close()
        # catch any errors that would not unable us to download the
        # image
        except Exception as e:
            # check to see if our exception is in our list of
            # exceptions to check for
            if type(e) in EXCEPTIONS:
                print("[INFO] skipping: {}".format(v["contentUrl"]))
                continue

        image = cv2.imread(p)
        # if the image is `None` then we could not properly load the
        # image from disk (so it should be ignored)
        if image is None:
            print("[INFO] deleting: {}".format(p))
            os.remove(p)
            continue

        # else update the counter
        total += 1

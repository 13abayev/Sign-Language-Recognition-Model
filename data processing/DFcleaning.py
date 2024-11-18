
import pandas as pd
import json

import yt_dlp
import pickle
import json
import os

def loadVoc():
    try:
        with open("datas/videoDictionary.json", "r") as f:
            return json.load(f)
    except:
        return {}
    
    
def saveData(data, name, prefix = "../datas/demo dataset 100/",file_format = "pkl"):
    path = prefix + f"{name}.{file_format}"
    with open(path, "wb") as f:
        pickle.dump(data, f)


def saveVoc(voc):
    with open("../datas/videoDictionary.json", "w") as f:
        return json.dump(voc, f, indent=1)


def downloadVideo(url, _id, save_path="../datas/videos/"):
    options = {
        'format': 'bestvideo[ext=mp4]',
        'outtmpl': f'{save_path}{_id}.mp4',
    }
    
    with yt_dlp.YoutubeDL(options) as ydl:
        try:
            ydl.download([url])
        except Exception as e:
            print(f"Error downloading video: {e}")
            return False
                
    return True


def changeURL(col):
    if col.startswith("www"):
        return "https://" + col
    return col


def getData(path : str, ) -> dict:
    with open(path, "r") as f :
        jsonData = json.load(f)
    
    dataDF = pd.DataFrame(jsonData)
    
    print("Removing irrelevant data...")
    
    dataDF.drop(["org_text", "clean_text", "signer_id", "signer", "review", "height", "width", "box", "file", "start_time", "end_time", "fps", "label"], axis = 1, inplace=True)
    dataDF["url"] = dataDF["url"].apply(changeURL)
    
    voc = loadVoc()
    if voc:
        revVoc = {v : k for k, v in voc.items()}
    
    videosHaveIssue = []
    
    print("Checking the videos...")
    
    for url in dataDF["url"]:
        if url not in revVoc:
            _id = len(revVoc) + 1
            revVoc[url] = _id
            voc[_id] = url
            if not downloadVideo(url, _id):
                videosHaveIssue.append(_id)
        elif not os.path.exists(f"../datas/videos/{revVoc[url]}.mp4"):
            videosHaveIssue.append(revVoc[url])
    
    print(f"{len(videosHaveIssue)} video is not available")
    print("Removing not available videos...")
    
    dataDF["VideoID"] = [revVoc[url] for url in dataDF["url"]]
    dataDF = dataDF[~dataDF["VideoID"].isin(videosHaveIssue)]
    
    dataDF.drop("url", axis = 1, inplace = True)
    
    dataDF["duration"] = dataDF["end"] - dataDF["start"] + 1
    
    saveVoc(voc)
    print("Done")
    return dataDF
            

trainDF = getData("../datas/dataset/MSASL_train.json")
print(trainDF.head())


testDF = getData("../datas/dataset/MSASL_test.json")
print(testDF.head())


valDF = getData("../datas/dataset/MSASL_val.json")
print(valDF.head())


# demo_words = [ # 10 words
#     "hello",
#     "thanks",
#     "please",
#     "sorry",
#     "yes",
#     "no",
#     "bye",
#     "help",
#     "eat", 
#     "drink"
# ]

# demo_words = [  # 100 words
#   "hello", "teacher", "school", "friend", "mother", "student", "learn", "spring", "good", "table",
#   "father", "brother", "sister", "book", "girl", "boy", "family", "water", "computer", "help",
#   "doctor", "write", "hungry", "understand", "beautiful", "blue", "green", "english", "name", "you",
#   "same", "day", "now", "thanks", "grandmother", "walk", "read", "dance", "play", "sign",
#   "big", "work", "woman", "know", "live", "night", "apple", "home", "coffee", "phone",
#   "teach", "light", "uncle", "cheese", "party", "jacket", "movie", "daughter", "future", "love",
#   "son", "baby", "remember", "restaurant", "early", "candy", "kitchen", "hour", "start", "house",
#   "cold", "soccer", "mean", "practice", "car", "cookie", "enjoy", "study", "see", "door",
#   "week", "busy", "money", "morning", "shop", "old", "homework", "vacation", "lunch", "fast",
#   "newspaper", "children", "flower", "month", "number", "teacher", "baby", "phone", "movie", "library"
# ]
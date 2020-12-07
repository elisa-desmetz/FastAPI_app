from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import os
import re
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from stop_words import get_stop_words

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def check_msg(text: str) -> str:
    """
    Check the offensivity level of a text

    Parameter:
    text (str): string to analyse

    Returns:
    str : level of offensivity
    """
    clf = load('static/clf.joblib')
    data_input=[re.sub("[^A-Za-z']+", ' ', text.lower())]
    predict=clf.predict(data_input)
    if predict==0:
        return "Your message contains : Hate speech."
    elif predict==1:
        return "Your message contains : Offensive language."
    else:
        return "Your message doesn't contain hate speech neither offensive language."

@app.get("/", response_class=HTMLResponse)
def root(request:Request):
    return templates.TemplateResponse("index.html",{"request": request})

@app.post("/predict/", response_class=HTMLResponse)
async def predict(request:Request, text: str = Form(...)):
    result = check_msg(text)
    return templates.TemplateResponse("predict.html",{"request": request, "text":text, "result":result})

@app.get("/redirect/{url:str}")
async def redirection(request:Request,url=None):
    """
    Redirect to other pages
    """
    url=url
    if url=="portfolio":
        return RedirectResponse(url="http://my-portfolio-edesmetz.herokuapp.com/") 
    elif url=="github":
        return RedirectResponse(url="https://github.com/elisa-desmetz") 
    else :
        return RedirectResponse(url="/") 
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from model import  preprocessing, load_sentiment_analyzer, load_summarizer, predict_sentiment, summarize, train

app = FastAPI(title="Sentiment + Summary API")

sentiment_model, sentiment_tokenizer = load_sentiment_analyzer("./saved/model_state.pth")
summarizer_model, summarizer_tokenizer = load_summarizer()

class TextRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Sentiment and Summary Generator"}


@app.post("/training/")
async def training(background_tasks: BackgroundTasks):
    background_tasks.add_task(train)
    return {"message": "Training started..."}

@app.post("/predict/")
def predict(request: TextRequest):
    cleaned = preprocessing(request.text)
    sentiment = predict_sentiment(cleaned, sentiment_model, sentiment_tokenizer)
    summary = summarize(cleaned, summarizer_model, summarizer_tokenizer)
    return {"input_text": request.text,"predicted_sentiment": sentiment,"summary_insight": summary }

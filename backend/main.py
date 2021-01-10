import uvicorn
from fastapi import FastAPI
from predict import predict_price


app = FastAPI()
origins = [
    "http://localhost:8000",
    "localhost:8000"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get('/')
def index():
    return {'message': 'Welcome to the Machine Price Predictor!'}


@app.get('/{ID}')
def return_predict_price(ID: int): -> dict
    print(f'Evaluating ID {ID}...')
    prediction = round(predict_price(ID=ID))
    print(f'Predicted price: {prediction}')
    return {
        'prediction': prediction,
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, reload=True)